"""Bridge the v5 hypothesis to the existing bounded, uncompressed DNG writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prd_v5.analytic import AnalyticISP
from prd_v5.geometry import GeometryPlan
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.runtime import software_identity
from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import DNGFieldRecord, DNGMetadataBuilder
from synthraw.dng_writer import DNGWriter
from synthraw.provenance import FieldRecord, Provenance

# IEC sRGB linear RGB -> CIE XYZ, D65. This is a virtual profile, not Sony calibration.
SRGB_TO_XYZ_D65 = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)


def build_metadata(camera: CameraHypothesis, geometry: GeometryPlan, isp: AnalyticISP) -> dict[str, DNGFieldRecord]:
    """Derive container fields from exactly the state used for pixel simulation.

    Both illuminants use the same virtual D65 profile. No dual-illuminant camera
    calibration, original capture identity or independent scientific validation
    is implied. Exposure gain/tone parameters remain inverse-ISP manifest fields,
    not invented physical exposure estimates; BaselineExposure only records HDR
    headroom. Returned records use the legacy
    writer's serialization contract, not the complete v5 provenance registry.
    """
    config = ExperimentConfig(ADC_BITS=14 if camera.adc_bits <= 14 else 16, DNG_COMPRESSION="UNCOMPRESSED")
    xyz_to_camera = np.linalg.inv(np.asarray(SRGB_TO_XYZ_D65) @ np.asarray(isp.camera_to_srgb))
    neutral = np.linalg.solve(np.asarray(isp.camera_to_srgb), np.ones(3)) / np.asarray(isp.white_balance)
    if not np.all(neutral > 0):
        raise ValueError("camera_to_srgb maps sRGB white outside the positive camera gamut")
    neutral /= neutral[1]
    exposure = {
        name: FieldRecord(name, value, Provenance.FABRICATED, "Explicit v5 synthetic camera hypothesis")
        for name, value in (
            ("ExposureTime", camera.shutter_s),
            ("FNumber", camera.aperture),
            ("PhotographicSensitivity", camera.iso),
        )
    }
    records = DNGMetadataBuilder().build(
        geometry,
        config,
        cfa_phase=camera.cfa_pattern,
        black_level=camera.black_level,
        white_level=camera.white_level,
        color_matrix1=xyz_to_camera,
        color_matrix2=xyz_to_camera,
        as_shot_neutral=tuple(neutral),
        noise_profile=camera.noise_profile,
        software_version=software_identity(),
        exposure=exposure,
    )
    records["Model"].value = f"{camera.profile} Synthetic Sensor"
    records["BaselineExposure"].value = isp.baseline_exposure_ev
    records["BaselineExposure"].method = "Restores 18% grey after analytic HDR headroom scaling"
    records["UniqueCameraModel"].value = f"Synthetic PRD v5 {camera.profile}"
    for index in (1, 2):
        records[f"CalibrationIlluminant{index}"].value = 21
        records[f"ColorMatrix{index}"].method = "Virtual D65 XYZ-to-camera derived from analytic camera_to_srgb"
    DNGMetadataBuilder().validate(records, config)
    return records


def write_dng(
    path: str | Path,
    mosaic: np.ndarray,
    camera: CameraHypothesis,
    geometry: GeometryPlan,
    isp: AnalyticISP,
    *,
    strip_rows: int = 64,
) -> dict:
    """Write and structurally verify exact samples using bounded row strips.

    Caller owns the mosaic and output cleanup on failure. A passed structural
    check does not establish independent LibRaw decode or Adobe conformance.
    """
    records = build_metadata(camera, geometry, isp)
    writer_config = ExperimentConfig(ADC_BITS=14 if camera.adc_bits <= 14 else 16, DNG_COMPRESSION="UNCOMPRESSED")
    destination = DNGWriter().write(path, mosaic, records, writer_config, rows_per_strip=strip_rows)
    return {
        "path": str(destination),
        "structural_validation": "passed: serialized tags and exact samples",
        "independent_rawpy_validation": "not measured",
        "adobe_conformance": "not measured",
        "compression": "UNCOMPRESSED",
        "adc_bits": camera.adc_bits,
        "storage_bits": 16,
        "profile": "virtual D65; not calibrated Sony",
        "tags": {name: record.to_dict() for name, record in records.items()},
    }
