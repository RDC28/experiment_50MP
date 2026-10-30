"""Independent LibRaw acceptance and explicitly memory-gated TIFF development."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from prd_v5.analytic import AnalyticISP
from prd_v5.dng import build_metadata
from prd_v5.geometry import GeometryPlan
from prd_v5.hypothesis import CameraHypothesis
from synthraw.libraw_validator import HardValidationError, LibRawValidator


def develop_roundtrip(
    dng_path: str | Path,
    tiff_path: str | Path,
    mosaic: np.ndarray,
    camera: CameraHypothesis,
    geometry: GeometryPlan,
    isp: AnalyticISP,
    *,
    max_working_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """Validate independent RAW samples/metadata and export a 16-bit sRGB TIFF.

    LibRaw uses whole-frame buffers, unlike the strip-based synthetic stages.
    The conservative 64 bytes/photosite admission estimate is not a peak RSS
    guarantee. Colab callers must explicitly raise the budget for 50 MP. Caller
    discards partial TIFFs on failure. No calibration/truth acceptance is implied.
    """
    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("max_working_bytes must be a positive integer")
    estimate = geometry.width * geometry.height * 64
    if estimate > max_working_bytes:
        raise ValueError("LibRaw development exceeds the configured memory budget")
    source, destination = Path(dng_path), Path(tiff_path)
    if destination.resolve() == source.resolve():
        raise ValueError("TIFF destination must differ from DNG source")
    records = build_metadata(camera, geometry, isp)
    LibRawValidator().validate(source, mosaic, records)

    import rawpy

    # Deterministic baseline rendering: camera WB, sRGB transfer, DNG BaselineExposure
    # (LibRaw ignores the tag), no auto exposure or automatic orientation. Development is a render, not linear scene truth.
    with rawpy.imread(str(source)) as raw:
        if raw.sizes.flip != 0:
            raise HardValidationError("Unexpected RAW orientation")
        visible_shape = (raw.sizes.height, raw.sizes.width, 3)
        rgb = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
            bright=float(2**isp.baseline_exposure_ev),
            user_flip=0,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(2.4, 12.92),
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
        )
    if rgb.dtype != np.uint16 or rgb.shape != visible_shape:
        raise HardValidationError("Unexpected developed RGB shape or precision")
    tifffile.imwrite(destination, rgb, photometric="rgb", compression=None, metadata=None)
    reopened = tifffile.memmap(destination, mode="r")
    try:
        if reopened.shape != rgb.shape or reopened.dtype != rgb.dtype:
            raise HardValidationError("TIFF shape or precision differs")
        for start in range(0, rgb.shape[0], 64):
            if not np.array_equal(reopened[start : start + 64], rgb[start : start + 64]):
                raise HardValidationError("TIFF samples differ")
    finally:
        reopened._mmap.close()
    return {
        "dng_path": str(source),
        "tiff_path": str(destination),
        "independent_rawpy_validation": "passed: samples, CFA, levels, WB, colour matrix, dimensions",
        "tiff_roundtrip": "passed: exact uint16 RGB samples",
        "rawpy_version": rawpy.__version__,
        "libraw_version": rawpy.libraw_version,
        "render": "sRGB, camera WB, linear demosaic, no automatic brightness or orientation",
        "rgb_shape": list(rgb.shape),
        "default_crop": list(geometry.default_crop_size),
        "crop_policy": "LibRaw visible raster; DefaultCrop remains DNG metadata, not applied to TIFF",
        "working_bytes_estimate": estimate,
        "adobe_conformance": "not measured",
        "scientific_colour_accuracy": "not measured",
    }
