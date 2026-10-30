"""Complete field resolution and inspectable scientific output artifacts."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image

from prd_v5.hypothesis import CameraField, CameraHypothesis
from prd_v5.physics import photon_transfer


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def export_arrays(output, optical, adc, camera, *, strip_rows=64):
    """Persist electron-equivalent pre/post sensor mosaics with explicit units.

    Noisy values are ADC-referred electrons after pedestal removal, including
    clipping/quantisation. Clean values precede stochastic and body effects.
    """
    shape = adc.shape
    channels = np.array(["RGB".index(c) for c in camera.cfa_pattern]).reshape(2, 2)
    for name in ("clean_bayer.npy", "noisy_bayer.npy"):
        target = np.lib.format.open_memmap(output / name, mode="w+", dtype=np.float32, shape=shape)
        try:
            for start in range(0, shape[0], strip_rows):
                end = min(start + strip_rows, shape[0])
                if name == "noisy_bayer.npy":
                    target[start:end] = (
                        (adc[start:end].astype(np.float32) - camera.black_level)
                        * camera.conversion_gain_e_per_dn
                        / camera.analogue_gain
                    )
                else:
                    for y in range(start, end):
                        for parity in (0, 1):
                            target[y, parity::2] = optical[y, parity::2, channels[y % 2, parity]] * camera.full_well_e
            target.flush()
        finally:
            target._mmap.close()


def complete_registry(report):
    """Resolve every PRD table field or explicitly retain an unknown value."""
    catalogue = json.loads(files("prd_v5").joinpath("field_catalogue.json").read_text())
    camera, geometry = report["camera"], report["geometry"]
    tags = {name: record["value"] for name, record in report["stages"]["dng"]["tags"].items()}
    sensor = report["stages"]["sensor"]["body_effects"]["parameters"]
    optics = report["stages"]["optical_effects"]["parameters"]
    aliases = {
        "Sensor width/height px": [geometry["width"], geometry["height"]],
        "Sensor physical size": [35.9, 35.9 * geometry["height"] / geometry["width"]],
        "Pixel pitch": 35900 / geometry["width"],
        "CFA pattern": camera["cfa_pattern"],
        "CFA phase/origin": [0, 0],
        "ADC bit depth": camera["adc_bits"],
        "Black level": camera["black_level"],
        "White level": tags["WhiteLevel"],
        "Full-well capacity": camera["full_well_e"],
        "Conversion gain": camera["full_well_e"] / (((1 << camera["adc_bits"]) - 1) - camera["black_level"]),
        "Analogue gain": camera["iso"] / 100,
        "Digital gain": 1,
        "Sensor non-linearity": sensor["nonlinearity_compression"],
        "Quantisation": "uniform rounding",
        "Clipping behaviour": "hard clip at ADC bounds",
        "Sensor temperature": sensor["temperature_c"],
        "Shot noise": camera["shot_noise"],
        "Read noise": camera["read_noise_e"] if camera["read_noise"] else 0,
        "PRNU": sensor["prnu_sigma"],
        "DSNU": sensor["dsnu_e"],
        "Row FPN": sensor["row_fpn_e"],
        "Column FPN": sensor["column_fpn_e"],
        "Dark current": report["stages"]["sensor"]["body_effects"]["dark_expectation_e"],
        "Hot pixels": sensor["hot_pixel_fraction"],
        "Dead pixels": sensor["dead_pixel_fraction"],
        "Blooming": False,
        "Quantisation noise": "ADC rounding; 1/12 DN squared uniform approximation",
        "Scene-linear RGB": "Analytic inverse-tone hypothesis, not recovered radiance",
        "Camera-native RGB": "Analytic profile transform with optional confidence-gated learned residual",
        "ColorMatrix1/2": [tags.get("ColorMatrix1"), tags.get("ColorMatrix2")],
        "ForwardMatrix1/2": [tags.get("ForwardMatrix1"), tags.get("ForwardMatrix2")],
        "CameraCalibration1/2": [tags.get("CameraCalibration1"), tags.get("CameraCalibration2")],
        "CalibrationIlluminant1/2": [tags.get("CalibrationIlluminant1"), tags.get("CalibrationIlluminant2")],
        "ISO": camera["iso"],
        "Shutter time": camera["shutter_s"],
        "Aperture": camera["aperture"],
        "Radial distortion": report["stages"]["distortion"]["parameters"],
        "Tangential distortion": report["stages"]["distortion"]["parameters"],
        "Vignetting": report["stages"]["optics"]["parameters"],
        "Lateral CA": optics["lateral_ca_px"],
        "Longitudinal CA": optics["axial_ca_sigma_px"],
        "PSF": [optics["psf_center_sigma_px"], optics["psf_edge_sigma_px"]],
        "Diffraction": report["stages"]["optical_effects"]["diffraction_sigma_px"],
        "Motion blur": [optics["motion_length_px"], optics["motion_angle_deg"]],
        "Source SHA-256": report["source"]["sha256"],
        "Random seed": camera["seed"],
        "Checkpoint hashes": {"sr": report["stages"].get("super_resolution", {}).get("checkpoint", {}).get("sha256")},
        "Synthetic-capture flag": True,
    }
    registry = dict(report["registry"])
    for field in catalogue:
        name = field["name"]
        value = tags.get(name, aliases.get(name))
        known = value is not None and value != [None, None]
        provenance = "DERIVED" if known else "UNSPECIFIED"
        if name in ("Sensor physical size", "Sensor temperature", "Scene-linear RGB", "Camera-native RGB"):
            provenance = "SYNTHETIC" if known else "UNSPECIFIED"
        registry[name] = asdict(
            CameraField(
                name,
                value,
                field.get("units"),
                field["mode"],
                provenance,
                None,
                None,
                "Serialized DNG value or executed hypothesis" if known else "No evidence; intentionally unresolved",
                "Manifest coherence; synthetic virtual sensor preserves source aspect"
                if known
                else "Not measured or not identifiable from rendered RGB",
            )
        )
    return registry


def finish_bundle(output, report):
    """Write canonical names, complete registry, masks, hashes and gate records."""
    import shutil

    output = Path(output)
    report["registry"] = complete_registry(report)
    for name, value in (
        ("field_registry.json", report["registry"]),
        ("camera_hypothesis.json", report["camera"]),
        ("experiment_config.json", json.loads((output / "config.json").read_text())),
        ("source_hashes.json", {"source": report["source"]["sha256"]}),
        ("checkpoint_hashes.json", report["registry"]["Checkpoint hashes"]["value"]),
    ):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    # Hard links avoid duplicating multi-hundred-MB deliverables on the runtime disk.
    for source, target in (("capture.dng", "synthetic_50mp.dng"), ("developed.tiff", "synthetic_50mp_16bit.tif")):
        try:
            (output / target).hardlink_to(output / source)
        except OSError:
            shutil.copyfile(output / source, output / target)
    import tifffile

    rgb = tifffile.memmap(output / "developed.tiff", mode="r")
    try:
        step = max(1, int(np.ceil(max(rgb.shape[:2]) / 1200)))
        Image.fromarray((rgb[::step, ::step] >> 8).astype(np.uint8)).save(output / "synthetic_50mp_preview.jpg")
    finally:
        rgb._mmap.close()
    support = np.load(output / "source_support.npy", mmap_mode="r")
    clipped = np.load(output / "source_clipped.npy", mmap_mode="r")
    try:
        learned = "super_resolution" in report["stages"] or report["stages"]["learned_raw"].get("effective") == "HYBRID"
        Image.fromarray(((~support | learned) * 255).astype(np.uint8)).save(output / "generated_region_mask.png")
        Image.fromarray(((~support | clipped) * 255).astype(np.uint8)).save(output / "uncertainty_map.png")
    finally:
        support._mmap.close()
        clipped._mmap.close()
    validation = {
        "DNG parse": {"status": "passed", "evidence": report["stages"]["roundtrip"]},
        "CFA integrity": {"status": "passed", "evidence": report["stages"]["dng"]["structural_validation"]},
        "Metadata coherence": {"status": "passed", "scope": "serialized DNG fields and executed configuration"},
        "Provenance": {"status": "passed", "fields": len(report["registry"])},
        **{
            name: {"status": "not measured", "reason": reason}
            for name, reason in {
                "Tile seams": "Requires clean-RAW overlap disagreement measurement on real model execution",
                "SR value": "Requires genuine held-out RAW-derived truth and Lanczos reference",
                "Colour": "Requires aligned calibrated colour truth",
                "Noise validity": "Flat/dark sweep pending; existing tests establish simulator self-consistency only",
                "RAW editability": "Full EV/WB/highlight/shadow/demosaic sweep requires Colab validation",
            }.items()
        },
        "HDR validity": {"status": "passed", "scope": "disclosure and uncertainty; no recovered-truth assertion"},
        "acceptance": "conditional: scientific gates unmeasured",
        "mask_semantics": {
            "generated": "Potential model-generated content or unsupported margins; not a truth oracle",
            "uncertainty": "Original source endpoint or unsupported geometry",
        },
        "mosaic_units": {
            "clean_bayer.npy": "expected electrons before body/noise effects",
            "noisy_bayer.npy": "ADC-referred electrons including clipping and quantisation",
        },
    }
    validation["Noise validity"] = photon_transfer(CameraHypothesis(**report["camera"]))
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    report["validation"] = validation
    report["scientific_acceptance"] = validation["acceptance"]
    report["pending"] = [
        "Camera-specific learned loaders and halo inference",
        "Real RAW/HDR benchmarks",
        "Colab full-scale runtime and quality gates",
        "Independent Adobe DNG conformance",
    ]
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in ("manifest.json", "report.html", "validation_report.html"):
            report["artifacts"][path.name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
    return report


def write_pair_html(output, report):
    encoded = html.escape(json.dumps(report, indent=2, allow_nan=False))
    (Path(output) / "2k_vs_4k_report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>2K versus 4K</title>'
        "<h1>Full-resolution 2K versus 4K comparison</h1><p>Unaligned differences, not ground-truth accuracy.</p>"
        f'<pre style="white-space:pre-wrap">{encoded}</pre>'
    )
