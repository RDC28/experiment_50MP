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
    # (LibRaw ignores the tag), no auto exposure or automatic orientation.
    # Development is a render, not linear scene truth.
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


def develop_exposure_sweep(
    dng_path: str | Path,
    output_directory: str | Path,
    *,
    baseline_exposure_ev: float = 0.0,
    max_working_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """Develop five linear-light brightness edits sequentially from one DNG.

    This uses LibRaw brightness scaling, including -3 EV which is outside
    rawpy's documented exp_shift range. It measures developer response, not
    sensor exposure changes or recovered dynamic range. No automatic exposure,
    highlight reconstruction, or output transfer curve obscures the scaling.
    A fresh output directory is required and removed on any failed sweep.
    """
    import math
    import shutil

    import rawpy

    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("max_working_bytes must be a positive integer")
    if isinstance(baseline_exposure_ev, bool) or not isinstance(baseline_exposure_ev, (int, float)):
        raise ValueError("baseline_exposure_ev must be finite and between -16 and 16")
    if not math.isfinite(baseline_exposure_ev) or abs(baseline_exposure_ev) > 16:
        raise ValueError("baseline_exposure_ev must be finite and between -16 and 16")
    source, destination = Path(dng_path), Path(output_directory)
    entries = []
    # Admission precedes directory creation and pixel unpacking.
    with rawpy.RawPy() as header:
        header.open_file(str(source))
        estimate = header.sizes.raw_width * header.sizes.raw_height * 64
    if estimate > max_working_bytes:
        raise ValueError("LibRaw exposure sweep exceeds the configured memory budget")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for ev in (-3, -2, 0, 2, 3):
            with rawpy.imread(str(source)) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    bright=float(2 ** (baseline_exposure_ev + ev)),
                    user_flip=0,
                    output_color=rawpy.ColorSpace.sRGB,
                    gamma=(1, 1),
                    highlight_mode=rawpy.HighlightMode.Clip,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                )
                expected_shape = (raw.sizes.height, raw.sizes.width, 3)
            if rgb.dtype != np.uint16 or rgb.shape != expected_shape:
                raise HardValidationError("Unexpected exposure sweep shape or precision")
            path = destination / f"exposure_{ev:+d}EV.tiff"
            tifffile.imwrite(path, rgb, photometric="rgb", compression=None, metadata=None)
            entries.append(
                {
                    "ev": ev,
                    "brightness_multiplier": float(2 ** (baseline_exposure_ev + ev)),
                    "path": str(path),
                    "shape": list(rgb.shape),
                    "mean_linear_code": float(np.mean(rgb, dtype=np.float64)),
                    "zero_samples": int(np.count_nonzero(rgb == 0)),
                    "saturated_samples": int(np.count_nonzero(rgb == 65535)),
                }
            )
            del rgb
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {
        "status": "measured",
        "method": "LibRaw linear-light brightness sweep; camera WB, linear demosaic, highlight clipping",
        "baseline_exposure_ev": baseline_exposure_ev,
        "working_bytes_estimate": estimate,
        "renders": entries,
        "highlight_recovery": "not measured",
        "physical_dynamic_range": "not measured",
        "limitations": "Developer edits only; unchanged sensor samples, no calibrated truth or visual acceptance",
    }


def develop_wb_sweep(
    dng_path: str | Path,
    output_directory: str | Path,
    *,
    baseline_exposure_ev: float = 0.0,
    strength: float = 1.25,
    max_working_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """Export relative cool/neutral/warm WB edits of unchanged DNG samples.

    Neutral uses the DNG camera multipliers, normalized to green. Cool divides
    red and multiplies blue by strength; warm does the reverse. These are
    explicit relative gains, not calibrated colour temperatures or illuminants.
    """
    import math
    import shutil

    import rawpy

    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("max_working_bytes must be a positive integer")
    for name, value, low, high in (
        ("baseline_exposure_ev", baseline_exposure_ev, -16, 16),
        ("strength", strength, 1.000001, 4),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite and between {low} and {high}")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and between {low} and {high}")
    source, destination = Path(dng_path), Path(output_directory)
    with rawpy.RawPy() as header:
        header.open_file(str(source))
        estimate = header.sizes.raw_width * header.sizes.raw_height * 64
        gains = np.asarray(header.camera_whitebalance, dtype=np.float64)
    if estimate > max_working_bytes:
        raise ValueError("LibRaw WB sweep exceeds the configured memory budget")
    if gains.shape != (4,) or not np.isfinite(gains).all() or np.any(gains[:3] <= 0) or gains[3] < 0:
        raise HardValidationError("Invalid camera white balance for WB sweep")
    # LibRaw may leave the second green multiplier zero for a three-colour CFA.
    if gains[3] == 0:
        gains[3] = gains[1]
    gains /= gains[1]
    edits = (
        ("cool", gains * (1 / strength, 1, strength, 1)),
        ("neutral", gains.copy()),
        ("warm", gains * (strength, 1, 1 / strength, 1)),
    )
    entries = []
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, wb in edits:
            with rawpy.imread(str(source)) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    use_camera_wb=False,
                    use_auto_wb=False,
                    user_wb=wb.tolist(),
                    no_auto_bright=True,
                    bright=float(2**baseline_exposure_ev),
                    user_flip=0,
                    output_color=rawpy.ColorSpace.sRGB,
                    gamma=(1, 1),
                    highlight_mode=rawpy.HighlightMode.Clip,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                )
                expected_shape = (raw.sizes.height, raw.sizes.width, 3)
            if rgb.dtype != np.uint16 or rgb.shape != expected_shape:
                raise HardValidationError("Unexpected WB sweep shape or precision")
            path = destination / f"wb_{name}.tiff"
            tifffile.imwrite(path, rgb, photometric="rgb", compression=None, metadata=None)
            entries.append(
                {
                    "name": name,
                    "user_wb": wb.tolist(),
                    "path": str(path),
                    "shape": list(rgb.shape),
                    "mean_linear_rgb_code": np.mean(rgb, axis=(0, 1), dtype=np.float64).tolist(),
                    "zero_samples": int(np.count_nonzero(rgb == 0)),
                    "saturated_samples": int(np.count_nonzero(rgb == 65535)),
                }
            )
            del rgb
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {
        "status": "measured",
        "method": "LibRaw relative camera-WB gains; linear light, linear demosaic, no automatic brightness",
        "baseline_exposure_ev": baseline_exposure_ev,
        "strength": strength,
        "working_bytes_estimate": estimate,
        "renders": entries,
        "white_balance_accuracy": "not measured",
        "limitations": "Relative developer edits only; no Kelvin calibration, illuminant truth or visual acceptance",
    }


def develop_demosaic_sweep(
    dng_path: str | Path,
    output_directory: str | Path,
    *,
    baseline_exposure_ev: float = 0.0,
    max_working_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """Sequential LINEAR/VNG/AHD renders with otherwise identical development.

    Unavailable LibRaw algorithms are visibly skipped. Successful development
    measures algorithm response only, without certifying detail or editability.
    """
    import math
    import shutil

    import rawpy

    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("max_working_bytes must be a positive integer")
    if (
        isinstance(baseline_exposure_ev, bool)
        or not isinstance(baseline_exposure_ev, (int, float))
        or not math.isfinite(baseline_exposure_ev)
        or abs(baseline_exposure_ev) > 16
    ):
        raise ValueError("baseline_exposure_ev must be finite and between -16 and 16")
    source, destination = Path(dng_path), Path(output_directory)
    with rawpy.RawPy() as header:
        header.open_file(str(source))
        estimate = header.sizes.raw_width * header.sizes.raw_height * 64
    if estimate > max_working_bytes:
        raise ValueError("LibRaw demosaic sweep exceeds the configured memory budget")
    entries, skipped = [], []
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("LINEAR", "VNG", "AHD"):
            algorithm = rawpy.DemosaicAlgorithm[name]
            if not algorithm.isSupported:
                skipped.append({"algorithm": name, "reason": "Unavailable in this LibRaw build"})
                continue
            with rawpy.imread(str(source)) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    use_camera_wb=True,
                    use_auto_wb=False,
                    no_auto_bright=True,
                    bright=float(2**baseline_exposure_ev),
                    user_flip=0,
                    output_color=rawpy.ColorSpace.sRGB,
                    gamma=(1, 1),
                    highlight_mode=rawpy.HighlightMode.Clip,
                    demosaic_algorithm=algorithm,
                )
                expected_shape = (raw.sizes.height, raw.sizes.width, 3)
            if rgb.dtype != np.uint16 or rgb.shape != expected_shape:
                raise HardValidationError("Unexpected demosaic sweep shape or precision")
            path = destination / f"demosaic_{name.lower()}.tiff"
            tifffile.imwrite(path, rgb, photometric="rgb", compression=None, metadata=None)
            entries.append(
                {
                    "algorithm": name,
                    "path": str(path),
                    "shape": list(rgb.shape),
                    "mean_linear_rgb_code": np.mean(rgb, axis=(0, 1), dtype=np.float64).tolist(),
                    "zero_samples": int(np.count_nonzero(rgb == 0)),
                    "saturated_samples": int(np.count_nonzero(rgb == 65535)),
                }
            )
            del rgb
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {
        "status": "measured" if entries else "not measured",
        "method": "LibRaw demosaic sweep; camera WB, linear light, fixed highlight clipping, no auto brightness",
        "baseline_exposure_ev": baseline_exposure_ev,
        "working_bytes_estimate": estimate,
        "rawpy_version": rawpy.__version__,
        "libraw_version": rawpy.libraw_version,
        "renders": entries,
        "skipped": skipped,
        "detail_accuracy": "not measured",
        "limitations": "Developer response only; unchanged RAW, no ground truth or visual editability acceptance",
    }
