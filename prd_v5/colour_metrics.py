"""Colour-reference measurements with explicit caller-attested calibration."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile
from skimage.color import deltaE_ciede2000, rgb2lab

from prd_v5.comparison import validate_region


def calibrated_delta_e(
    candidate: str | Path,
    reference: str | Path | None,
    *,
    calibration: dict | None = None,
    region: tuple[int, int, int, int] | None = None,
    strip_rows: int = 64,
) -> dict:
    """Measure CIEDE2000 on trusted-development, aligned uint16 sRGB TIFFs.

    Calibration is caller evidence, not a calibration inferred from pixels. It
    must identify the camera/profile pair, trusted RAW development and alignment
    procedure. Both rasters must already encode sRGB, D65, 2-degree observer.
    No registration, exposure fitting, chromatic adaptation or ICC conversion
    occurs. Only bounded strips are converted to Lab; no full error map is kept.
    """
    if type(strip_rows) is not int or strip_rows < 1:
        raise ValueError("strip_rows must be a positive integer")
    validate_region(region)
    report = {
        "status": "not measured",
        "metric": "CIEDE2000; kL=kC=kH=1",
        "domain": "encoded sRGB to CIELAB; D65, 2-degree observer",
        "limitations": "Caller-attested calibration/alignment; no recovered RAW or HDR truth claim",
    }
    if reference is None:
        return {**report, "reason": "No aligned calibrated reference supplied"}
    if calibration is None:
        return {**report, "reason": "No calibration contract supplied"}
    required = {"camera_profile", "trusted_development", "alignment", "colour_space"}
    if not isinstance(calibration, dict) or set(calibration) != required:
        raise ValueError(f"calibration requires exactly {sorted(required)}")
    if any(not isinstance(value, str) or not value.strip() for value in calibration.values()):
        raise ValueError("calibration evidence must contain nonempty strings")
    if calibration["colour_space"] != "sRGB_D65_2deg":
        raise ValueError("Only explicitly calibrated sRGB_D65_2deg is supported")
    report["calibration"] = dict(calibration)
    with ExitStack() as stack:
        arrays = []
        for path in (candidate, reference):
            array = tifffile.memmap(path, mode="r")
            stack.callback(array._mmap.close)
            if array.dtype != np.uint16 or array.ndim != 3 or array.shape[2] != 3 or not all(array.shape):
                raise ValueError("Colour comparison requires nonempty uint16 RGB TIFFs")
            arrays.append(array)
        if arrays[0].shape != arrays[1].shape:
            return {**report, "reason": "Unequal rasters; no implicit registration or resampling"}
        top, left, bottom, right = region or (0, 0, *arrays[0].shape[:2])
        if bottom > arrays[0].shape[0] or right > arrays[0].shape[1]:
            raise ValueError("region exceeds reference raster")
        count = 0
        total = squared = maximum = 0.0
        for start in range(top, bottom, strip_rows):
            labs = [
                rgb2lab(
                    array[start : min(start + strip_rows, bottom), left:right].astype(np.float64) / 65535,
                    illuminant="D65",
                    observer="2",
                )
                for array in arrays
            ]
            errors = deltaE_ciede2000(labs[1], labs[0], kL=1, kC=1, kH=1)
            count += errors.size
            total += float(errors.sum())
            squared += float(np.sum(errors**2))
            maximum = max(maximum, float(errors.max()))
        return {
            **report,
            "status": "measured reference colour difference",
            "region": [top, left, bottom, right],
            "pixels": count,
            "mean_delta_e2000": total / count,
            "rms_delta_e2000": float(np.sqrt(squared / count)),
            "max_delta_e2000": maximum,
        }
