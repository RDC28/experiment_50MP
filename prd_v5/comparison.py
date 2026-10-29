"""Bounded full-resolution render disagreement, without registration or truth claims."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile


def compare_renders(
    first: str | Path,
    second: str | Path,
    *,
    strip_rows: int = 64,
    region: tuple[int, int, int, int] | None = None,
) -> dict:
    """Compare uint16 RGB TIFFs in their existing coordinates using row strips.

    Only uncompressed memory-mappable files are accepted. No alignment, resizing,
    downsampling or exposure fitting hides differences. The entire visible raster
    includes sensor margins. An optional (top, left, bottom, right) region uses
    half-open visible-raster coordinates at native pixel resolution, identically
    in both files. It does not establish registration or apply DNG DefaultCrop.
    Neither image is a ground-truth reference.
    """
    if type(strip_rows) is not int or strip_rows <= 0:
        raise ValueError("strip_rows must be a positive integer")
    if region is not None and (
        not isinstance(region, tuple)
        or len(region) != 4
        or any(type(value) is not int for value in region)
        or not (0 <= region[0] < region[2] and 0 <= region[1] < region[3])
    ):
        raise ValueError("region must be (top, left, bottom, right) nonempty nonnegative integer bounds")
    report = {
        "status": "not measured",
        "method": "Full-resolution unaligned uint16 sRGB render differences; normalized by 65535",
        "domain": "Entire LibRaw visible raster including sensor margins; DefaultCrop not applied",
        "alignment": "none",
        "scientific_fidelity": "not measured; neither render is a ground-truth reference",
    }
    with ExitStack() as stack:
        arrays = []
        for path in (first, second):
            array = tifffile.memmap(path, mode="r")
            stack.callback(array._mmap.close)
            if array.dtype != np.uint16 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("Comparison requires uint16 RGB TIFFs")
            arrays.append(array)
        left, right = arrays
        report["shapes"] = [list(array.shape) for array in arrays]
        if region is not None:
            top, x0, bottom, x1 = region
            if any(bottom > array.shape[0] or x1 > array.shape[1] for array in arrays):
                raise ValueError("region exceeds a visible raster")
            arrays = [array[top:bottom, x0:x1] for array in arrays]
            left, right = arrays
            report["region"] = list(region)
            report["domain"] = "Explicit native-pixel visible-raster region; DefaultCrop not applied"
        if left.shape != right.shape:
            return {**report, "reason": "Unequal raster shapes; no implicit resampling or alignment"}
        count = left.size
        absolute_sum = squared_sum = maximum = 0.0
        bias = np.zeros(3, dtype=np.float64)
        saturated = [0, 0]
        for start in range(0, left.shape[0], strip_rows):
            a, b = (array[start : start + strip_rows] for array in arrays)
            difference = a.astype(np.float64) - b
            difference /= 65535.0
            bias += np.sum(difference, axis=(0, 1), dtype=np.float64)
            absolute_sum += float(np.sum(np.abs(difference), dtype=np.float64))
            squared_sum += float(np.sum(np.square(difference), dtype=np.float64))
            maximum = max(maximum, float(np.max(np.abs(difference))))
            for index, block in enumerate((a, b)):
                saturated[index] += int(np.count_nonzero(np.any(block == 65535, axis=-1)))
        mse = squared_sum / count
        pixels = count // 3
        return {
            **report,
            "status": "measured render disagreement",
            "samples": count,
            "mae": absolute_sum / count,
            "rmse": float(np.sqrt(mse)),
            "max_absolute_error": maximum,
            "channel_bias_first_minus_second": (bias / pixels).tolist(),
            "psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None,
            "identical": maximum == 0,
            "psnr_null_reason": "Identical samples have infinite PSNR" if mse == 0 else None,
            "saturated_pixel_fraction": [value / pixels for value in saturated],
        }
