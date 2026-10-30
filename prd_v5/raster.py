"""Disk-backed baseline geometry resampling, without learned reconstruction."""

from __future__ import annotations

import numpy as np

from prd_v5.geometry import GeometryPlan


def _check_buffers(source, destination, support, plan: GeometryPlan) -> None:
    if plan.generated_bands:
        raise ValueError("Outpainting required but no generator is available")
    if source.shape != (plan.source_size[1], plan.source_size[0], 3) or source.dtype.kind != "f":
        raise ValueError("source must be floating HxWx3 matching the geometry plan")
    for array, dtype, shape in (
        (destination, np.float32, (plan.height, plan.width, 3)),
        (support, np.bool_, (plan.height, plan.width)),
    ):
        if (
            not isinstance(array, np.memmap)
            or array.dtype != dtype
            or array.shape != shape
            or not array.flags.writeable
        ):
            raise ValueError("outputs must be writable disk-backed arrays matching geometry and dtype")
    arrays = (source, destination, support)
    for i, array in enumerate(arrays):
        for other in arrays[i + 1 :]:
            if np.may_share_memory(array, other) or (isinstance(array, np.memmap) and array.filename == other.filename):
                raise ValueError("source and outputs must use separate nonoverlapping buffers")


def rasterize(source: np.ndarray, destination: np.memmap, support: np.memmap, plan: GeometryPlan) -> dict:
    """Apply edge-coordinate geometry with pixel-center bilinear sampling.

    Source must already be normalized encoded sRGB. This explicit baseline is
    not an SR fallback. Sensor margins are zero signal with false support;
    outpainting requests fail before writing. Working arrays are one row wide.
    Caller owns buffers and must discard partial output on invalid source data.
    """
    _check_buffers(source, destination, support, plan)
    x = np.arange(plan.width, dtype=np.float64) + 0.5
    source_x = np.clip((x - plan.offset_xy[0]) / plan.scale - 0.5, 0, plan.source_size[0] - 1)
    x0 = np.floor(source_x).astype(np.intp)
    x1 = np.minimum(x0 + 1, plan.source_size[0] - 1)
    weight = (source_x - x0)[:, None]
    supported_x = (x >= plan.content_box[0]) & (x < plan.content_box[2])
    supported_pixels = 0
    for y in range(plan.height):
        sy = np.clip((y + 0.5 - plan.offset_xy[1]) / plan.scale - 0.5, 0, plan.source_size[1] - 1)
        y0 = int(np.floor(sy))
        y1 = min(y0 + 1, plan.source_size[1] - 1)
        rows = []
        for row in (source[y0], source[y1]):
            if not np.isfinite(row).all() or np.any(row < 0) or np.any(row > 1):
                raise ValueError("source samples must be finite normalized [0, 1]")
            rows.append(row[x0] * (1 - weight) + row[x1] * weight)
        valid = supported_x & (y + 0.5 >= plan.content_box[1]) & (y + 0.5 < plan.content_box[3])
        destination[y] = np.where(valid[:, None], rows[0] * (1 - (sy - y0)) + rows[1] * (sy - y0), 0)
        support[y] = valid
        supported_pixels += int(np.count_nonzero(valid))
    destination.flush()
    support.flush()
    return {
        "method": "explicit_bilinear_encoded_srgb_baseline",
        "geometry": plan.to_dict(),
        "source_supported_pixels": supported_pixels,
        "sensor_margin_pixels": plan.width * plan.height - supported_pixels,
        "margin_signal": 0,
        "learned_super_resolution": "not_requested_by_this_baseline_API",
        "outpainting": "not_performed",
        "validation": "Geometry only; reconstruction fidelity not measured",
    }


def rasterize_lanczos(source, destination, support, plan):
    """Lanczos-3 baseline using the shared row-bounded resizer and CFA crop.

    The crop maps back through the plan affine, matching bilinear sample centres.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from synthraw.super_resolution import resize_exact

    _check_buffers(source, destination, support, plan)
    x, y = plan.default_crop_origin
    width, height = plan.default_crop_size
    (ox, oy), scale = plan.offset_xy, plan.scale
    source_box = tuple(
        min(float(limit), max(0.0, (edge - offset) / scale))
        for edge, offset, limit in zip(
            (x, y, x + width, y + height), (ox, oy, ox, oy), plan.source_size * 2, strict=True
        )
    )
    destination[:] = 0
    support[:] = False
    with TemporaryDirectory(prefix="lanczos-", dir=Path(destination.filename).parent) as directory:
        path = resize_exact(source, Path(directory) / "resized.npy", width=width, height=height, source_box=source_box)
        resized = np.load(path, mmap_mode="r")
        try:
            for row in range(height):
                destination[y + row, x : x + width] = resized[row]
                support[y + row, x : x + width] = True
        finally:
            resized._mmap.close()
    destination.flush()
    support.flush()
    return {
        "method": "Lanczos-3 encoded sRGB baseline to Bayer-aligned source crop",
        "geometry": plan.to_dict(),
        "source_supported_pixels": width * height,
        "sensor_margin_pixels": plan.width * plan.height - width * height,
        "margin_signal": 0,
        "learned_super_resolution": "not_requested_by_this_baseline_API",
        "outpainting": "not_performed",
        "validation": "Source crop rounding is explicit; fidelity needs truth",
    }
