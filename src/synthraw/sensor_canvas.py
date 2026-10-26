"""Bounded-memory SR placement for PRD_v3 sensor-fit policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthraw.geometry import GeometryPlan
from synthraw.provenance import FieldRecord, Provenance


@dataclass(frozen=True)
class SensorCanvasResult:
    """Disk products and explicit provenance for deterministic invented bands."""

    path: Path
    generated_detail_mask: Path
    source_supported_mask: Path
    provenance: FieldRecord


def place_sensor_canvas(
    content: np.ndarray,
    plan: GeometryPlan,
    workdir: str | Path,
) -> SensorCanvasResult:
    """Place exact-resized visible content, extending outpaint with edge pixels.

    Rasterise fractional content edges by nearest-pixel rounding. The caller
    crops and uniformly resizes SR before placement; this function never resizes
    or stretches. Margins are neutral black. Source support describes geometry,
    not certainty that SR detail was present in the original capture.
    Only one RGB row and scalar mask row are materialised at a time.
    """
    left, top, right, bottom = (round(value) for value in plan.content_box)
    if not (0 <= left < right <= plan.width and 0 <= top < bottom <= plan.height):
        raise ValueError("content box must lie inside the sensor canvas")
    if content.shape != (bottom - top, right - left, 3):
        raise ValueError("content must match the rasterised content box")
    if plan.outpaint_bands and plan.margin_bands:
        raise ValueError("canvas cannot contain both outpaint and margin bands")
    directory = Path(workdir)
    paths = (
        directory / "intermediates/02_sr_rgb.npy",
        directory / "confidence/generated_detail_mask.npy",
        directory / "confidence/source_supported_mask.npy",
    )
    if any(path.exists() for path in paths):
        raise FileExistsError("canvas products must be new")
    arrays = []
    created = []
    try:
        for index, path in enumerate(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            shape = (plan.height, plan.width, 3) if index == 0 else (plan.height, plan.width)
            arrays.append(np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape))
            created.append(path)
        canvas, generated, supported = arrays
        columns = np.clip(np.arange(plan.width) - left, 0, right - left - 1)
        for y in range(plan.height):
            row = np.asarray(content[min(max(y - top, 0), bottom - top - 1)], dtype=np.float32)
            if not np.isfinite(row).all() or np.any((row < 0) | (row > 1)):
                raise ValueError("content must contain finite RGB values in [0, 1]")
            support = (np.arange(plan.width) >= left) & (np.arange(plan.width) < right) & (top <= y < bottom)
            canvas[y] = row[columns] if plan.outpaint_bands else 0
            if top <= y < bottom:
                canvas[y, left:right] = row
            supported[y] = support
            generated[y] = ~support if plan.outpaint_bands else 0
        for array in arrays:
            array.flush()
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        arrays.clear()
    record = FieldRecord(
        field="sensor_canvas_fill",
        value={"outpaint": "edge extension", "margins": "neutral black", "generated_detail_mask": str(paths[1])},
        provenance=Provenance.FABRICATED,
        method="Deterministic edge extension for outpaint; zero-valued sensor margins",
        disclaimer="Extended pixels are invented; source support is geometric support, not recovered-detail confidence",
    )
    return SensorCanvasResult(*paths, provenance=record)
