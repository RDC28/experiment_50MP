"""Allocation-free target planning for PRD_v3 geometry and compute budgets."""

from __future__ import annotations

import math
from dataclasses import dataclass

from synthraw.config import ExperimentConfig

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class GeometryPlan:
    """Coordinates are output pixels, boxes are (left, top, right, bottom).

    crop_box instead uses source pixels. ActiveArea uses DNG (top, left,
    bottom, right); default crop origin/size use DNG (x, y)/(width, height).
    Fractional content edges retain exact uniform scale before rasterisation.
    """

    width: int
    height: int
    scale_factor: float
    crop_box: Box
    content_box: Box
    outpaint_bands: tuple[Box, ...]
    margin_bands: tuple[Box, ...]
    default_crop_origin: tuple[int, int]
    default_crop_size: tuple[int, int]
    active_area: tuple[int, int, int, int]
    memory_bytes: dict[str, int]


class TargetGeometryPlanner:
    """Plan uniform source fitting without allocating any sensor arrays."""

    def plan(
        self,
        source_width: int,
        source_height: int,
        config: ExperimentConfig | None = None,
        *,
        fast_dev_max_side: int = 512,
    ) -> GeometryPlan:
        """Resolve native/virtual geometry and the selected sensor-fit policy.

        Cinema rounds the ideal 50.1 MP dimensions to the nearest even pixels.
        OUTPAINT_VERTICAL contains wide sources; portrait sources require side
        outpainting. FAST_DEV_RUN limits the longest sensor side to fast_dev_max_side
        pixels (default 512) while retaining the selected geometry's aspect ratio.
        """
        if any(type(value) is not int or value <= 0 for value in (source_width, source_height)):
            raise ValueError("Source dimensions must be positive integers")
        if type(fast_dev_max_side) is not int or fast_dev_max_side < 2:
            raise ValueError("fast_dev_max_side must be an integer of at least 2")
        config = config or ExperimentConfig()
        if config.OUTPUT_GEOMETRY == "SONY_A1II_NATIVE":
            width, height = 8640, 5760
        else:
            ratio = source_width / source_height
            width = max(2, 2 * round(math.sqrt(50_100_000 * ratio) / 2))
            height = max(2, 2 * round(math.sqrt(50_100_000 / ratio) / 2))
        if config.FAST_DEV_RUN:
            reduction = min(1.0, fast_dev_max_side / max(width, height))
            width, height = (max(2, 2 * round(side * reduction / 2)) for side in (width, height))

        cover = config.SENSOR_FIT_MODE == "CROP_HORIZONTAL"
        scale = (max if cover else min)(width / source_width, height / source_height)
        fitted_width, fitted_height = source_width * scale, source_height * scale
        left, top = (width - fitted_width) / 2, (height - fitted_height) / 2
        content = (
            max(0.0, left),
            max(0.0, top),
            min(float(width), left + fitted_width),
            min(float(height), top + fitted_height),
        )
        crop = (
            (content[0] - left) / scale,
            (content[1] - top) / scale,
            (content[2] - left) / scale,
            (content[3] - top) / scale,
        )
        assert math.isclose((crop[2] - crop[0]) * scale, content[2] - content[0])
        assert math.isclose((crop[3] - crop[1]) * scale, content[3] - content[1])
        bands = []
        if top > 0:
            bands.extend(((0.0, 0.0, float(width), top), (0.0, height - top, float(width), float(height))))
        if left > 0:
            bands.extend(((0.0, 0.0, left, float(height)), (width - left, 0.0, float(width), float(height))))
        margins = config.SENSOR_FIT_MODE == "FIT_WITH_SENSOR_MARGINS"
        # Inward even rounding keeps the default crop inside valid source content
        # and aligned to the Bayer tile. Padding remains part of ActiveArea.
        if margins:
            x, y = (2 * math.ceil(content[i] / 2) for i in (0, 1))
            right, bottom = (2 * math.floor(content[i] / 2) for i in (2, 3))
            if right <= x or bottom <= y:
                raise ValueError("Source aspect ratio leaves no CFA-aligned default crop")
        else:
            x, y, right, bottom = 0, 0, width, height
        pixels = width * height
        memory = {
            "rgb_float32": pixels * 12,
            "rgb_float16": pixels * 6,
            "rgb_uint16": pixels * 6,
            "cfa_uint16": pixels * 2,
            "scalar_float32": pixels * 4,
        }
        return GeometryPlan(
            width,
            height,
            scale,
            crop,
            content,
            tuple(bands) if not margins else (),
            tuple(bands) if margins else (),
            (x, y),
            (right - x, bottom - y),
            (0, 0, height, width),
            memory,
        )
