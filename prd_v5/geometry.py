"""Allocation-free geometry for synthetic Bayer captures, not image generation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GeometryPlan:
    """Boxes use (left, top, right, bottom); source_crop uses source pixels.

    Other boxes use output pixels. DNG ActiveArea uses (top, left, bottom,
    right), while default crop uses (x, y) and (width, height). The affine
    mapping is output_xy = source_xy * scale + offset_xy, with uniform scale.
    No resampling, padding or outpainting is performed by this plan.
    """

    mode: str
    fit_policy: str
    source_size: tuple[int, int]
    width: int
    height: int
    requested_megapixels: float
    scale: float
    offset_xy: tuple[float, float]
    source_crop: tuple[float, float, float, float]
    content_box: tuple[float, float, float, float]
    margin_bands: tuple[tuple[float, float, float, float], ...]
    generated_bands: tuple[tuple[float, float, float, float], ...]
    default_crop_origin: tuple[int, int]
    default_crop_size: tuple[int, int]

    @property
    def actual_megapixels(self) -> float:
        return self.width * self.height / 1_000_000

    @property
    def active_area(self) -> tuple[int, int, int, int]:
        return (0, 0, self.height, self.width)

    def to_dict(self) -> dict:
        """Return detached JSON-compatible dimensions, mapping and byte budgets.

        Budgets are payloads for individual buffers, not peak RAM estimates.
        Native geometry deliberately ignores target MP in favor of 8640x5760.
        """
        pixels = self.width * self.height
        return {
            **asdict(self),
            "actual_megapixels": self.actual_megapixels,
            "active_area": self.active_area,
            "outpainting_status": "required_not_executed" if self.generated_bands else "not_required",
            "buffer_bytes": {"bayer_uint16": pixels * 2, "bayer_float32": pixels * 4, "rgb_float32": pixels * 12},
        }


class GeometryPlanner:
    """Resolve PRD v5 geometry without allocating frame-sized arrays.

    Even rounding can slightly change the raster aspect. Contain fitting keeps
    source content undistorted; an inward CFA-aligned crop excludes margins.
    CROP_HORIZONTAL means center-cover and can crop vertically for tall inputs.
    OUTPAINT_VERTICAL also requires side generation for tall inputs. Generated
    bands are only a request; an unavailable generator must not fill them silently.
    """

    def plan(
        self,
        source_width: int,
        source_height: int,
        *,
        mode: str = "PRESERVE_SOURCE_ASPECT_50MP",
        target_megapixels: float = 50.0,
        native_fit_policy: str = "LETTERBOX_SENSOR_MARGIN",
    ) -> GeometryPlan:
        if any(type(v) is not int or v <= 0 for v in (source_width, source_height)):
            raise ValueError("Source dimensions must be positive integers")
        if (
            isinstance(target_megapixels, bool)
            or not isinstance(target_megapixels, (int, float))
            or not math.isfinite(target_megapixels)
            or not 0 < target_megapixels <= 1000
        ):
            raise ValueError("Target megapixels must be finite in (0, 1000]")
        if mode not in ("PRESERVE_SOURCE_ASPECT_50MP", "SONY_A1II_NATIVE"):
            raise ValueError("Unknown geometry mode")
        if native_fit_policy not in ("LETTERBOX_SENSOR_MARGIN", "CROP_HORIZONTAL", "OUTPAINT_VERTICAL"):
            raise ValueError("Unknown native fit policy")
        policy = native_fit_policy if mode == "SONY_A1II_NATIVE" else "LETTERBOX_SENSOR_MARGIN"
        if mode == "SONY_A1II_NATIVE":
            width, height = 8640, 5760
        else:
            aspect = source_width / source_height
            pixels = target_megapixels * 1_000_000
            width = max(2, 2 * round(math.sqrt(pixels * aspect) / 2))
            height = max(2, 2 * round(math.sqrt(pixels / aspect) / 2))
        scale = (max if policy == "CROP_HORIZONTAL" else min)(width / source_width, height / source_height)
        left, top = (width - source_width * scale) / 2, (height - source_height * scale) / 2
        content = (max(0.0, left), max(0.0, top), min(float(width), width - left), min(float(height), height - top))
        crop = tuple(
            min(float(limit), max(0.0, (edge - offset) / scale))
            for edge, offset, limit in zip(
                content, (left, top, left, top), (source_width, source_height) * 2, strict=True
            )
        )
        bands = []
        if top > 0:
            bands.extend(((0.0, 0.0, float(width), top), (0.0, height - top, float(width), float(height))))
        if left > 0:
            bands.extend(((0.0, 0.0, left, float(height)), (width - left, 0.0, float(width), float(height))))
        if policy == "LETTERBOX_SENSOR_MARGIN":
            x, y = (2 * math.ceil(content[i] / 2) for i in (0, 1))
            right, bottom = (2 * math.floor(content[i] / 2) for i in (2, 3))
            if right <= x or bottom <= y:
                raise ValueError("Source aspect leaves no CFA-aligned supported crop")
        else:
            x, y, right, bottom = 0, 0, width, height
        return GeometryPlan(
            mode,
            policy,
            (source_width, source_height),
            width,
            height,
            float(target_megapixels),
            scale,
            (left, top),
            crop,
            content,
            tuple(bands) if policy == "LETTERBOX_SENSOR_MARGIN" else (),
            tuple(bands) if policy == "OUTPAINT_VERTICAL" else (),
            (x, y),
            (right - x, bottom - y),
        )
