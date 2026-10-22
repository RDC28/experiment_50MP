"""Exact 50 MP target-resolution calculation (PRD sections 3.5, 6.1 and 25.3).

The source is never assumed to be 3840 x 2160. For a W x H source the nominal
linear scale is ``s = sqrt(N / (W * H))``; integer target dimensions are then
chosen by a deterministic search:

* ``tolerance`` mode: every candidate whose pixel count lies within
  ``pixel_count_tolerance`` of N is ranked by aspect-ratio error first, then by
  pixel-count error, then by width. The first candidate wins.
* ``exact`` mode: only divisor pairs of N qualify, so the pixel count is exactly
  N. Their aspect ratio can be far from the source's, so when
  ``preserve_aspect_ratio`` is set the best pair must stay within
  ``max_aspect_ratio_error`` or the calculation fails with an explanation.

All comparisons use exact rational arithmetic so the result never depends on
floating-point rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

RESOLUTION_MODES = ("tolerance", "exact")


@dataclass(frozen=True)
class TargetResolution:
    source_width: int
    source_height: int
    width: int
    height: int
    target_pixels: int
    mode: str

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def megapixels(self) -> float:
        return self.pixels / 1e6

    @property
    def pixel_count_error(self) -> float:
        """Signed relative difference between the achieved and requested pixel count."""
        return (self.pixels - self.target_pixels) / self.target_pixels

    @property
    def aspect_ratio_error(self) -> float:
        """Signed relative difference between the target and source aspect ratios."""
        return float(_aspect_error(self.width, self.height, self.source_width, self.source_height))

    @property
    def nominal_scale(self) -> float:
        return math.sqrt(self.target_pixels / (self.source_width * self.source_height))

    @property
    def scale_x(self) -> float:
        return self.width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.height / self.source_height

    def to_dict(self) -> dict:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "width": self.width,
            "height": self.height,
            "pixels": self.pixels,
            "megapixels": round(self.megapixels, 6),
            "target_pixels": self.target_pixels,
            "mode": self.mode,
            "pixel_count_error": self.pixel_count_error,
            "aspect_ratio_error": self.aspect_ratio_error,
            "nominal_scale": self.nominal_scale,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
        }


def nominal_scale(source_width: int, source_height: int, target_megapixels: float = 50.0) -> float:
    """Linear scale factor ``sqrt(N / (W * H))`` from PRD section 3.5."""
    _check_dimensions(source_width, source_height)
    return math.sqrt(target_megapixels * 1e6 / (source_width * source_height))


def compute_target_resolution(
    source_width: int,
    source_height: int,
    target_megapixels: float = 50.0,
    *,
    mode: str = "tolerance",
    pixel_count_tolerance: float = 0.001,
    dimension_multiple: int = 2,
    preserve_aspect_ratio: bool = True,
    max_aspect_ratio_error: float = 0.001,
) -> TargetResolution:
    _check_dimensions(source_width, source_height)
    if mode not in RESOLUTION_MODES:
        raise ValueError(f"mode must be one of {RESOLUTION_MODES}, got {mode!r}")
    if target_megapixels <= 0:
        raise ValueError("target_megapixels must be positive")
    if dimension_multiple < 1:
        raise ValueError("dimension_multiple must be at least 1")
    if pixel_count_tolerance < 0:
        raise ValueError("pixel_count_tolerance must not be negative")

    target_pixels = round(target_megapixels * 1e6)
    if mode == "exact":
        width, height = _exact_dimensions(source_width, source_height, target_pixels, dimension_multiple)
        error = abs(_aspect_error(width, height, source_width, source_height))
        if preserve_aspect_ratio and error > Fraction(max_aspect_ratio_error):
            raise ValueError(
                f"exact mode: no divisor pair of {target_pixels} pixels keeps the "
                f"{source_width}x{source_height} aspect ratio within {max_aspect_ratio_error:.4%} "
                f"(best is {width}x{height}, aspect error {float(error):.4%}). "
                "Use resolution_mode='tolerance' or disable preserve_aspect_ratio."
            )
    else:
        width, height = _tolerance_dimensions(
            source_width, source_height, target_pixels, pixel_count_tolerance, dimension_multiple
        )
    return TargetResolution(source_width, source_height, width, height, target_pixels, mode)


def _tolerance_dimensions(
    source_width: int, source_height: int, target_pixels: int, tolerance: float, multiple: int
) -> tuple[int, int]:
    tol = Fraction(tolerance)
    ideal_width = source_width * math.sqrt(target_pixels / (source_width * source_height))
    # Widths outside this window cannot meet the pixel tolerance at the source aspect ratio.
    span = ideal_width * (float(tol) + 0.01) + 2 * multiple
    low = max(multiple, math.floor((ideal_width - span) / multiple) * multiple)
    high = math.ceil((ideal_width + span) / multiple) * multiple

    best: tuple | None = None
    for width in range(low, high + 1, multiple):
        ideal_height = Fraction(width * source_height, source_width)
        floor_height = (ideal_height.numerator // ideal_height.denominator) // multiple * multiple
        for height in {floor_height, floor_height + multiple}:
            if height < multiple:
                continue
            pixel_error = abs(Fraction(width * height - target_pixels, target_pixels))
            if pixel_error > tol:
                continue
            key = (abs(_aspect_error(width, height, source_width, source_height)), pixel_error, width)
            if best is None or key < best[0]:
                best = (key, width, height)
    if best is None:
        raise ValueError(
            f"no dimensions that are multiples of {multiple} reach {target_pixels} pixels within "
            f"{tolerance:.4%}; widen pixel_count_tolerance or use resolution_mode='exact'"
        )
    return best[1], best[2]


def _exact_dimensions(source_width: int, source_height: int, target_pixels: int, multiple: int) -> tuple[int, int]:
    best: tuple | None = None
    for width in _divisors(target_pixels):
        height = target_pixels // width
        if width % multiple or height % multiple:
            continue
        key = (abs(_aspect_error(width, height, source_width, source_height)), width)
        if best is None or key < best[0]:
            best = (key, width, height)
    if best is None:
        raise ValueError(f"{target_pixels} has no divisor pair whose sides are both multiples of {multiple}")
    return best[1], best[2]


def _aspect_error(width: int, height: int, source_width: int, source_height: int) -> Fraction:
    return Fraction(width * source_height, height * source_width) - 1


def _divisors(n: int) -> list[int]:
    small, large = [], []
    for d in range(1, math.isqrt(n) + 1):
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
    return small + large[::-1]


def _check_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"source dimensions must be positive, got {width}x{height}")
