"""Resampling in linear light and back-projection consistency (sections 5.2, 9.3)."""

from __future__ import annotations

import cv2
import numpy as np

from prd_v6 import constants


def _resize(image: np.ndarray, out_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.shape[:2] == tuple(out_hw):
        return image.copy()
    return cv2.resize(image, (out_hw[1], out_hw[0]), interpolation=interpolation)


def area(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Exact area (box) average; the render model of section 6.1 when shrinking."""
    if out_hw[0] > image.shape[0] or out_hw[1] > image.shape[1]:
        raise ValueError("area resampling only shrinks")
    return _resize(image, out_hw, cv2.INTER_AREA)


def cubic(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    return _resize(image, out_hw, cv2.INTER_CUBIC)


def lanczos(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    return _resize(image, out_hw, cv2.INTER_LANCZOS4)


def nearest(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    image = np.ascontiguousarray(image)
    return cv2.resize(image, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)


def to_grid(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Area when shrinking, Lanczos for a final step up (section 5.2)."""
    if out_hw[0] <= image.shape[0] and out_hw[1] <= image.shape[1]:
        return area(image, out_hw)
    return lanczos(image, out_hw)


def back_project(
    estimate: np.ndarray,
    source: np.ndarray,
    *,
    iterations: int = constants.BACK_PROJECTION_ITERATIONS,
    source_clipped: np.ndarray | None = None,
) -> np.ndarray:
    """Make the estimate, area-downsampled to the source grid, agree with the source in linear light.

    Each iteration adds the cubic-upsampled residual. The residual is zeroed where the source is
    clipped, because there the source is a bound and not a measurement, and the result stays >= 0.
    """
    out = np.array(estimate, dtype=np.float32, copy=True)
    source = np.asarray(source, dtype=np.float32)
    for _ in range(iterations):
        residual = source - area(out, source.shape[:2])
        if source_clipped is not None:
            residual[source_clipped] = 0.0
        out += cubic(residual, out.shape[:2])
        np.maximum(out, 0.0, out=out)
    return out


def source_clipped(encoded_uint8: np.ndarray) -> np.ndarray:
    """Pixels of an 8-bit render where any channel is at 255."""
    return np.any(np.asarray(encoded_uint8) >= 255, axis=-1)
