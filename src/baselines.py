"""Classical interpolation baselines (PRD section 9).

Resampling runs in the linear working space at float32 precision, one channel
at a time, with Pillow's separable convolution resampler. Pillow keeps float
("F" mode) values unclamped, so kernel overshoot (ringing) survives into the
float output where it can be measured, and is only clipped when the integer
TIFF master is encoded. Pillow resamples single-threaded, which makes the
result deterministic for a given Pillow build.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BaselineMethod:
    name: str
    filter: Image.Resampling
    description: str


METHODS = {
    "lanczos": BaselineMethod(
        "lanczos",
        Image.Resampling.LANCZOS,
        "Lanczos-3 windowed sinc (a = 3, 6x6 taps when upscaling), Pillow float32",
    ),
    "bicubic": BaselineMethod(
        "bicubic",
        Image.Resampling.BICUBIC,
        "Keys cubic convolution (a = -0.5, 4x4 taps when upscaling), Pillow float32",
    ),
}
DOWNSAMPLE_DESCRIPTION = "area average (box filter over each output pixel footprint), Pillow float32"


def resample(linear: np.ndarray, width: int, height: int, method: str) -> np.ndarray:
    """Resize a working-space (H, W, 3) float32 image to (height, width, 3) float32."""
    if method not in METHODS:
        raise ValueError(f"unknown baseline {method!r}; expected one of {sorted(METHODS)}")
    return _resize(linear, width, height, METHODS[method].filter)


def downsample_area(
    linear: np.ndarray, width: int, height: int, box: tuple[float, float, float, float] | None = None
) -> np.ndarray:
    """Area-average reduction used to compare outputs with the source at source resolution.

    ``box`` (left, top, right, bottom in input pixels) reduces only that region at the same scale as a
    whole-image reduction; input pixels on an exact sampling tie may be assigned differently.
    """
    return _resize(linear, width, height, Image.Resampling.BOX, box)


def _resize(
    image: np.ndarray,
    width: int,
    height: int,
    resampling: Image.Resampling,
    box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    if image.ndim != 3 or image.dtype != np.float32:
        raise ValueError(f"expected an (H, W, C) float32 array, got {image.shape} {image.dtype}")
    out = np.empty((height, width, image.shape[2]), dtype=np.float32)
    for channel in range(image.shape[2]):
        plane = Image.fromarray(np.ascontiguousarray(image[..., channel]))
        resized = plane.resize((width, height), resampling, box=box)
        del plane
        out[..., channel] = np.asarray(resized, dtype=np.float32)
        del resized
    return out
