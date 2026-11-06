from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from video_v1 import ycbcr

TINY_HW = (48, 64)
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg and ffprobe are not installed"
)


def tiny_rgb(index: int, hw=TINY_HW) -> np.ndarray:
    """Smooth non-linear R'G'B' in [0.05, 0.85] with a moving bar, so frames differ a little.

    The bar stays clear of the 8 px border the stability check ignores: demosaic edge error as it crosses
    that border moves the colour ratios of a frame this small by about 0.5%, the flicker threshold.
    """
    y, x = np.mgrid[0 : hw[0], 0 : hw[1]].astype(np.float32)
    y /= hw[0]
    x /= hw[1]
    rgb = np.stack([0.15 + 0.6 * x, 0.2 + 0.5 * y, 0.6 - 0.4 * x * y], axis=-1)
    column = (16 + 4 * index) % hw[1]
    rgb[:, column : column + 4] = [0.8, 0.75, 0.7]
    return np.clip(rgb, 0.05, 0.85)


def encode_planes(rgb: np.ndarray, *, bits: int, subsampling=(2, 2), full_range=False, matrix="bt709"):
    """R'G'B' to integer Y, Cb, Cr planes (box-filtered chroma), the inverse of `ycbcr.planes_to_rgb`."""
    kr, kb = ycbcr.MATRICES[matrix]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = kr * r + (1 - kr - kb) * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))
    sx, sy = subsampling
    h, w = y.shape
    cb = cb.reshape(h // sy, sy, w // sx, sx).mean(axis=(1, 3))
    cr = cr.reshape(h // sy, sy, w // sx, sx).mean(axis=(1, 3))
    top = 2**bits - 1
    if full_range:
        planes = [y * top, cb * top + 2 ** (bits - 1), cr * top + 2 ** (bits - 1)]
    else:
        scale = 2 ** (bits - 8)
        planes = [(16 + 219 * y) * scale, (128 + 224 * cb) * scale, (128 + 224 * cr) * scale]
    dtype = np.uint8 if bits == 8 else np.uint16
    return [np.clip(np.rint(p), 0, top).astype(dtype) for p in planes]


def make_clip(
    path: Path,
    frames: int = 3,
    *,
    bits: int = 8,
    sub: str = "420",
    rate: str = "30000/1001",
    tags: bool = True,
    extra_output: tuple[str, ...] = (),
    extra_inputs: tuple[str, ...] = (),
    codec: tuple[str, ...] = ("-c:v", "ffv1"),
    planes_out: list | None = None,
    hw: tuple[int, int] = TINY_HW,
) -> Path:
    """A lossless tiny clip from known planes; `planes_out` receives the planes of every frame.

    `extra_inputs` add further inputs after the raw video (e.g. an audio source).
    """
    subsampling = {"420": (2, 2), "422": (2, 1), "444": (1, 1)}[sub]
    pix_fmt = f"yuv{sub}p" + ("" if bits == 8 else f"{bits}le")
    raw = b""
    for index in range(frames):
        planes = encode_planes(tiny_rgb(index, hw), bits=bits, subsampling=subsampling)
        if planes_out is not None:
            planes_out.append(planes)
        raw += b"".join(p.astype("<u2" if bits > 8 else np.uint8).tobytes() for p in planes)
    # Colour tags go on the raw input (output-side tags are dropped by some encoders); siting is an output option.
    colour = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    siting = ["-chroma_sample_location", "left"] if sub != "444" else []
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-s", f"{hw[1]}x{hw[0]}", "-framerate", rate, *(colour if tags else []), "-i", "pipe:0",
        *extra_inputs, *codec, *(siting if tags else []), *extra_output, "-fflags", "+bitexact", str(path),
    ]  # fmt: skip
    subprocess.run(command, input=raw, check=True)
    return path
