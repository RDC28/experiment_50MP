"""Deterministic synthetic test images for feasibility runs and tests.

The two real AI-generated inputs are not stored in the repository, so Phase 0
and 1 are exercised on a generated image with the structures the baselines
must handle: smooth gradients (banding), a zone plate (aliasing and ringing),
hard-edged colour patches, thin strokes and text, fine noise texture, deep
shadows and small clipped light sources.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_PATCH_COLORS = (  # sRGB 8-bit, roughly the ColorChecker classic chart
    (115, 82, 68),
    (194, 150, 130),
    (98, 122, 157),
    (87, 108, 67),
    (133, 128, 177),
    (103, 189, 170),
    (214, 126, 44),
    (80, 91, 166),
    (193, 90, 99),
    (94, 60, 108),
    (157, 188, 64),
    (224, 163, 46),
    (56, 61, 150),
    (70, 148, 73),
    (175, 54, 60),
    (231, 199, 31),
    (187, 86, 149),
    (8, 133, 161),
    (243, 243, 242),
    (200, 200, 200),
    (160, 160, 160),
    (122, 122, 121),
    (85, 85, 85),
    (52, 52, 52),
)


def make_test_image(width: int = 3840, height: int = 2160, seed: int = 42) -> np.ndarray:
    """Return an (height, width, 3) uint8 sRGB test image."""
    if width < 64 or height < 64:
        raise ValueError("test images must be at least 64 x 64")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = x / (width - 1), y / (height - 1)

    # Background: a slow luminance ramp with a gentle hue drift (banding check).
    image = np.stack([0.15 + 0.6 * u, 0.2 + 0.5 * u * (1 - 0.3 * v), 0.35 + 0.4 * (1 - u) * v], axis=-1)

    # Zone plate, top left quarter: local frequency rises linearly to 0.35 cycles/pixel at the rim,
    # below the 0.5 Nyquist limit, so the source itself is alias-free.
    cx, cy, radius = width * 0.2, height * 0.28, min(width, height) * 0.22
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    zone = 0.5 + 0.5 * np.cos(np.pi * 0.35 * d2 / radius)
    inside = d2 < radius**2
    image[inside] = np.repeat(zone[inside][:, None], 3, axis=1)

    # Colour patches, top right.
    cols = 6
    patch = max(4, int(min(width * 0.06, height * 0.1)))
    x0, y0 = int(width * 0.55), int(height * 0.05)
    for index, color in enumerate(_PATCH_COLORS):
        px, py = x0 + (index % cols) * (patch + patch // 8), y0 + (index // cols) * (patch + patch // 8)
        image[py : py + patch, px : px + patch] = np.array(color, dtype=np.float32) / 255

    # Night region, bottom right: deep shadows, faint texture and small clipped lights.
    night = (u > 0.55) & (v > 0.63)
    texture = rng.normal(0, 0.012, size=(height, width)).astype(np.float32)
    image[night] = (0.03 + 0.04 * v[night] + texture[night])[:, None] * np.array([0.8, 0.9, 1.2], dtype=np.float32)
    canvas = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    for _ in range(24):
        lx = int(rng.uniform(0.57, 0.98) * width)
        ly = int(rng.uniform(0.66, 0.98) * height)
        size = max(2, int(rng.uniform(0.002, 0.006) * width))
        glow = tuple(int(c) for c in rng.choice([(255, 196, 120), (180, 220, 255), (255, 120, 90)]))
        cv2.circle(canvas, (lx, ly), size * 3, tuple(int(c * 0.35) for c in glow), -1, cv2.LINE_AA)
        cv2.circle(canvas, (lx, ly), size, (255, 255, 255), -1, cv2.LINE_AA)

    # Fabric-like noise texture, bottom left.
    fabric = (u < 0.45) & (v > 0.73)
    weave = 0.5 + 0.18 * np.sin(x * 0.9) * np.sin(y * 0.7) + rng.normal(0, 0.06, size=(height, width))
    canvas[fabric] = (np.clip(weave[fabric][:, None] * np.array([0.75, 0.55, 0.45]), 0, 1) * 255).round()

    # Thin, slightly slanted strokes between the patches and the night region; text below the zone plate.
    scale = width / 3840
    for i in range(12):
        y_start = int(height * 0.525 + i * 12 * scale)
        cv2.line(
            canvas,
            (int(width * 0.42), y_start),
            (int(width * 0.98), y_start + int((8 + i * 3) * scale)),
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )
    for row, text_scale in zip((0.56, 0.59, 0.635, 0.69), (0.5, 0.9, 1.6, 2.6), strict=True):
        cv2.putText(
            canvas,
            "SYNTHETIC RAW 50MP LAB 0123456789",
            (int(width * 0.03), int(height * row)),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale * scale,
            (20, 20, 20),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )
    return canvas


def write_test_image(
    path: str | Path, width: int = 3840, height: int = 2160, seed: int = 42, *, embed_srgb_profile: bool = False
) -> Path:
    """Write the test image as an 8-bit PNG (untagged by default, like most AI-generated images)."""
    from PIL import Image

    from .color_management import srgb_icc_profile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"icc_profile": srgb_icc_profile()} if embed_srgb_profile else {}
    Image.fromarray(make_test_image(width, height, seed)).save(path, **options)
    return path
