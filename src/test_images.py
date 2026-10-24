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


def make_crowd_test_image(width: int = 384, height: int = 256, seed: int = 42, *, tier_config=None):
    """Return drawn uint8 sRGB people and ground-truth source-pixel face boxes.

    This Simulation fixture contains no photographs or real identities. At the
    default size its 128, 64 and 16 pixel faces span all default source tiers.
    Sizes are absolute source pixels derived from tier_config (default thresholds),
    independent of canvas size. Canvases at least 1024x768 contain 14 people.
    Boxes describe the complete face drawing, including hair; the first is the
    primary subject. Manual annotations have confidence 1, not an identity score.
    A smooth background provides bokeh-like low-variance regions; clothing has
    deterministic woven texture. It is a processing fixture, not a detector
    accuracy benchmark or anatomical ground truth.
    """
    from .face_preservation import FaceBox, FaceTierConfig

    if any(not isinstance(v, int) or isinstance(v, bool) or v < 256 for v in (width, height)):
        raise ValueError("crowd image dimensions must be integers >= 256")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    # Broadcast the smooth background without full-canvas float coordinate grids.
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    u = np.arange(width, dtype=np.float32) / (width - 1)
    v = np.arange(height, dtype=np.float32) / (height - 1)
    canvas[..., 0] = np.round((0.18 + 0.08 * u) * 255).astype(np.uint8)[None, :]
    canvas[..., 1] = np.round((0.26 + 0.08 * v) * 255).astype(np.uint8)[:, None]
    canvas[..., 2] = np.round((0.32 + 0.06 * u) * 255).astype(np.uint8)[None, :]
    tiers = tier_config or FaceTierConfig()
    primary = (tiers.tier1_min_pixels * 4 + 2) // 3
    medium = (tiers.tier1_min_pixels + tiers.tier2_min_pixels) // 2
    small = max(1, tiers.tier2_min_pixels // 2)
    layout = [(0.06, 0.18, primary), (0.58, 0.25, medium), (0.87, 0.32, small)]
    if width >= 1024 and height >= 768:
        layout.extend([(0.35, 0.25, medium), (0.72, 0.25, medium)])
        layout.extend((0.1 + column * 0.09, 0.60, small) for column in range(9))
    faces = tuple(FaceBox(int(width * px), int(height * py), size, size) for px, py, size in layout)
    skins = ((208, 155, 119), (150, 98, 69), (234, 185, 146))
    for index, face in enumerate(faces):
        skin = skins[index % len(skins)]
        face.validate_bounds(width, height)
        fx, fy, size = face.x, face.y, face.width
        cx = fx + size // 2
        # Clothing stays below the annotated face and has reproducible texture.
        top, bottom = fy + size, min(height, fy + size * 2)
        left, right = max(0, fx - size // 6), min(width, fx + size + size // 6)
        yy, xx = np.mgrid[top:bottom, left:right]
        color = rng.integers(55, 155, size=3)
        weave = 5 * np.sin(xx * 1.3) * np.cos(yy * 1.1)
        canvas[top:bottom, left:right] = np.clip(color + weave[..., None], 0, 255).astype(np.uint8)
        # Draw into an isolated patch so antialiasing stays within its box.
        patch = canvas[fy : fy + size, fx : fx + size].copy()
        center = (size // 2, size // 2)
        cv2.ellipse(patch, center, (size * 2 // 5, size // 2 - 1), 0, 0, 360, skin, -1, cv2.LINE_AA)
        cv2.ellipse(
            patch, (size // 2, size // 4), (size * 2 // 5, size // 4), 0, 180, 360, (42, 30, 25), -1, cv2.LINE_AA
        )
        for eye_x in (size * 3 // 8, size * 5 // 8):
            cv2.circle(patch, (eye_x, size * 7 // 16), max(1, size // 40), (28, 24, 23), -1)
        cv2.line(
            patch,
            (size // 2, size // 2),
            (size // 2 - max(1, size // 30), size * 5 // 8),
            (125, 79, 58),
            max(1, size // 60),
        )
        cv2.ellipse(
            patch,
            (size // 2, size * 11 // 16),
            (max(1, size // 8), max(1, size // 16)),
            0,
            0,
            180,
            (99, 49, 43),
            max(1, size // 60),
            cv2.LINE_AA,
        )
        canvas[fy : fy + size, fx : fx + size] = patch
        # Small visible hands are illustrative only, not hand annotations.
        cv2.circle(canvas, (cx, min(height - 1, bottom - max(2, size // 8))), max(2, size // 12), skin, -1, cv2.LINE_AA)
    return canvas, faces
