"""Explainable estimated human-region masks (PRD 8.2, 11.5).

These are spatial/chroma heuristics, not semantic segmentation or recovered
anatomy. Masks partition weights using documented protective priorities.
Hands are manual-only. All masks use source dimensions and [0, 1] weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from .color_management import linear_to_srgb
from .face_preservation import FaceDetections
from .human_pipeline import _validate_image
from .pipeline import MEASUREMENT, StageRecord

PRECEDENCE = ("primary_face", "background_faces", "hands", "skin", "hair", "clothing", "bokeh", "background")

REGIONS = ("primary_face", "background_faces", "skin", "hair", "clothing", "hands", "background", "bokeh")


@dataclass(frozen=True)
class RegionMasks:
    masks: dict[str, np.ndarray]
    overridden: tuple[str, ...]
    feather_pixels: float
    bokeh_variance: float
    skin_samples: np.ndarray | None = None

    def stage_record(self) -> StageRecord:
        return StageRecord(
            name="region_masks",
            classification=MEASUREMENT,
            parameters={
                "regions": list(REGIONS),
                "manual_overrides": list(self.overridden),
                "feather_pixels": self.feather_pixels,
                "bokeh_variance": self.bokeh_variance,
                "precedence": list(PRECEDENCE),
                "hair_background_distance": 0.10,
                "clothing_sample_distance": 0.15,
                "hair_sample_distance": 0.12,
                "skin_chroma_distance": 0.06,
            },
            output_precision="float32 source-scale weights [0, 1]",
            notes=[
                "Estimated face ellipses, locally colour-constrained hair/clothing and sRGB skin heuristic; "
                "not semantic truth.",
                "Hair uses colour-consistent components connected to the top of each face; "
                "estimated hair is excluded from automatic face ellipses before protective partitioning.",
                "Skin controls touch estimated exposed neck, hand and arm skin near each face, selected by "
                "normalised RGB chroma similarity to facial skin; faces retain precedence. "
                "Skin smoothing also samples facial skin subject to face blend caps.",
                "Hands are manual-only as a separate region; skin estimates do not establish hand anatomy.",
                "Bokeh is estimated by low local encoded-RGB variance, not measured optical defocus.",
                "Manual weights reserve capacity first; conflicting manual sums above 1 are rejected. "
                "Remaining capacity uses faces > hands > skin > hair > clothing > bokeh > background.",
                "Skin measurement samples retain face chroma before weight partitioning; not enhancement weights.",
            ],
        )


def estimate_masks(
    linear: np.ndarray,
    detections: FaceDetections,
    *,
    overrides: Mapping[str, np.ndarray] | None = None,
    feather_pixels: float = 2.0,
    bokeh_variance: float = 0.0004,
) -> RegionMasks:
    """Estimate masks; full-size manual weights replace any named mask exactly.

    feather_pixels: Gaussian sigma in source pixels, 0 to 32 (default 2).
    bokeh_variance: encoded RGB variance threshold, (0, 1] (default .0004).
    Both are provisional, to be calibrated experimentally. Broad masks or high
    bokeh thresholds may suppress real detail; inspect overlays before use.
    """
    _validate_image(linear)
    for value, lower, upper, name in (
        (feather_pixels, 0, 32, "feather_pixels"),
        (bokeh_variance, 0, 1, "bokeh_variance"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite numeric")
        if not lower <= value <= upper or (name == "bokeh_variance" and value == 0):
            raise ValueError(f"invalid {name}")
    height, width = linear.shape[:2]
    if detections.primary_index is not None and not 0 <= detections.primary_index < len(detections.faces):
        raise ValueError("primary_index must reference a face")
    masks = {name: np.zeros((height, width), np.float32) for name in REGIONS}
    encoded = np.clip(linear_to_srgb(linear), 0, 1).astype(np.float32)
    r, g, b = np.moveaxis(encoded, -1, 0)
    chroma = (r > g) & (g > b) & (r - b > 0.04) & (r - g < 0.45)
    normalised = encoded / np.maximum(encoded.sum(axis=2, keepdims=True), 1e-6)
    # Broadcast coordinates avoid a separate full-size coordinate pair per face.
    y, x = np.ogrid[:height, :width]

    def ellipse(cx, cy, rx, ry):
        return (((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1).astype(np.float32)

    for index, face in enumerate(detections.faces):
        face.validate_bounds(width, height)
        cx, cy = face.x + face.width / 2, face.y + face.height / 2
        region = "primary_face" if index == detections.primary_index else "background_faces"
        masks[region] = np.maximum(masks[region], ellipse(cx, cy, face.width / 2, face.height / 2))
        # Sample a narrow perimeter outside the head band, clipped at image edges.
        head = ellipse(cx, face.y + face.height * 0.05, face.width * 0.6, face.height * 0.3)
        outer = ellipse(cx, face.y + face.height * 0.05, face.width * 0.85, face.height * 0.55)
        samples = encoded[(outer > 0) & (head == 0) & (y < face.y)]
        if samples.size:
            background_colour = np.median(samples, axis=0)
            distinct = np.linalg.norm(encoded - background_colour, axis=2) > 0.10
            seeds = (head > 0) & distinct & (y < face.y + face.height * 0.2)
            if seeds.any():
                hair_colour = np.median(encoded[seeds], axis=0)
                similar = np.linalg.norm(encoded - hair_colour, axis=2) < 0.12
                # Bound the search to the head, but allow the whole cap and side hair.
                search = ellipse(cx, face.y + face.height * 0.15, face.width * 0.75, face.height * 0.7)
                candidates = (similar & distinct & (search > 0)).astype(np.uint8)
                _, labels = cv2.connectedComponents(candidates, connectivity=8)
                connected = np.unique(labels[seeds & (candidates > 0)])
                connected = connected[connected != 0]
                hair = np.isin(labels, connected).astype(np.float32)
                masks["hair"] = np.maximum(masks["hair"], hair)
                # Face boxes can include hair. Preserve facial skin precedence by
                # removing only this connected, colour-matched cap from the ellipse.
                masks[region] *= 1 - hair
        body = ellipse(cx, face.y + face.height * 1.8, face.width * 0.95, face.height * 0.8)
        body *= y >= face.y + face.height
        # A central patch immediately below the chin supplies the garment colour.
        x0, x1 = max(0, int(cx - face.width * 0.15)), min(width, int(cx + face.width * 0.15) + 1)
        y0 = min(height, face.y + face.height)
        y1 = min(height, y0 + max(1, int(face.height * 0.2)))
        samples = encoded[y0:y1, x0:x1]
        if samples.size:
            garment_colour = np.median(samples.reshape(-1, 3), axis=0)
            similar = np.linalg.norm(encoded - garment_colour, axis=2) < 0.15
            masks["clothing"] = np.maximum(masks["clothing"], body * similar)
        # Sample within the facial ellipse, excluding non-skin features and hair.
        samples = normalised[(ellipse(cx, cy, face.width * 0.4, face.height * 0.4) > 0) & chroma]
        if samples.size:
            skin_chroma = np.median(samples, axis=0)
            similar = np.linalg.norm(normalised - skin_chroma, axis=2) < 0.06
            # Local body window includes disconnected exposed hands and arms.
            # Colour similarity is a heuristic, never an anatomical annotation.
            nearby = (
                (abs(x + 0.5 - cx) <= face.width * 1.25) & (y >= face.y + face.height) & (y < face.y + 3 * face.height)
            )
            masks["skin"] = np.maximum(masks["skin"], (nearby & chroma & similar).astype(np.float32))
    for name in REGIONS[:6]:
        if feather_pixels:
            # Feather inward only: colour-rejected background must stay excluded.
            support = masks[name] > 0
            masks[name] = cv2.GaussianBlur(masks[name], (0, 0), feather_pixels) * support
    overrides = {} if overrides is None else overrides
    unknown = set(overrides) - set(REGIONS)
    if unknown:
        raise ValueError(f"unknown mask names: {sorted(unknown)}")
    for name, mask in overrides.items():
        _validate_mask(mask, (height, width))
        masks[name] = mask.copy()
    skin_samples = (
        masks["skin"].copy()
        if "skin" in overrides
        else np.maximum(masks["primary_face"], masks["background_faces"]) * chroma
    )
    manual_total = sum((masks[name] for name in overrides), np.zeros((height, width), np.float32))
    if np.any(manual_total > 1):
        raise ValueError("manual mask weights must sum to at most 1 per pixel")
    remaining = 1 - manual_total
    # Reserve manual weights before resolving automatic masks in protective order.
    for name in PRECEDENCE[:6]:
        if name not in overrides:
            masks[name] = np.minimum(masks[name], remaining)
            remaining = np.maximum(remaining - masks[name], 0)
    mean = cv2.blur(encoded, (9, 9))
    variance = np.maximum(cv2.blur(encoded * encoded, (9, 9)) - mean * mean, 0).mean(axis=2)
    if "bokeh" not in overrides:
        # Smooth low-variance subset of available non-person weight, never all of it.
        smoothness = np.clip(1 - variance / bokeh_variance, 0, 1)
        smoothness = smoothness * smoothness * (3 - 2 * smoothness)
        person = np.maximum.reduce([masks[name] for name in REGIONS[:6]]) > 0
        masks["bokeh"] = remaining * smoothness * 0.95 * ~person
        remaining = np.maximum(remaining - masks["bokeh"], 0)
    if "background" not in overrides:
        masks["background"] = remaining
    return RegionMasks(masks, tuple(sorted(overrides)), float(feather_pixels), float(bokeh_variance), skin_samples)


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> None:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.float32
        or mask.shape != shape
        or not np.isfinite(mask).all()
        or np.any(mask < 0)
        or np.any(mask > 1)
    ):
        raise ValueError("mask must be float32 source-sized finite weights in [0, 1]")


def mask_overlay(linear: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Packaging: RGB uint8 preview, magenta at 50% opacity for weight 1."""
    _validate_image(linear)
    _validate_mask(mask, linear.shape[:2])
    encoded = np.clip(linear_to_srgb(linear), 0, 1)
    alpha = mask[..., None] * 0.5
    return np.round((encoded * (1 - alpha) + np.array([1, 0, 1]) * alpha) * 255).astype(np.uint8)
