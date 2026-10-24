"""Region-aware simulated microcontrast, never recovered texture (PRD 11.5).

Masks are explainable estimates. Protective masks cap enhancement rather than
competing in an additive sum, so overlapping hair/clothing cannot sharpen faces.
No restoration, generative replacement or identity inference is performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

from .config import _param
from .human_pipeline import _validate_image
from .pipeline import StageRecord
from .region_masks import REGIONS, RegionMasks, _validate_mask


def _strength(default: float, region: str):
    return _param(
        default,
        f"{region} simulated detail strength.",
        "finite number 0 to 1",
        "Scales bounded linear-light unsharp detail; zero preserves the input.",
        warning="Provisional, calibrate experimentally; high strengths can change appearance and create halos.",
        bounds=(0, 1),
    )


@dataclass(frozen=True)
class RegionDetailConfig:
    """Numeric interpretation of low/moderate PRD profile strengths."""

    skin: float = _strength(0.05, "Skin (low)")
    hair: float = _strength(0.25, "Hair (moderate)")
    clothing: float = _strength(0.25, "Clothing (moderate)")
    hands: float = _strength(0.05, "Hands (low)")
    background: float = _strength(0.25, "Background/architecture (moderate)")
    primary_face: float = _strength(0.0, "Primary face (very conservative)")
    background_faces: float = _strength(0.0, "Background faces (no restoration)")
    bokeh: float = _strength(0.0, "Bokeh (minimal)")
    radius_pixels: float = _param(
        1.0,
        "Gaussian microcontrast radius in output pixels.",
        "0.1 to 8",
        "Sets the simulated detail scale.",
        warning="Large radii can produce visible halos.",
        bounds=(0.1, 8),
    )
    max_detail: float = _param(
        0.02,
        "Maximum absolute linear-light detail before strength scaling.",
        "0 to 0.1",
        "Bounds each channel's change to max_detail times local strength.",
        warning="Higher bounds can shift skin tone and amplify artifacts.",
        bounds=(0, 0.1),
    )

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            lo, hi = field.metadata["bounds"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not lo <= value <= hi
            ):
                raise ValueError(f"{field.name} must be finite numeric in [{lo}, {hi}]")


@dataclass(frozen=True)
class RegionDetailResult:
    linear: np.ndarray
    strength_map: np.ndarray
    config: RegionDetailConfig

    def stage_record(self) -> StageRecord:
        return StageRecord(
            name="region_detail",
            classification="Simulation",
            parameters=asdict(self.config),
            input_precision="float32 linear",
            output_precision="float32 linear",
            notes=[
                "Simulated bounded unsharp microcontrast with estimated region weights; no recovered detail.",
                "Bilinear mask resizing; protective face, skin, hand and bokeh caps win overlaps.",
                "No face reconstruction or generative replacement; identity evaluation remains required.",
            ],
        )


def enhance_regions(
    linear: np.ndarray,
    regions: RegionMasks,
    config: RegionDetailConfig | None = None,
) -> RegionDetailResult:
    """Blend source-scale feathered masks at output size, without hard mask cuts.

    Non-protective regions use a normalized weighted mean. Protective regions
    then impose soft upper bounds using min(base, base*(1-mask)+cap*mask).
    Gaussian filtering uses reflected boundaries. Float overshoot is retained;
    clipping the master here would silently alter original-derived samples.
    """
    _validate_image(linear)
    config = config or RegionDetailConfig()
    if set(regions.masks) != set(REGIONS):
        raise ValueError("region masks must contain exactly the documented regions")
    shape = regions.masks["background"].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("region masks must be non-empty 2D arrays")
    masks = {}
    for name in REGIONS:
        _validate_mask(regions.masks[name], shape)
        masks[name] = cv2.resize(
            regions.masks[name],
            (linear.shape[1], linear.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    total = sum(masks[name] for name in ("background", "hair", "clothing"))
    weighted = sum(masks[name] * getattr(config, name) for name in ("background", "hair", "clothing"))
    strength = np.divide(weighted, total, out=np.full_like(total, config.background), where=total > 0)
    base = strength.copy()
    for name in ("skin", "hands", "primary_face", "background_faces", "bokeh"):
        cap = base * (1 - masks[name]) + getattr(config, name) * masks[name]
        np.minimum(strength, cap, out=strength)
    # Masks and weighting scratch are no longer needed when RGB filtering starts.
    del masks, total, weighted, base, cap
    output = cv2.GaussianBlur(linear, (0, 0), config.radius_pixels, borderType=cv2.BORDER_REFLECT)
    # Reuse the blur allocation for each elementwise step, retaining the original
    # float32 operation order without output-sized arithmetic temporaries.
    np.subtract(linear, output, out=output)
    np.clip(output, -config.max_detail, config.max_detail, out=output)
    np.multiply(strength[..., None], output, out=output)
    np.add(linear, output, out=output)
    return RegionDetailResult(output, strength, config)


HUMAN_REVIEW_ITEMS = (
    "extra eyes or teeth",
    "distorted ears",
    "broken glasses",
    "implausible fingers",
    "altered limb count",
    "disappearing clothing seams",
    "inconsistent background anatomy",
)


def evaluate_crowd_consistency(
    source: np.ndarray,
    candidate: np.ndarray,
    detections,
    *,
    skin_mask: np.ndarray | None = None,
    detector=None,
    duplicate_threshold: float = 0.95,
    source_match_threshold: float = 0.85,
    texture_ratio_threshold: float = 0.25,
    overlap_threshold: float = 0.3,
) -> dict:
    """Estimated classical warning signals, never anatomical or identity proof.

    Provisional thresholds require experimental calibration. Face crops are
    area-reduced to source scale, then resized to 32x32 for zero-mean normalized
    cross-correlation. Constant crops have unavailable correlation, not a match.
    Skin texture uses mask-weighted local luminance variance (3x3 reflected
    window), compared with the same samples in the source. An optional detector
    runs at source scale on both images; manual annotations are also accepted as
    source observations. Unmatched output boxes are warnings, not proven new people.
    """
    from . import baselines, quality_metrics
    from .pipeline import MEASUREMENT

    _validate_image(source)
    _validate_image(candidate)
    height, width = source.shape[:2]
    if candidate.shape[0] < height or candidate.shape[1] < width:
        raise ValueError("candidate must be at least source size on both axes")
    for value in (duplicate_threshold, source_match_threshold, texture_ratio_threshold, overlap_threshold):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("crowd thresholds must be finite numbers in [0, 1]")
    if source_match_threshold >= duplicate_threshold:
        raise ValueError("source_match_threshold must be below duplicate_threshold")
    for face in detections.faces:
        face.validate_bounds(width, height)
    if skin_mask is not None:
        _validate_mask(skin_mask, (height, width))
    reduced = candidate if candidate.shape == source.shape else baselines.downsample_area(candidate, width, height)

    def crops(image):
        return [
            cv2.resize(image[f.y : f.y + f.height, f.x : f.x + f.width], (32, 32), interpolation=cv2.INTER_AREA)
            for f in detections.faces
        ]

    def correlation(a, b):
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        a -= a.mean()
        b -= b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator > 1e-12 else None

    before, after = crops(source), crops(reduced)
    pairs = []
    for i in range(len(before)):
        for j in range(i + 1, len(before)):
            original = correlation(before[i], before[j])
            processed = correlation(after[i], after[j])
            pairs.append(
                {
                    "face_indices": [i, j],
                    "source_correlation": original,
                    "output_correlation": processed,
                    "flagged": original is not None
                    and processed is not None
                    and original < source_match_threshold
                    and processed >= duplicate_threshold,
                }
            )

    def local_variance(image):
        luma = quality_metrics.luma(image).astype(np.float64)
        mean = cv2.blur(luma, (3, 3), borderType=cv2.BORDER_REFLECT)
        return np.maximum(cv2.blur(luma * luma, (3, 3), borderType=cv2.BORDER_REFLECT) - mean * mean, 0)

    texture = []
    source_variance, output_variance = local_variance(source), local_variance(reduced)
    for index, face in enumerate(detections.faces):
        region = np.s_[face.y : face.y + face.height, face.x : face.x + face.width]
        weights = None if skin_mask is None else skin_mask[region]
        ratio = None
        reason = "not available: no skin samples or no measurable source texture"
        if weights is not None and weights.sum() > 0:
            original = float(np.average(source_variance[region], weights=weights))
            if original > 1e-12:
                ratio = float(np.average(output_variance[region], weights=weights) / original)
                reason = "Estimated local linear-luminance variance ratio at source scale"
        texture.append(
            {
                "face_index": index,
                "variance_ratio": ratio,
                "flagged": ratio is not None and ratio < texture_ratio_threshold,
                "note": reason,
            }
        )

    new_faces = None
    detection_note = "not available: no automatic detector configured"
    if detector is not None:
        source_faces = (*detections.faces, *detector(source).faces)
        output_faces = detector(reduced).faces
        for face in (*source_faces, *output_faces):
            face.validate_bounds(width, height)

        def overlap(a, b):
            intersection = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)) * max(
                0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
            )
            return intersection / (a.width * a.height + b.width * b.height - intersection)

        new_faces = [
            {"box": [f.x, f.y, f.width, f.height], "confidence": f.confidence}
            for f in output_faces
            if not any(overlap(f, s) >= overlap_threshold for s in source_faces)
        ]
        detection_note = (
            "Estimated unmatched face-like detections using source-scale intersection over union; "
            "not proof of new faces"
        )
    return {
        "classification": MEASUREMENT,
        "thresholds": {
            "duplicate": duplicate_threshold,
            "source_match": source_match_threshold,
            "texture_ratio": texture_ratio_threshold,
            "overlap_iou": overlap_threshold,
        },
        "duplicate_pairs": pairs,
        "skin_texture": texture,
        "new_face_like_detections": new_faces,
        "detection_note": detection_note,
        "requires_human_review": list(HUMAN_REVIEW_ITEMS),
        "notes": [
            "Thresholds must be calibrated experimentally; warnings are not proof of identity or anatomy.",
            "No recovered detail; unavailable checks are not passing checks.",
        ],
    }
