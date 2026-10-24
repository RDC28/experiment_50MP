"""Source-scale face observations and conservative tier policies (PRD 8.2, 11.2).

Boxes and confidence describe observations, never proof of identity. Manual
annotations are preserved user input; size/confidence tiers are estimated.
Thresholds are provisional and must be calibrated experimentally. This module
does not reconstruct faces or implement generative replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import _param
from .pipeline import MEASUREMENT, PRESERVATION, StageRecord


@dataclass(frozen=True)
class FaceBox:
    """An unscaled source-pixel rectangle; confidence is not an identity score.

    Coordinates use [x, y, width, height], with exclusive right/bottom edges.
    Manual confidence means annotation certainty, not detector probability.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise ValueError("face coordinates must be integers")
        if self.x < 0 or self.y < 0 or self.width < 1 or self.height < 1:
            raise ValueError("face origin must be non-negative and dimensions positive")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not np.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("face confidence must be finite and between 0 and 1")

    @property
    def source_size(self) -> int:
        """Shortest side, so a narrow crop cannot masquerade as high information."""
        return min(self.width, self.height)

    def validate_bounds(self, width: int, height: int) -> None:
        if self.x + self.width > width or self.y + self.height > height:
            raise ValueError("face box extends outside the source image")


@dataclass(frozen=True)
class FaceTierConfig:
    """Provisional thresholds, to be calibrated experimentally, not universal constants.

    Field metadata uses the shared configuration parameter-table conventions.
    Confidence from different detectors must be calibrated independently.
    """

    tier1_min_pixels: int = _param(
        96,
        "Minimum source shortest side for tier 1.",
        "integer >= 2",
        "Requires sufficient source structure for conservative comparison.",
        warning="To be calibrated experimentally; lowering can overstate face reliability.",
    )
    tier2_min_pixels: int = _param(
        32,
        "Minimum source shortest side for tier 2.",
        "integer >= 1 and below tier1_min_pixels",
        "Smaller faces retain silhouette only.",
        warning="To be calibrated experimentally; no upscaled pixel count is used.",
    )
    tier1_min_confidence: float = _param(
        0.9,
        "Minimum observation confidence for tier 1.",
        "0 to 1",
        "Low-certainty large faces cannot enter tier 1.",
        warning="To be calibrated experimentally per detector; not an identity probability.",
    )
    tier2_min_confidence: float = _param(
        0.6,
        "Minimum observation confidence for tier 2.",
        "0 to tier1_min_confidence",
        "Lower-confidence observations receive tier 3 protection.",
        warning="To be calibrated experimentally per detector; manual certainty differs from detection confidence.",
    )

    def __post_init__(self) -> None:
        sizes = (self.tier1_min_pixels, self.tier2_min_pixels)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in sizes):
            raise ValueError("tier pixel thresholds must be integers")
        if not 1 <= self.tier2_min_pixels < self.tier1_min_pixels:
            raise ValueError("tier pixel thresholds must satisfy 1 <= tier2 < tier1")
        scores = (self.tier1_min_confidence, self.tier2_min_confidence)
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or not 0 <= v <= 1
            for v in scores
        ):
            raise ValueError("tier confidence thresholds must be finite and between 0 and 1")
        if self.tier2_min_confidence > self.tier1_min_confidence:
            raise ValueError("tier2 confidence threshold must not exceed tier1")


def face_tier(face: FaceBox, config: FaceTierConfig | None = None) -> int:
    """Estimate a protection tier using source size AND observation confidence."""
    config = config or FaceTierConfig()
    if face.source_size >= config.tier1_min_pixels and face.confidence >= config.tier1_min_confidence:
        return 1
    if face.source_size >= config.tier2_min_pixels and face.confidence >= config.tier2_min_confidence:
        return 2
    return 3


@dataclass(frozen=True)
class FacePolicy:
    tier: int
    enhancement: str
    landmark_check: str
    identity_claim: str = "none: detection and consistency metrics do not establish identity"
    requires_human_review: bool = True


def tier_policy(tier: int) -> FacePolicy:
    """Declarative processing requirements for the forthcoming enhancement stage."""
    if not isinstance(tier, int) or isinstance(tier, bool) or tier not in (1, 2, 3):
        raise ValueError("face tier must be 1, 2 or 3")
    return {
        1: FacePolicy(1, "conservative SR; minimal restoration", "check when available; report reason otherwise"),
        2: FacePolicy(2, "limited sharpening; texture regularization; no generative restoration", "when available"),
        3: FacePolicy(3, "preserve silhouette and broad appearance; no reconstruction", "insufficient source detail"),
    }[tier]


@dataclass(frozen=True)
class FaceDetections:
    faces: tuple[FaceBox, ...]
    primary_index: int | None
    method: str
    classification: str
    description: str
    primary_selection_rule: str = "unassigned"

    def stage_record(self) -> StageRecord:
        return StageRecord(
            name="face_observations",
            classification=self.classification,
            parameters={
                "method": self.method,
                "primary_index": self.primary_index,
                "primary_selection_rule": self.primary_selection_rule,
                "faces": [{"box": [f.x, f.y, f.width, f.height], "confidence": f.confidence} for f in self.faces],
            },
            input_precision="float32 linear at source scale",
            output_precision="source-pixel boxes",
            notes=[self.description, "Detections are not proof of identity; tiers are estimated, not inferred detail."],
        )


def select_primary_subject(detections: FaceDetections, config: FaceTierConfig | None = None) -> FaceDetections:
    """Prefer highest source tier, then largest area, then first observation.

    An explicit index always wins. Selection is a protection heuristic, never
    an assertion of identity or of the photographer's intended subject.
    """
    from dataclasses import replace

    if detections.primary_index is not None:
        if not 0 <= detections.primary_index < len(detections.faces):
            raise ValueError("primary_index must reference a face")
        return replace(detections, primary_selection_rule="manual_index_override")
    if not detections.faces:
        return replace(detections, primary_selection_rule="no_faces_available")
    index = min(
        range(len(detections.faces)),
        key=lambda i: (
            face_tier(detections.faces[i], config),
            -detections.faces[i].width * detections.faces[i].height,
            i,
        ),
    )
    return replace(
        detections,
        primary_index=index,
        primary_selection_rule="highest_source_tier_then_largest_area_then_first_observation",
    )


class FaceDetector(Protocol):
    """Return source-scale observations without mutating the linear RGB image."""

    def __call__(self, linear: np.ndarray) -> FaceDetections: ...


@dataclass(frozen=True)
class ManualFaceDetector:
    """Preserve manual boxes for the manual-first merge.

    This adapter preserves annotations only. The orchestration merges automatic
    observations and selects a primary subject afterwards.
    """

    faces: tuple[FaceBox, ...] = ()
    primary_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.faces, tuple) or any(not isinstance(f, FaceBox) for f in self.faces):
            raise ValueError("faces must be a tuple of FaceBox observations")
        if len({(f.x, f.y, f.width, f.height) for f in self.faces}) != len(self.faces):
            raise ValueError("duplicate manual face observations")
        index = self.primary_index
        if index is not None and (
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.faces)
        ):
            raise ValueError("primary_index must reference a manual face or be None")

    def __call__(self, linear: np.ndarray) -> FaceDetections:
        from .human_pipeline import _validate_image

        _validate_image(linear)
        for face in self.faces:
            face.validate_bounds(linear.shape[1], linear.shape[0])
        return FaceDetections(
            self.faces,
            self.primary_index,
            "manual",
            PRESERVATION,
            "Preserved manual source-pixel annotations; confidence is user certainty, not identity probability.",
        )


@dataclass(frozen=True)
class HaarFaceDetector:
    """Bundled frontal Haar inference, with no training or model download.

    Accepted rectangles receive a conservative fixed 0.6 observation score,
    not a calibrated probability or identity score. This caps automatic faces
    at tier 2 with the default thresholds. Frontal/occluded faces may be missed.
    """

    def __call__(self, linear: np.ndarray) -> FaceDetections:
        import cv2

        from .color_management import linear_to_srgb
        from .human_pipeline import _validate_image

        _validate_image(linear)
        model = haar_model_metadata()

        def unavailable(reason):
            return FaceDetections((), None, "haar", "Inference", f"Haar {reason}; manual boxes only.")

        if model["availability"] != "available":
            return unavailable(model["availability"])
        cascade = cv2.CascadeClassifier(model["path"])
        if cascade.empty():
            return unavailable("unavailable on this OpenCV build: bundled XML could not be loaded")
        encoded = np.round(np.clip(linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8)
        gray = cv2.cvtColor(encoded, cv2.COLOR_RGB2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        faces = tuple(FaceBox(*(int(v) for v in box), confidence=0.6) for box in sorted(map(tuple, boxes)))
        return FaceDetections(
            faces,
            None,
            "haar",
            "Inference",
            f"Bundled frontal Haar inference: {len(faces)} detections. "
            "Fixed confidence 0.6 is an uncalibrated acceptance score, never identity proof; "
            "zero detections does not mean no faces. Inspect and add manual boxes.",
        )


def haar_model_metadata() -> dict:
    """Audit the installed artifact, retaining explicit absence provenance."""
    import hashlib
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    import cv2

    directory = getattr(getattr(cv2, "data", None), "haarcascades", None)
    path = Path(directory) / "haarcascade_frontalface_default.xml" if directory else None
    digest = None
    availability = "available"
    try:
        if path is None:
            raise FileNotFoundError("cv2.data.haarcascades is absent")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        availability = "unavailable on this OpenCV build"
    package_versions = {}
    for package in ("opencv-python", "opencv-python-headless"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return {
        "name": "OpenCV frontal face Haar cascade",
        "classification": "Inference",
        "packages": package_versions,
        "opencv_version": cv2.__version__,
        "file_name": "haarcascade_frontalface_default.xml",
        "path": str(path) if path else None,
        "sha256": digest,
        "availability": availability,
        "license": "Intel Open Source Computer Vision License (BSD-3-style)",
        "license_text_location": f"{path}: opening XML comment; see MODELS.md",
        "commercial_use": "Permitted with copyright, license and disclaimer retention; no endorsement.",
        "confidence": "Fixed uncalibrated acceptance score 0.6; not identity proof.",
        "parameters": {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": [24, 24]},
    }


def merge_face_detections(
    manual: FaceDetections, automatic: FaceDetections, *, iou_threshold: float = 0.3
) -> FaceDetections:
    """Preserve manual order/coordinates/scores/index, then append unique detections.

    IoU >= 0.3 suppresses a later observation; automatic duplicates are also
    suppressed in detector order. Manual observations are never removed.
    """
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    def iou(a, b):
        overlap = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)) * max(
            0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
        )
        return overlap / (a.width * a.height + b.width * b.height - overlap)

    faces = list(manual.faces)
    for face in automatic.faces:
        if all(iou(face, previous) < iou_threshold for previous in faces):
            faces.append(face)
    return FaceDetections(
        tuple(faces),
        manual.primary_index,
        "manual+" + automatic.method,
        "Inference",
        manual.description
        + " "
        + automatic.description
        + f" Manual observations take precedence; IoU >= {iou_threshold} removes later detections. "
        f"Merged total: {len(faces)} faces.",
    )


def observe_faces(linear: np.ndarray, manual: ManualFaceDetector, detector: FaceDetector | None) -> FaceDetections:
    """Shared manual-first observation path, injectable for inference tests."""
    observations = manual(linear)
    if detector is None:
        return observations
    automatic = detector(linear)
    for face in automatic.faces:
        face.validate_bounds(linear.shape[1], linear.shape[0])
    return merge_face_detections(observations, automatic)


def tier_stage_record(faces: FaceDetections, config: FaceTierConfig | None = None) -> StageRecord:
    """Record the calibrated inputs and estimated tier assignments in the manifest."""
    from dataclasses import asdict

    config = config or FaceTierConfig()
    return StageRecord(
        name="face_tiers",
        classification=MEASUREMENT,
        parameters={"thresholds": asdict(config), "tiers": [face_tier(f, config) for f in faces.faces]},
        notes=["Estimated tiers using source shortest side and confidence; to be calibrated experimentally."],
    )


class LandmarkEstimator(Protocol):
    """Return five ordered crop-local (x, y) points, or None when unavailable.

    Implementations may raise RuntimeError with an acquisition/inference reason.
    The same estimator receives source and reduced-output crops in that order.
    """

    def __call__(self, linear: np.ndarray) -> np.ndarray | None: ...


def _landmark_measurement(source, candidate, face_size, estimator):
    if estimator is None:
        return None, None, "not available: no landmark estimator configured"
    points = []
    for label, crop in (("source", source), ("output", candidate)):
        try:
            found = estimator(crop)
        except RuntimeError as error:
            return None, None, f"not available: {label} crop: {error}"
        if found is None:
            return None, None, f"not available: no landmark found in {label} crop"
        found = np.asarray(found, dtype=np.float64)
        if found.shape != (5, 2) or not np.isfinite(found).all():
            return None, None, f"not available: invalid five-point landmarks in {label} crop"
        points.append(found)
    distances = np.linalg.norm(points[1] - points[0], axis=1)
    pixels = {"mean": float(distances.mean()), "max": float(distances.max())}
    normalized = {key: value / face_size for key, value in pixels.items()}
    return pixels, normalized, "Inference: five-point displacement; normalized by source shortest face side"


def evaluate_face_identity(
    source: np.ndarray,
    candidate: np.ndarray,
    detections: FaceDetections,
    *,
    tier_config: FaceTierConfig | None = None,
    skin_mask: np.ndarray | None = None,
    landmark_estimator: LandmarkEstimator | None = None,
) -> dict:
    """Measure consistency per face and tier, never establish personal identity.

    Reduce the entire candidate with the shared area filter before cropping so
    every metric is in source coordinates. Color/SSIM/edges use the shared sRGB
    evaluation domain. Optional source-scale skin weights select CIEDE2000
    samples; without an annotation, skin-tone shift is explicitly unavailable.
    Linear overshoot is clipped only by the shared display-domain metrics.
    Optional landmarks use identical source-scale crops; embeddings remain disabled.
    """
    from . import baselines, quality_metrics
    from .human_pipeline import _validate_image

    _validate_image(source)
    _validate_image(candidate)
    height, width = source.shape[:2]
    if candidate.shape[0] < height or candidate.shape[1] < width:
        raise ValueError("candidate must be at least source size on both axes")
    if skin_mask is not None:
        if (
            not isinstance(skin_mask, np.ndarray)
            or skin_mask.shape != (height, width)
            or not np.issubdtype(skin_mask.dtype, np.number)
            or not np.isfinite(skin_mask).all()
            or np.any((skin_mask < 0) | (skin_mask > 1))
        ):
            raise ValueError("skin_mask must contain finite source-scale weights in [0, 1]")
    for face in detections.faces:
        face.validate_bounds(width, height)
    reduced = candidate if candidate.shape == source.shape else baselines.downsample_area(candidate, width, height)
    rows = []
    for index, face in enumerate(detections.faces):
        region = np.s_[face.y : face.y + face.height, face.x : face.x + face.width]
        metrics, delta_e = quality_metrics.source_consistency(source[region], reduced[region])
        weights = None if skin_mask is None else skin_mask[region]
        skin_shift = None if weights is None or weights.sum() == 0 else float(np.average(delta_e, weights=weights))
        landmark_px, landmark_normalized, landmark_note = _landmark_measurement(
            source[region], reduced[region], face.source_size, landmark_estimator
        )
        rows.append(
            {
                "face_index": index,
                "tier": face_tier(face, tier_config),
                "primary": index == detections.primary_index,
                "box": [face.x, face.y, face.width, face.height],
                "ssim": metrics["ssim"],
                "color_difference_ciede2000": metrics["color_difference_ciede2000"],
                "skin_tone_shift_ciede2000": skin_shift,
                "skin_tone_note": (
                    "Estimated mask-weighted mean CIEDE2000, not a signed color change"
                    if skin_shift is not None
                    else "not available: no skin mask or no skin samples inside this face"
                ),
                "edge_displacement": metrics["edge_displacement"],
                "landmark_displacement_px": landmark_px,
                "landmark_displacement_normalized": landmark_normalized,
                "landmark_note": landmark_note,
                "embedding_similarity": None,
                "embedding_note": "disabled by default; no recognition model loaded",
                "human_reviewer_judgment": None,
                "identity_claim": "none; source consistency is not proof of identity",
            }
        )
    tiers = []
    for tier in (1, 2, 3):
        selected = [row for row in rows if row["tier"] == tier]
        available = [row for row in selected if row["landmark_displacement_px"] is not None]
        landmark_summary = {}
        for key in ("landmark_displacement_px", "landmark_displacement_normalized"):
            landmark_summary[key] = (
                {
                    "mean": float(np.mean([row[key]["mean"] for row in available])),
                    "max": max(row[key]["max"] for row in available),
                }
                if available
                else None
            )
        tiers.append(
            {
                "tier": tier,
                "face_count": len(selected),
                "landmark_face_count": len(available),
                **landmark_summary,
                "mean_ssim": float(np.mean([row["ssim"] for row in selected])) if selected else None,
                "mean_ciede2000": (
                    float(np.mean([row["color_difference_ciede2000"]["mean"] for row in selected]))
                    if selected
                    else None
                ),
            }
        )
    return {
        "classification": MEASUREMENT,
        "domain": "sRGB at source scale; candidate area-averaged before cropping",
        "faces": rows,
        "per_tier": tiers,
        "notes": [
            "Estimated consistency metrics, not identity verification or recovered detail.",
            "Tier 3 preserves broad appearance only; no identity claims.",
            "Missing edges yield unavailable displacement; a missing metric is not a passing check.",
        ],
    }


@dataclass(frozen=True)
class IdentityThresholds:
    """Provisional source-consistency gates, to be calibrated experimentally.

    Missing optional measurements stay unavailable and require human review;
    they never become evidence that identity has been verified.
    """

    min_ssim: float = _param(
        0.8,
        "Minimum face SSIM.",
        "-1 to 1",
        "Flags structural drift.",
        warning="To be calibrated experimentally per tier; not identity verification.",
    )
    max_color_difference: float = _param(
        5.0,
        "Maximum mean face CIEDE2000.",
        "finite >= 0",
        "Flags face color drift.",
        warning="Provisional; calibrate experimentally.",
    )
    max_skin_shift: float = _param(
        4.0,
        "Maximum mask-weighted skin CIEDE2000.",
        "finite >= 0",
        "Flags skin-tone drift when measured.",
        warning="Estimated masks require review.",
    )
    max_edge_displacement: float = _param(
        1.0,
        "Maximum mean source-pixel edge displacement.",
        "finite >= 0",
        "Flags geometric drift when edges exist.",
        warning="Missing edges are unavailable, not passing evidence.",
    )
    max_landmark_displacement_normalized: float = _param(
        0.05,
        "Maximum five-point landmark displacement normalized by source face shortest side.",
        "finite >= 0",
        "Flags geometric drift when the maximum landmark displacement exceeds this fraction.",
        warning="To be calibrated experimentally; unavailable landmarks require human review.",
    )
    retry_strength: float = _param(
        0.25,
        "Enhancement fraction retained on retry.",
        "0 to below 1",
        "Blends enhancement toward original-derived Lanczos.",
        warning="Higher strengths can retain facial alteration.",
    )

    def __post_init__(self) -> None:
        from dataclasses import asdict

        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if not -1 <= self.min_ssim <= 1 or not 0 <= self.retry_strength < 1:
            raise ValueError("invalid SSIM or retry strength range")
        if (
            min(
                self.max_color_difference,
                self.max_skin_shift,
                self.max_edge_displacement,
                self.max_landmark_displacement_normalized,
            )
            < 0
        ):
            raise ValueError("maximum metric thresholds must be non-negative")


def _identity_failures(row: dict, thresholds: IdentityThresholds) -> list[str]:
    landmarks = row["landmark_displacement_normalized"]
    checks = (
        (
            "landmark_displacement",
            None if landmarks is None else landmarks["max"],
            thresholds.max_landmark_displacement_normalized,
            False,
        ),
        ("ssim", row["ssim"], thresholds.min_ssim, True),
        ("color_difference", row["color_difference_ciede2000"]["mean"], thresholds.max_color_difference, False),
        ("skin_shift", row["skin_tone_shift_ciede2000"], thresholds.max_skin_shift, False),
        ("edge_displacement", row["edge_displacement"]["mean_px"], thresholds.max_edge_displacement, False),
    )
    return [
        name
        for name, value, limit, lower in checks
        if value is not None and (value < limit if lower else value > limit)
    ]


@dataclass(frozen=True)
class IdentityRecoveryResult:
    linear: np.ndarray
    evaluation: dict
    events: list[dict]
    stage: StageRecord


def preserve_face_identity(
    source: np.ndarray,
    candidate: np.ndarray,
    detections: FaceDetections,
    *,
    thresholds: IdentityThresholds | None = None,
    tier_config: FaceTierConfig | None = None,
    skin_mask: np.ndarray | None = None,
    landmark_estimator: LandmarkEstimator | None = None,
) -> IdentityRecoveryResult:
    """PRD 11.6: flag, reduce enhancement, remeasure, then use Lanczos if needed.

    Only flagged rectangles and a feathered exterior margin change. The full
    face box receives the retry/fallback, including its boundary pixels. The
    baseline is always recomputed from source, never trusted caller data.
    Final metrics remain reportable even when the baseline itself fails gates.
    No generative face replacement or identity claim is implemented.
    """
    from dataclasses import asdict

    from scipy.ndimage import distance_transform_edt

    from .baselines import resample

    thresholds = thresholds or IdentityThresholds()

    def evaluate(image):
        return evaluate_face_identity(
            source,
            image,
            detections,
            tier_config=tier_config,
            skin_mask=skin_mask,
            landmark_estimator=landmark_estimator,
        )

    initial = evaluate(candidate)  # validates source, candidate, boxes and skin weights
    output = candidate.copy()
    events = []
    baseline = None
    height, width = candidate.shape[:2]
    for row, face in zip(initial["faces"], detections.faces, strict=True):
        failures = _identity_failures(row, thresholds)
        if not failures:
            continue
        if baseline is None:
            baseline = resample(source, width, height, "lanczos")
        # Cover every target pixel intersecting the source-pixel rectangle.
        x0 = face.x * width // source.shape[1]
        y0 = face.y * height // source.shape[0]
        x1 = int(np.ceil((face.x + face.width) * width / source.shape[1]))
        y1 = int(np.ceil((face.y + face.height) * height / source.shape[0]))
        outside = np.ones((height, width), dtype=bool)
        outside[y0:y1, x0:x1] = False
        feather = max(1, min(x1 - x0, y1 - y0) // 8)
        weight = np.maximum(0, 1 - distance_transform_edt(outside) / feather).astype(np.float32)[..., None]
        before = output.copy()
        retry = baseline + thresholds.retry_strength * (before - baseline)
        output = before + (retry - before) * weight
        retry_row = evaluate(output)["faces"][row["face_index"]]
        persistent = _identity_failures(retry_row, thresholds)
        steps = [
            {"action": "flag", "metrics": row, "failed_metrics": failures},
            {
                "action": "reduced_enhancement_retry",
                "strength": thresholds.retry_strength,
                "metrics": retry_row,
                "failed_metrics": persistent,
            },
        ]
        if persistent:
            output = output + (baseline - output) * weight
            output[y0:y1, x0:x1] = baseline[y0:y1, x0:x1]
            fallback_row = evaluate(output)["faces"][row["face_index"]]
            steps.append(
                {
                    "action": "original_derived_lanczos_fallback",
                    "metrics": fallback_row,
                    "failed_metrics": _identity_failures(fallback_row, thresholds),
                }
            )
        events.append({"face_index": row["face_index"], "tier": row["tier"], "steps": steps})
    final = evaluate(output)
    stage = StageRecord(
        name="identity_failure_recovery",
        classification="Simulation",
        parameters={"thresholds": asdict(thresholds), "events": events},
        input_precision="float32 linear",
        output_precision="float32 linear",
        notes=[
            "Reduced simulated enhancement with source-derived Interpolation fallback; no recovered detail.",
            "Final metrics must still be reviewed; unavailable landmarks and embeddings do not pass identity.",
            "Overlapping face repairs may affect earlier faces; final evaluation rechecks every face.",
        ],
    )
    return IdentityRecoveryResult(output, final, events, stage)


@dataclass(frozen=True)
class YuNetFaceDetector:
    """Verified runtime YuNet inference at source scale; never identity proof."""

    def __call__(self, linear: np.ndarray) -> FaceDetections:
        import cv2

        from .color_management import linear_to_srgb
        from .human_models import ensure_yunet_model
        from .human_pipeline import _validate_image

        _validate_image(linear)
        try:
            model = ensure_yunet_model()
            height, width = linear.shape[:2]
            detector = cv2.FaceDetectorYN.create(str(model), "", (width, height), 0.9, 0.3, 5000)
            encoded = np.round(np.clip(linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8)
            _, rows = detector.detect(cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR))
            faces = []
            for row in () if rows is None else rows:
                # Clip inferred rectangles to source bounds before downstream crops.
                x0, y0 = np.maximum(0, np.floor(row[:2])).astype(int)
                x1 = min(width, int(np.ceil(row[0] + row[2])))
                y1 = min(height, int(np.ceil(row[1] + row[3])))
                if x1 > x0 and y1 > y0:
                    faces.append(FaceBox(int(x0), int(y0), x1 - int(x0), y1 - int(y0), float(row[14])))
            return FaceDetections(
                tuple(faces),
                None,
                "yunet",
                "Inference",
                f"YuNet inference: {len(faces)} detections; uncalibrated confidence, never identity proof. "
                "Zero detections does not mean no faces; inspect and add manual boxes.",
            )
        except (RuntimeError, OSError, AttributeError, cv2.error) as error:
            return FaceDetections((), None, "yunet", "Inference", f"YuNet unavailable: {error}; manual boxes only.")


class YuNetLandmarkEstimator:
    """Infer ordered five-point landmarks in a source-scale face crop.

    Uses the same verified model and inference settings as YuNet face detection.
    The largest detected face is selected to avoid choosing a smaller neighbour.
    These inferred points measure geometric consistency, never identity proof.
    """

    def __call__(self, linear: np.ndarray) -> np.ndarray | None:
        import cv2

        from .color_management import linear_to_srgb
        from .human_models import ensure_yunet_model
        from .human_pipeline import _validate_image

        _validate_image(linear)
        try:
            model = ensure_yunet_model()
            height, width = linear.shape[:2]
            detector = cv2.FaceDetectorYN.create(str(model), "", (width, height), 0.9, 0.3, 5000)
            encoded = np.round(np.clip(linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8)
            _, rows = detector.detect(cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR))
            if rows is None or len(rows) == 0:
                return None
            row = max(rows, key=lambda face: float(face[2]) * float(face[3]))
            return np.asarray(row[4:14], dtype=np.float64).reshape(5, 2)
        except (RuntimeError, OSError, AttributeError, cv2.error) as error:
            raise RuntimeError(f"YuNet landmarks unavailable: {error}") from error
