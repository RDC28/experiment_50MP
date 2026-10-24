"""Scene analysis and content classification for the universal notebook (PRD 4.3, 20.2 and Phase 4).

Measures scene statistics on the source (on a reduced copy, which is enough for statistics), observes faces
through the same manual-first path as notebook 01 (manual boxes plus optional YuNet or Haar inference), and
classifies the scene into the category whose pipeline processes it:

- ``human``: face- or crowd-dominant, by day or night. Processed by the notebook 01 human and crowd pipeline
  (:mod:`human_run`), whose identity protection is the strongest face policy the project has.
- ``night``: a night or low-light scene without faces. Processed by the notebook 02 night pipeline
  (:mod:`night_pipeline`).
- ``night_with_faces``: a night scene with faces that do not dominate it (PRD 20.2, both present). Processed by
  the universal common pipeline with night policies outside the face masks and face policies inside them.
- ``general``: everything else, and every low-confidence decision (PRD 20.2: use conservative universal
  settings and ask for confirmation). Processed by the universal common pipeline.

Every decision records its cues: the measured value, the threshold, whether it passed and the relative
margin. The thresholds are provisional, to be calibrated experimentally (PRD 25.3). Classification is a
processing heuristic, never a statement about who or what is in the image, and zero detected faces does not
mean there are no faces.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import baselines
from . import color_management as cm
from . import face_preservation as faces
from . import night_processing as night
from .quality_metrics import luma

CATEGORIES = ("human", "night", "night_with_faces", "general")
OVERRIDES = ("auto", *CATEGORIES)
FACE_DETECTORS = ("yunet", "haar", "none")
PIPELINES = {
    "human": "notebook 01 human and crowd pipeline (src/human_run.py)",
    "night": "notebook 02 night pipeline (src/night_pipeline.py)",
    "night_with_faces": "universal common pipeline: night policies outside the face masks, face policies inside",
    "general": "universal common pipeline: conservative universal settings",
}
NEAR_BLACK_LUMA = night.RegionThresholds().near_black
ANALYSIS_NOTE = (
    "Statistics are measured on an area-reduced copy of the source (sRGB-encoded luma); faces are observed at "
    "source resolution. Thresholds are provisional, to be calibrated experimentally. Classification is a "
    "processing heuristic, not a statement about the content, and zero detections does not mean no faces."
)


@dataclass(frozen=True)
class SceneThresholds:
    """Provisional classification thresholds; luma values are sRGB-encoded in [0, 1]."""

    night_max_median_luma: float = 0.3
    night_min_dark_fraction: float = 0.5
    dark_luma: float = 0.25
    human_min_face_px: int = 96
    crowd_min_faces: int = 4
    crowd_min_face_px: int = 32
    min_face_confidence: float = 0.6
    low_confidence_margin: float = 0.15
    max_edge: int = 1536

    def __post_init__(self) -> None:
        unit = ("night_max_median_luma", "night_min_dark_fraction", "dark_luma", "min_face_confidence")
        for name in unit:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ValueError(f"analysis threshold {name} must be a number between 0 and 1, got {value!r}")
        for name in ("human_min_face_px", "crowd_min_faces", "crowd_min_face_px", "max_edge"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"analysis threshold {name} must be a positive integer, got {value!r}")
        margin = self.low_confidence_margin
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin < 1:
            raise ValueError(f"analysis threshold low_confidence_margin must be in [0, 1), got {margin!r}")
        if self.max_edge < 64:
            raise ValueError("analysis threshold max_edge must be at least 64")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def observe_faces(
    linear: np.ndarray,
    detector: str = "yunet",
    weights_dir: str | Path = "weights",
    manual_boxes=(),
) -> faces.FaceDetections:
    """Manual boxes first, then the optional detector, merged and with a primary subject (notebook 01 path)."""
    if detector not in FACE_DETECTORS:
        raise ValueError(f"face detector must be one of {FACE_DETECTORS}, got {detector!r}")
    manual = faces.ManualFaceDetector(tuple(faces.FaceBox(*(box)) for box in manual_boxes))
    automatic = {
        "yunet": lambda: faces.YuNetFaceDetector(faces.YuNetNetwork(weights_dir)),
        "haar": faces.HaarFaceDetector,
        "none": lambda: None,
    }[detector]()
    return faces.select_primary_subject(faces.observe_faces(linear, manual, automatic))


def scene_statistics(
    linear: np.ndarray, detections: faces.FaceDetections, thresholds: SceneThresholds | None = None
) -> dict:
    """PRD 20 scene statistics: tone distribution, luminance regions, colourfulness and observed faces."""
    t = thresholds or SceneThresholds()
    height, width = linear.shape[:2]
    scale = min(1.0, t.max_edge / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reduced = linear if scale == 1.0 else baselines.downsample_area(linear, *size)
    encoded = cm.linear_to_srgb(np.clip(reduced, 0, 1))
    y = luma(encoded)
    high, low = encoded.max(axis=2), encoded.min(axis=2)
    lit = high > 0.05
    saturation = float(((high - low)[lit] / high[lit]).mean()) if lit.any() else 0.0
    regions = night.segment_regions(np.ascontiguousarray(reduced, dtype=np.float32)).fractions()
    return {
        "measured_on": f"{size[0]}x{size[1]} area-reduced copy of the {width}x{height} source",
        "luma": {
            "mean": float(y.mean()),
            "p05": float(np.percentile(y, 5)),
            "median": float(np.median(y)),
            "p95": float(np.percentile(y, 95)),
        },
        "dark_fraction": float((y < t.dark_luma).mean()),
        "near_black_fraction": float((y < NEAR_BLACK_LUMA).mean()),
        "mean_saturation": saturation,
        "regions": regions,
        "faces": face_summary(detections, width, height, t),
    }


def face_summary(detections: faces.FaceDetections, width: int, height: int, thresholds: SceneThresholds) -> dict:
    """Faces counted for classification: observations at or above ``min_face_confidence``."""
    counted = [f for f in detections.faces if f.confidence >= thresholds.min_face_confidence]
    largest = max(counted, key=lambda f: f.width * f.height, default=None)
    return {
        "method": detections.method,
        "observed": len(detections.faces),
        "counted": len(counted),
        "min_confidence": thresholds.min_face_confidence,
        "crowd_faces": sum(f.source_size >= thresholds.crowd_min_face_px for f in counted),
        "largest_face_px": max((f.source_size for f in counted), default=0),
        "largest_face_box": [largest.x, largest.y, largest.width, largest.height] if largest else None,
        "largest_face_area_fraction": largest.width * largest.height / (width * height) if largest else 0.0,
        "total_face_area_fraction": sum(f.width * f.height for f in counted) / (width * height),
        "boxes": [[f.x, f.y, f.width, f.height, round(f.confidence, 4)] for f in detections.faces],
        "description": detections.description,
    }


def _cue(name: str, value: float, threshold: float, at_most: bool) -> dict:
    """One classification cue; ``margin`` is the signed distance from the threshold relative to it."""
    margin = (threshold - value) / threshold if at_most else (value - threshold) / threshold
    return {
        "cue": name,
        "value": value,
        "threshold": threshold,
        "rule": f"{'<=' if at_most else '>='} {threshold:g}",
        "passed": margin >= 0,
        "margin": margin,
    }


def classify(
    statistics: dict,
    thresholds: SceneThresholds | None = None,
    *,
    override: str = "auto",
    automatic: bool = True,
    night_detection: bool = True,
    face_detection: bool = True,
) -> dict:
    """Classify a scene from :func:`scene_statistics` into one of :data:`CATEGORIES` and record why.

    A scene is human-dominant when its largest counted face reaches ``human_min_face_px`` (PRD 11.2 tier 1,
    provisional) or at least ``crowd_min_faces`` faces reach ``crowd_min_face_px``; a human-dominant scene is
    ``human`` by day or night. Otherwise night needs every night cue (median luma and dark fraction). A
    decision closer to its threshold than ``low_confidence_margin`` is low confidence and the scene is processed
    with the conservative general settings (PRD 20.2).
    """
    t = thresholds or SceneThresholds()
    if override not in OVERRIDES:
        raise ValueError(f"manual_override must be one of {OVERRIDES}, got {override!r}")
    face = statistics["faces"]
    night_cues = [
        _cue("median luma", statistics["luma"]["median"], t.night_max_median_luma, at_most=True),
        _cue(
            f"share of pixels darker than luma {t.dark_luma:g}",
            statistics["dark_fraction"],
            t.night_min_dark_fraction,
            False,
        ),
    ]
    human_cues = [
        _cue(
            "largest face, shortest side (source px)", float(face["largest_face_px"]), float(t.human_min_face_px), False
        ),
        _cue(
            f"faces of at least {t.crowd_min_face_px} px", float(face["crowd_faces"]), float(t.crowd_min_faces), False
        ),
    ]
    is_night = night_detection and all(c["passed"] for c in night_cues)
    # An AND is as certain as its weakest pass when it holds, and as its clearest failure when it does not.
    night_margin = min(c["margin"] for c in night_cues) if is_night else max(-c["margin"] for c in night_cues)
    human = any(c["passed"] for c in human_cues)
    if human:
        human_margin = max(c["margin"] for c in human_cues)
    elif face["counted"] == 0:
        human_margin = 1.0  # nothing to weigh; the detector's limits are stated in the notes
    else:
        human_margin = min(-c["margin"] for c in human_cues)
    present = face["counted"] > 0
    if human:
        detected = "human"
    elif is_night:
        detected = "night_with_faces" if present else "night"
    else:
        detected = "general"
    # The human decision always counts; the night decision only decides scenes that are not human-dominant.
    decisive = [("human", human_margin)]
    if night_detection and not human:
        decisive.append(("night", night_margin))
    weakest, margin = min(decisive, key=lambda item: item[1])
    low = margin < t.low_confidence_margin

    reasons = []
    if not night_detection:
        reasons.append("night detection is off, so the scene is treated as not night")
    else:
        verdicts = "; ".join(
            f"{c['cue']} {c['value']:.3f} ({c['rule']}: {'yes' if c['passed'] else 'no'})" for c in night_cues
        )
        reasons.append(f"{'night' if is_night else 'not night'}: {verdicts}")
    if not face_detection:
        reasons.append("automatic face detection is off; only manual face boxes count")
    if present:
        reasons.append(
            f"{face['counted']} face(s) at confidence >= {t.min_face_confidence:g}; largest "
            f"{face['largest_face_px']} px shortest side, {face['crowd_faces']} of at least {t.crowd_min_face_px} px"
        )
    else:
        reasons.append(f"no face at confidence >= {t.min_face_confidence:g} ({face['observed']} observed)")
    reasons.append(
        f"{'human-dominant' if human else 'not human-dominant'}: largest face >= {t.human_min_face_px} px "
        f"or at least {t.crowd_min_faces} faces >= {t.crowd_min_face_px} px"
        + (" (human-dominant scenes go to the human pipeline by day or night)" if human and is_night else "")
    )

    category, confirm = detected, False
    if override != "auto":
        category = override
        reasons.append(f"manual override: {override} (automatic classification said {detected})")
    elif not automatic:
        category, confirm = "general", True
        reasons.append("automatic classification is off and no manual override is set: conservative general settings")
    elif low:
        category, confirm = "general", True
        reasons.append(
            f"low confidence: the {weakest} decision is {margin:.1%} from its threshold (below "
            f"{t.low_confidence_margin:.0%}), so the conservative general settings are used (PRD 20.2); "
            f"confirm with manual_override (automatic classification said {detected})"
        )
    return {
        "category": category,
        "detected": detected,
        "pipeline": PIPELINES[category],
        "night": is_night,
        "faces_present": present,
        "human_dominant": human,
        "confidence": "manual" if override != "auto" else ("low" if low or not automatic else "high"),
        "decision_margin": margin,
        "needs_confirmation": confirm,
        "override": override,
        "cues": {"night": night_cues, "human": human_cues},
        "thresholds": t.to_dict(),
        "reasons": reasons,
        "note": ANALYSIS_NOTE,
    }
