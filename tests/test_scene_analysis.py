"""Scene statistics and classification for the universal notebook (PRD 20.2)."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import night_processing as night
from synthetic_raw_lab import scene_analysis
from synthetic_raw_lab.face_preservation import FaceBox, FaceDetections

T = scene_analysis.SceneThresholds()


def detections(*boxes) -> FaceDetections:
    faces = tuple(FaceBox(*box) for box in boxes)
    return FaceDetections(faces, 0 if faces else None, "manual", "Preservation", "test boxes")


def statistics(median: float, dark: float, *boxes, width: int = 2000, height: int = 1000) -> dict:
    return {
        "luma": {"mean": median, "p05": 0.0, "median": median, "p95": 1.0},
        "dark_fraction": dark,
        "faces": scene_analysis.face_summary(detections(*boxes), width, height, T),
    }


def test_night_scene_without_faces_is_night_with_high_confidence():
    result = scene_analysis.classify(statistics(0.08, 0.9))
    assert result["category"] == result["detected"] == "night"
    assert result["confidence"] == "high" and not result["needs_confirmation"]
    assert result["night"] and not result["faces_present"]
    assert all(c["passed"] for c in result["cues"]["night"])
    assert result["pipeline"] == scene_analysis.PIPELINES["night"]


def test_night_scene_with_a_face_gets_both_policies():
    result = scene_analysis.classify(statistics(0.08, 0.9, (100, 100, 40, 40)))
    assert result["category"] == "night_with_faces"
    assert result["faces_present"] and not result["human_dominant"]


def test_large_face_in_daylight_is_human():
    result = scene_analysis.classify(statistics(0.45, 0.1, (100, 100, 300, 320)))
    assert result["category"] == "human" and result["confidence"] == "high"
    assert result["cues"]["human"][0]["value"] == 300.0


def test_human_dominant_night_scene_goes_to_the_human_pipeline():
    result = scene_analysis.classify(statistics(0.08, 0.9, (100, 100, 300, 320)))
    assert result["category"] == "human" and result["night"]
    assert any("by day or night" in reason for reason in result["reasons"])


def test_crowd_of_small_faces_is_human():
    boxes = [(100 * i, 100, 40, 40) for i in range(1, 6)]
    result = scene_analysis.classify(statistics(0.45, 0.1, *boxes))
    assert result["category"] == "human"
    assert result["cues"]["human"][1]["passed"]


def test_low_confidence_faces_do_not_count():
    result = scene_analysis.classify(statistics(0.45, 0.1, (100, 100, 300, 300, 0.3)))
    assert result["category"] == "general" and not result["faces_present"]


def test_decision_close_to_a_threshold_is_low_confidence_and_conservative():
    result = scene_analysis.classify(statistics(0.45, 0.1, (100, 100, 100, 100)))
    assert result["detected"] == "human"
    assert result["category"] == "general"
    assert result["confidence"] == "low" and result["needs_confirmation"]
    assert any("low confidence" in reason for reason in result["reasons"])

    near_night = scene_analysis.classify(statistics(0.28, 0.9))
    assert near_night["detected"] == "night" and near_night["category"] == "general"


def test_manual_override_wins_and_is_recorded():
    result = scene_analysis.classify(statistics(0.45, 0.1), override="night")
    assert result["category"] == "night" and result["detected"] == "general"
    assert result["confidence"] == "manual" and not result["needs_confirmation"]
    with pytest.raises(ValueError, match="manual_override"):
        scene_analysis.classify(statistics(0.45, 0.1), override="portrait")


def test_disabled_classification_and_night_detection():
    off = scene_analysis.classify(statistics(0.08, 0.9), automatic=False)
    assert off["category"] == "general" and off["needs_confirmation"]
    no_night = scene_analysis.classify(statistics(0.08, 0.9), night_detection=False)
    assert no_night["category"] == "general" and not no_night["night"]


@pytest.mark.parametrize(
    "change",
    [{"night_max_median_luma": 0}, {"crowd_min_faces": 0}, {"low_confidence_margin": 1.0}, {"max_edge": 32}],
)
def test_thresholds_are_validated(change):
    with pytest.raises(ValueError):
        scene_analysis.SceneThresholds(**change)


def test_statistics_of_a_night_scene_and_a_bright_scene():
    dark = cm.srgb_to_linear(night.make_night_scene(192, 128).astype(np.float32) / 255)
    stats = scene_analysis.scene_statistics(dark, detections())
    assert stats["luma"]["median"] < 0.25 and stats["dark_fraction"] > 0.55
    assert set(stats["regions"]) == set(night.REGION_NAMES)
    assert scene_analysis.classify(stats)["category"] == "night"

    bright = np.full((64, 96, 3), 0.4, np.float32)
    stats = scene_analysis.scene_statistics(bright, detections((10, 10, 20, 30)))
    assert stats["dark_fraction"] == 0.0
    assert stats["faces"]["counted"] == 1 and stats["faces"]["largest_face_px"] == 20
    assert stats["faces"]["largest_face_area_fraction"] == pytest.approx(600 / (64 * 96))


def test_statistics_use_a_reduced_copy_of_large_sources():
    image = np.full((300, 900, 3), 0.2, np.float32)
    stats = scene_analysis.scene_statistics(image, detections(), scene_analysis.SceneThresholds(max_edge=300))
    assert stats["measured_on"].startswith("300x100 area-reduced copy of the 900x300 source")


def test_observe_faces_keeps_manual_boxes_without_a_detector():
    image = np.full((64, 96, 3), 0.4, np.float32)
    found = scene_analysis.observe_faces(image, "none", "unused", [[10, 10, 20, 20]])
    assert [(f.x, f.y, f.width, f.height) for f in found.faces] == [(10, 10, 20, 20)]
    assert found.primary_index == 0
    with pytest.raises(ValueError, match="face detector"):
        scene_analysis.observe_faces(image, "mtcnn")
