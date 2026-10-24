"""Human failure recovery on small end-to-end source/candidate arrays."""

import json
from dataclasses import fields

import numpy as np
import pytest

from synthetic_raw_lab.face_preservation import (
    FaceBox,
    IdentityThresholds,
    ManualFaceDetector,
    preserve_face_identity,
)


def fixture():
    source = np.full((48, 64, 3), 0.2, np.float32)
    source[12:28, 12:28] = 0.6
    detections = ManualFaceDetector((FaceBox(8, 8, 24, 24),), 0)(source)
    candidate = source.copy()
    candidate[8:32, 8:32, 0] += 0.3
    return source, candidate, detections


def test_persistent_drift_falls_back_and_records_every_comparison():
    source, candidate, faces = fixture()
    original = candidate.copy()
    result = preserve_face_identity(source, candidate, faces, thresholds=IdentityThresholds(max_color_difference=0.1))
    assert [step["action"] for step in result.events[0]["steps"]] == [
        "flag",
        "reduced_enhancement_retry",
        "original_derived_lanczos_fallback",
    ]
    assert result.events[0]["steps"][1]["failed_metrics"]
    assert result.events[0]["steps"][2]["failed_metrics"] == []
    np.testing.assert_array_equal(result.linear, source)
    np.testing.assert_array_equal(candidate, original)
    assert result.evaluation["faces"][0]["ssim"] == pytest.approx(1)
    assert result.stage.classification == "Simulation"
    json.dumps(result.stage.to_dict(), allow_nan=False)


def test_successful_retry_retains_reduced_enhancement_and_unmodified_background():
    source, candidate, faces = fixture()
    result = preserve_face_identity(source, candidate, faces, thresholds=IdentityThresholds(max_color_difference=10))
    assert len(result.events[0]["steps"]) == 2
    assert result.events[0]["steps"][1]["failed_metrics"] == []
    np.testing.assert_allclose(result.linear[16, 16], source[16, 16] + [0.075, 0, 0])
    np.testing.assert_array_equal(result.linear[40:], candidate[40:])
    assert result.evaluation["faces"][0]["landmark_displacement_px"] is None


def test_passing_and_empty_faces_are_noops_with_independent_output():
    source, _, faces = fixture()
    for observations in (faces, ManualFaceDetector()(source)):
        result = preserve_face_identity(source, source, observations)
        assert result.events == []
        np.testing.assert_array_equal(result.linear, source)
        assert not np.shares_memory(result.linear, source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_ssim": 2},
        {"retry_strength": 1},
        {"retry_strength": -0.1},
        {"max_color_difference": -1},
        {"max_skin_shift": float("nan")},
        {"max_edge_displacement": True},
    ],
)
def test_invalid_gates_rejected(kwargs):
    with pytest.raises(ValueError):
        IdentityThresholds(**kwargs)


def test_thresholds_have_parameter_documentation():
    for field in fields(IdentityThresholds):
        for key in ("description", "accepted", "effect", "warning"):
            assert field.metadata[key]


def test_upscaled_recovery_reports_all_faces_and_preserves_baseline_failure():
    from synthetic_raw_lab.baselines import resample
    from synthetic_raw_lab.color_management import srgb_to_linear
    from synthetic_raw_lab.test_images import make_crowd_test_image

    # Extremely strict gates can reject Lanczos itself; do not hide that fact.
    image, boxes = make_crowd_test_image()
    source = srgb_to_linear(image.astype(np.float32) / 255)
    faces = ManualFaceDetector(boxes, 0)(source)
    candidate = resample(source, source.shape[1] * 2, source.shape[0] * 2, "lanczos")
    result = preserve_face_identity(source, candidate, faces, thresholds=IdentityThresholds(min_ssim=1))
    assert result.events
    assert len(result.evaluation["faces"]) == len(faces.faces)
    assert all(event["steps"][-1]["failed_metrics"] for event in result.events)
    assert [row["face_count"] for row in result.evaluation["per_tier"]] == [1, 1, 1]


@pytest.mark.parametrize("persistent", [False, True])
def test_landmark_only_drift_triggers_remeasurement_and_fallback(persistent):
    source, _, faces = fixture()
    calls = 0

    def estimator(crop):
        nonlocal calls
        calls += 1
        points = np.zeros((5, 2))
        # Source and candidate crops alternate on every evaluation. Only one
        # displaced point proves the gate uses maximum, rather than mean drift.
        if calls % 2 == 0 and (persistent or calls == 2):
            points[0, 0] = 2.4
        return points

    result = preserve_face_identity(source, source, faces, landmark_estimator=estimator)
    steps = result.events[0]["steps"]
    assert steps[0]["failed_metrics"] == ["landmark_displacement"]
    assert steps[0]["metrics"]["landmark_displacement_normalized"]["max"] == pytest.approx(0.1)
    assert len(steps) == (3 if persistent else 2)
    assert steps[-1]["failed_metrics"] == (["landmark_displacement"] if persistent else [])
    assert calls == (8 if persistent else 6)
    assert result.evaluation["per_tier"][2]["landmark_face_count"] == 1


def test_landmark_gate_accepts_boundary_and_preserves_missing_reason():
    source, _, faces = fixture()
    calls = 0

    def estimator(crop):
        nonlocal calls
        calls += 1
        return np.full((5, 2), [1.5 if calls % 2 == 0 else 0, 0])

    result = preserve_face_identity(
        source,
        source,
        faces,
        thresholds=IdentityThresholds(max_landmark_displacement_normalized=1.5 / 24),
        landmark_estimator=estimator,
    )
    assert result.events == []
    missing = preserve_face_identity(source, source, faces, landmark_estimator=lambda crop: None)
    assert missing.events == []
    assert "no landmark found in source crop" in missing.evaluation["faces"][0]["landmark_note"]


@pytest.mark.parametrize("value", [-0.01, float("inf"), True])
def test_invalid_landmark_threshold_rejected(value):
    with pytest.raises(ValueError):
        IdentityThresholds(max_landmark_displacement_normalized=value)


@pytest.mark.parametrize("width,height,exact", [(80, 60, True), (120, 90, True), (97, 71, False), (41, 33, False)])
def test_face_metrics_follow_a_whole_candidate_area_reduction(width, height, exact):
    from synthetic_raw_lab import baselines, quality_metrics
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    rows, columns = np.mgrid[0:30, 0:40]
    source = np.stack(
        [0.2 + 0.6 * columns / 40, 0.2 + 0.5 * rows / 30, 0.5 + 0.3 * np.sin(columns / 5) * np.cos(rows / 7)], -1
    ).astype(np.float32)
    detections = ManualFaceDetector((FaceBox(0, 0, 12, 10), FaceBox(17, 9, 15, 13), FaceBox(25, 18, 15, 12)))(source)
    candidate = baselines.resample(source, width, height, "lanczos")
    reduced = baselines.downsample_area(candidate, 40, 30)
    measured = evaluate_face_identity(source, candidate, detections)["faces"]
    for row, face in zip(measured, detections.faces, strict=True):
        region = np.s_[face.y : face.y + face.height, face.x : face.x + face.width]
        expected, _ = quality_metrics.source_consistency(source[region], reduced[region])
        tolerance = 0 if exact else 1e-3
        assert row["ssim"] == pytest.approx(expected["ssim"], abs=tolerance)
        assert row["color_difference_ciede2000"]["mean"] == pytest.approx(
            expected["color_difference_ciede2000"]["mean"], abs=50 * tolerance
        )


def test_recovery_remeasures_only_the_flagged_face():
    source = np.full((48, 96, 3), 0.2, np.float32)
    boxes = (FaceBox(8, 8, 24, 24), FaceBox(40, 8, 24, 24), FaceBox(70, 20, 20, 20))
    detections = ManualFaceDetector(boxes, 0)(source)
    candidate = source.copy()
    candidate[8:32, 8:32, 0] += 0.3
    calls = []

    def landmarks(crop):
        calls.append(crop.shape)
        return np.zeros((5, 2))

    result = preserve_face_identity(
        source,
        candidate,
        detections,
        thresholds=IdentityThresholds(max_color_difference=0.1),
        landmark_estimator=landmarks,
    )
    assert [event["face_index"] for event in result.events] == [0]
    remeasured = len(result.events[0]["steps"]) - 1
    assert len(calls) == 2 * len(boxes) + 2 * remeasured + 2 * len(boxes)
    np.testing.assert_array_equal(result.linear[:, 40:], candidate[:, 40:])
