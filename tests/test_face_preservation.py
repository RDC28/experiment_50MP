"""Source-pixel annotations, tier boundaries and manifest provenance."""

import dataclasses
import json

import numpy as np
import pytest

from synthetic_raw_lab.face_preservation import (
    FaceBox,
    FaceTierConfig,
    ManualFaceDetector,
    face_tier,
    tier_policy,
    tier_stage_record,
)
from synthetic_raw_lab.pipeline import MEASUREMENT, PRESERVATION


@pytest.mark.parametrize(
    "size,confidence,expected",
    [(96, 0.9, 1), (95, 1, 2), (96, 0.89, 2), (32, 0.6, 2), (31, 1, 3), (100, 0.59, 3)],
)
def test_tiers_require_source_size_and_confidence(size, confidence, expected):
    assert face_tier(FaceBox(0, 0, size, size + 20, confidence)) == expected
    assert face_tier(FaceBox(0, 0, size + 20, size, confidence)) == expected


def test_thresholds_are_configurable_and_documented():
    config = FaceTierConfig(tier1_min_pixels=12, tier2_min_pixels=6)
    assert face_tier(FaceBox(0, 0, 12, 12), config) == 1
    assert face_tier(FaceBox(0, 0, 12, 12)) == 3
    for field in dataclasses.fields(config):
        for key in ("description", "accepted", "effect", "warning"):
            assert field.metadata[key]
        assert "calibrated experimentally" in field.metadata["warning"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tier1_min_pixels": True},
        {"tier2_min_pixels": 0},
        {"tier2_min_pixels": 96},
        {"tier1_min_confidence": float("nan")},
        {"tier2_min_confidence": float("inf")},
        {"tier1_min_confidence": 0.5},
        {"tier2_min_confidence": True},
    ],
)
def test_invalid_tier_config_fails(kwargs):
    with pytest.raises(ValueError):
        FaceTierConfig(**kwargs)


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 3, 3),
        (0, 0, 0, 3),
        (0.5, 0, 3, 3),
        (True, 0, 3, 3),
        (0, 0, 3, 3, -0.1),
        (0, 0, 3, 3, float("nan")),
        (0, 0, 3, 3, True),
    ],
)
def test_invalid_face_boxes_fail(args):
    with pytest.raises(ValueError):
        FaceBox(*args)


def test_manual_selection_is_preserved_without_modifying_image():
    image = np.zeros((120, 160, 3), np.float32)
    original = image.copy()
    faces = (FaceBox(0, 0, 96, 96), FaceBox(140, 100, 20, 20, 0.7))
    detections = ManualFaceDetector(faces, primary_index=1)(image)
    assert detections.primary_index == 1  # explicitly select the smaller face
    assert detections.faces == faces
    np.testing.assert_array_equal(image, original)
    record = detections.stage_record().to_dict()
    assert record["classification"] == PRESERVATION
    assert record["parameters"]["faces"][1]["box"] == [140, 100, 20, 20]
    assert "not proof of identity" in json.dumps(record)
    tiers = tier_stage_record(detections).to_dict()
    assert tiers["classification"] == MEASUREMENT
    assert tiers["parameters"]["tiers"] == [1, 3]
    assert "calibrated experimentally" in json.dumps(tiers)


def test_empty_manual_override_and_no_implicit_primary():
    image = np.zeros((10, 10, 3), np.float32)
    assert ManualFaceDetector()(image).faces == ()
    assert ManualFaceDetector((FaceBox(0, 0, 10, 10),))(image).primary_index is None
    with pytest.raises(ValueError, match="outside"):
        ManualFaceDetector((FaceBox(1, 0, 10, 10),))(image)
    with pytest.raises(ValueError, match="linear float32"):
        ManualFaceDetector()(image.astype(np.uint8))


@pytest.mark.parametrize("index", [0, -1, True, 0.5])
def test_primary_index_must_reference_a_face(index):
    with pytest.raises(ValueError, match="primary_index"):
        ManualFaceDetector(primary_index=index)


def test_duplicate_annotations_rejected():
    face = FaceBox(0, 0, 10, 10)
    with pytest.raises(ValueError, match="duplicate"):
        ManualFaceDetector((face, face))


def test_tier_policies_do_not_claim_identity():
    assert "conservative SR" in tier_policy(1).enhancement
    assert "minimal restoration" in tier_policy(1).enhancement
    assert "check when available" in tier_policy(1).landmark_check
    assert "texture regularization" in tier_policy(2).enhancement
    assert "no reconstruction" in tier_policy(3).enhancement
    for tier in (1, 2, 3):
        assert tier_policy(tier).requires_human_review
        assert "do not establish identity" in tier_policy(tier).identity_claim
    for invalid in (0, 4, True, 1.0):
        with pytest.raises(ValueError):
            tier_policy(invalid)


def test_identity_metrics_identical_faces_and_tier_summary():
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    image = np.random.default_rng(12).uniform(0.1, 0.8, (120, 160, 3)).astype(np.float32)
    detections = ManualFaceDetector((FaceBox(0, 0, 96, 96), FaceBox(100, 0, 32, 32), FaceBox(140, 100, 8, 8)), 0)(image)
    result = evaluate_face_identity(image, image.copy(), detections, skin_mask=np.ones((120, 160)))
    assert [r["face_count"] for r in result["per_tier"]] == [1, 1, 1]
    for row in result["faces"]:
        assert row["ssim"] == pytest.approx(1)
        assert row["color_difference_ciede2000"]["mean"] == pytest.approx(0)
        assert row["skin_tone_shift_ciede2000"] == pytest.approx(0)
        assert row["landmark_displacement_px"] is None
        assert "no landmark estimator" in row["landmark_note"]
        assert row["human_reviewer_judgment"] is None
    assert result["faces"][0]["primary"]
    assert not result["faces"][1]["primary"]
    json.dumps(result, allow_nan=False)


def test_identity_metrics_reduce_before_cropping_and_detect_color_drift():
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    source = np.full((24, 40, 3), 0.2, np.float32)
    source[5:15, 6:20] = 0.6
    candidate = np.repeat(np.repeat(source, 2, axis=0), 3, axis=1)
    faces = ManualFaceDetector((FaceBox(3, 2, 22, 18),))(source)
    row = evaluate_face_identity(source, candidate, faces)["faces"][0]
    assert row["ssim"] == pytest.approx(1)
    assert row["edge_displacement"]["mean_px"] == pytest.approx(0)
    assert row["skin_tone_shift_ciede2000"] is None
    candidate[..., 0] += 0.2
    changed = evaluate_face_identity(source, candidate, faces)["faces"][0]
    assert changed["ssim"] < 0.99
    assert changed["color_difference_ciede2000"]["mean"] > 1
    np.testing.assert_array_equal(source[0, 0], np.full(3, 0.2, np.float32))


def test_identity_skin_samples_are_weighted_and_empty_tiers_unavailable():
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    source = np.full((16, 16, 3), 0.2, np.float32)
    candidate = source.copy()
    candidate[8:] += 0.3
    faces = ManualFaceDetector((FaceBox(0, 0, 16, 16),))(source)
    mask = np.zeros((16, 16), np.float32)
    mask[:8] = 1
    result = evaluate_face_identity(source, candidate, faces, skin_mask=mask)
    assert result["faces"][0]["skin_tone_shift_ciede2000"] == 0
    assert result["faces"][0]["color_difference_ciede2000"]["mean"] > 1
    assert result["per_tier"][0]["mean_ssim"] is None
    assert evaluate_face_identity(source, source, ManualFaceDetector()(source))["faces"] == []
    assert (
        evaluate_face_identity(source, source, faces, skin_mask=mask * 0)["faces"][0]["skin_tone_shift_ciede2000"]
        is None
    )


@pytest.mark.parametrize("mask", [np.zeros((3, 3)), np.full((16, 16), np.nan), np.full((16, 16), -1)])
def test_identity_rejects_invalid_skin_masks(mask):
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    image = np.zeros((16, 16, 3), np.float32)
    with pytest.raises(ValueError, match="skin_mask"):
        evaluate_face_identity(image, image, ManualFaceDetector()(image), skin_mask=mask)


def test_identity_rejects_undersized_output_and_out_of_bounds_faces():
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    image = np.zeros((16, 16, 3), np.float32)
    faces = ManualFaceDetector((FaceBox(0, 0, 16, 16),))(image)
    with pytest.raises(ValueError, match="at least source size"):
        evaluate_face_identity(image, image[:8], faces)
    with pytest.raises(ValueError, match="outside"):
        evaluate_face_identity(image[:8], image, faces)


def test_primary_selection_prefers_tier_then_area_with_stable_ties():
    from synthetic_raw_lab.face_preservation import select_primary_subject

    faces = (
        FaceBox(0, 0, 110, 110, 0.5),  # largest, but low-confidence tier 3
        FaceBox(0, 0, 40, 40),
        FaceBox(0, 0, 40, 50),
        FaceBox(1, 0, 40, 50),
    )
    detections = ManualFaceDetector(faces)(np.zeros((120, 120, 3), np.float32))
    chosen = select_primary_subject(detections)
    assert chosen.primary_index == 2
    assert chosen.primary_selection_rule == "highest_source_tier_then_largest_area_then_first_observation"
    assert detections.primary_index is None
    custom = FaceTierConfig(
        tier1_min_pixels=100, tier2_min_pixels=80, tier1_min_confidence=0.4, tier2_min_confidence=0.4
    )
    assert select_primary_subject(detections, custom).primary_index == 0
    manual = ManualFaceDetector(faces, 3)(np.zeros((120, 120, 3), np.float32))
    assert select_primary_subject(manual).primary_index == 3
    assert select_primary_subject(manual).primary_selection_rule == "manual_index_override"
    empty = select_primary_subject(ManualFaceDetector()(np.zeros((4, 4, 3), np.float32)))
    assert empty.primary_index is None
    assert empty.primary_selection_rule == "no_faces_available"


def test_manual_first_merge_with_fake_detector():
    from synthetic_raw_lab.face_preservation import FaceDetections, observe_faces

    image = np.zeros((100, 100, 3), np.float32)
    manual = FaceBox(5, 5, 30, 30, 0.95)
    added = FaceBox(60, 60, 24, 24, 0.6)

    def fake(source):
        np.testing.assert_array_equal(source, image)
        return FaceDetections(
            (FaceBox(6, 6, 30, 30, 0.6), added, FaceBox(61, 61, 24, 24, 0.6)),
            None,
            "fake",
            "Inference",
            "Fake inference, not identity proof.",
        )

    result = observe_faces(image, ManualFaceDetector((manual,), 0), fake)
    assert result.faces == (manual, added)
    assert result.primary_index == 0
    assert result.classification == "Inference"
    assert observe_faces(image, ManualFaceDetector((manual,)), None).faces == (manual,)
    assert observe_faces(image, ManualFaceDetector(), fake).faces == (FaceBox(6, 6, 30, 30, 0.6), added)


def test_bundled_haar_runs_on_synthetic_crowd_and_records_exact_artifact():
    import hashlib
    from pathlib import Path

    from synthetic_raw_lab import color_management, test_images
    from synthetic_raw_lab.face_preservation import HaarFaceDetector, haar_model_metadata

    pixels, _ = test_images.make_crowd_test_image()
    source = color_management.srgb_to_linear(pixels.astype(np.float32) / 255).astype(np.float32)
    before = source.copy()
    result = HaarFaceDetector()(source)
    metadata = haar_model_metadata()
    assert metadata["packages"]
    np.testing.assert_array_equal(source, before)
    if metadata["path"] is None or not Path(metadata["path"]).is_file():
        assert metadata["availability"] == "unavailable on this OpenCV build"
        assert metadata["sha256"] is None
        assert result.faces == ()
        assert "unavailable on this OpenCV build" in result.description
        assert "manual boxes only" in result.description
        return
    assert metadata["availability"] == "available"
    assert f"{len(result.faces)} detections" in result.description
    assert "zero detections does not mean no faces" in result.description
    assert all(face.confidence == 0.6 for face in result.faces)
    for face in result.faces:
        face.validate_bounds(source.shape[1], source.shape[0])
    np.testing.assert_array_equal(source, before)
    metadata = haar_model_metadata()
    assert metadata["packages"]
    assert metadata["sha256"] == hashlib.sha256(Path(metadata["path"]).read_bytes()).hexdigest()
    assert "Intel" in Path(metadata["path"]).read_text()[:5000]
    assert metadata["classification"] == "Inference"
    assert metadata["commercial_use"].startswith("Permitted")


def test_landmarks_compare_identical_source_scale_crops():
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    source = np.full((40, 48, 3), 0.2, np.float32)
    candidate = np.full((80, 96, 3), 0.4, np.float32)
    detections = ManualFaceDetector((FaceBox(4, 8, 32, 24),))(source)
    crops = []

    def estimator(crop):
        crops.append(crop.copy())
        return np.zeros((5, 2)) + ([3, 4] if len(crops) == 2 else [0, 0])

    result = evaluate_face_identity(source, candidate, detections, landmark_estimator=estimator)
    assert [crop.shape for crop in crops] == [(24, 32, 3)] * 2
    np.testing.assert_allclose(crops[0], 0.2)
    np.testing.assert_allclose(crops[1], 0.4)
    face = result["faces"][0]
    assert face["landmark_displacement_px"] == {"mean": 5.0, "max": 5.0}
    assert face["landmark_displacement_normalized"]["mean"] == pytest.approx(5 / 24)
    assert result["per_tier"][2]["landmark_displacement_px"] == {"mean": 5.0, "max": 5.0}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("points", [None, np.zeros((4, 2)), np.full((5, 2), np.nan)])
def test_landmarks_missing_or_invalid_are_explicit(points):
    from synthetic_raw_lab.face_preservation import evaluate_face_identity

    image = np.full((32, 32, 3), 0.2, np.float32)
    detections = ManualFaceDetector((FaceBox(0, 0, 32, 32),))(image)
    result = evaluate_face_identity(image, image, detections, landmark_estimator=lambda crop: points)
    face = result["faces"][0]
    assert face["landmark_displacement_px"] is None
    assert "source crop" in face["landmark_note"]


def test_landmark_report_status_contains_numbers_and_missing_reason():
    from synthetic_raw_lab.human_run import _landmark_status

    assert (
        _landmark_status(
            {
                "face_index": 2,
                "landmark_displacement_px": {"mean": 3.0, "max": 5.0},
                "landmark_displacement_normalized": {"mean": 0.1, "max": 0.2},
            }
        )
        == "face 2: mean/max 3.0/5.0 source px; normalized 0.1/0.2"
    )
    assert "weights unavailable" in _landmark_status({"face_index": 0, "landmark_note": "weights unavailable"})


@pytest.mark.parametrize("empty", [False, True])
def test_yunet_landmarks_use_verified_model_and_crop_coordinates(monkeypatch, tmp_path, empty):
    import cv2

    from synthetic_raw_lab import human_models
    from synthetic_raw_lab.face_preservation import YuNetLandmarkEstimator, YuNetNetwork

    model = tmp_path / "verified.onnx"
    monkeypatch.setattr(human_models, "ensure_yunet_model", lambda cache_dir: model)
    rows = np.zeros((2, 15), np.float32)
    rows[:, 2:4] = [[4, 4], [12, 10]]
    rows[1, 4:14] = np.arange(10)
    calls = []

    class Detector:
        def detect(self, image):
            assert image.shape == (20, 24, 3)
            assert image.dtype == np.uint8
            return None, None if empty else rows

    def create(*args):
        calls.append(args)
        return Detector()

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", create)
    result = YuNetLandmarkEstimator(YuNetNetwork(tmp_path))(np.full((20, 24, 3), 0.2, np.float32))
    assert calls == [(str(model), "", (24, 20), 0.9, 0.3, 5000)]
    if empty:
        assert result is None
    else:
        np.testing.assert_array_equal(result, np.arange(10).reshape(5, 2))


def test_yunet_landmark_weights_failure_is_explicit(monkeypatch, tmp_path):
    from synthetic_raw_lab import human_models
    from synthetic_raw_lab.face_preservation import YuNetLandmarkEstimator, YuNetNetwork

    def unavailable(cache_dir):
        raise RuntimeError("runtime weights unavailable: offline")

    monkeypatch.setattr(human_models, "ensure_yunet_model", unavailable)
    with pytest.raises(RuntimeError, match="YuNet landmarks unavailable: runtime weights unavailable: offline"):
        YuNetLandmarkEstimator(YuNetNetwork(tmp_path))(np.full((20, 24, 3), 0.2, np.float32))
