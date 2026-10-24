"""Classical crowd warning signals on small synthetic arrays."""

import json

import numpy as np
import pytest

from synthetic_raw_lab.crowd_processing import evaluate_crowd_consistency
from synthetic_raw_lab.face_preservation import FaceBox, ManualFaceDetector


def fixture():
    image = np.random.default_rng(10).uniform(0.1, 0.8, (32, 64, 3)).astype(np.float32)
    faces = ManualFaceDetector((FaceBox(0, 0, 24, 24), FaceBox(32, 0, 24, 24)))(image)
    return image, faces


def test_duplicate_introduced_but_existing_similarity_not_flagged():
    image, faces = fixture()
    candidate = image.copy()
    candidate[:24, 32:56] = image[:24, :24]
    result = evaluate_crowd_consistency(image, candidate, faces)
    assert result["duplicate_pairs"][0]["flagged"]
    assert not evaluate_crowd_consistency(candidate, candidate, faces)["duplicate_pairs"][0]["flagged"]
    json.dumps(result, allow_nan=False)


def test_texture_flattening_and_unchanged_texture():
    image, faces = fixture()
    mask = np.ones(image.shape[:2], np.float32)
    unchanged = evaluate_crowd_consistency(image, image, faces, skin_mask=mask)
    assert all(row["variance_ratio"] == pytest.approx(1) for row in unchanged["skin_texture"])
    candidate = np.full_like(image, 0.4)
    result = evaluate_crowd_consistency(image, candidate, faces, skin_mask=mask)
    assert all(row["flagged"] for row in result["skin_texture"])
    assert result["duplicate_pairs"][0]["output_correlation"] is None
    assert result["new_face_like_detections"] is None
    assert len(result["requires_human_review"]) == 7


def test_missing_skin_and_flat_source_are_unavailable():
    image, faces = fixture()
    for mask in (None, np.zeros(image.shape[:2], np.float32), np.ones(image.shape[:2], np.float32)):
        result = evaluate_crowd_consistency(np.zeros_like(image), image, faces, skin_mask=mask)
        assert all(row["variance_ratio"] is None and not row["flagged"] for row in result["skin_texture"])


def test_output_detector_compared_with_both_manual_and_automatic_source_boxes():
    image, faces = fixture()
    candidate = image + np.float32(0.01)
    extra = FaceBox(50, 25, 8, 7)

    def detector(linear):
        boxes = (faces.faces[0], extra) if linear is candidate else (faces.faces[1],)
        return ManualFaceDetector(boxes)(linear)

    result = evaluate_crowd_consistency(image, candidate, faces, detector=detector)
    assert result["new_face_like_detections"] == [{"box": [50, 25, 8, 7], "confidence": 1.0}]
    assert evaluate_crowd_consistency(image, image, faces, detector=detector)["new_face_like_detections"] == []


def test_area_reduction_preserves_checks_and_inputs():
    image, faces = fixture()
    original = image.copy()
    candidate = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    result = evaluate_crowd_consistency(image, candidate, faces)
    assert not any(row["flagged"] for row in result["duplicate_pairs"])
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duplicate_threshold": float("nan")},
        {"texture_ratio_threshold": -1},
        {"overlap_threshold": True},
        {"source_match_threshold": 1},
        {"skin_mask": np.ones((2, 2))},
    ],
)
def test_invalid_controls_rejected(kwargs):
    image, faces = fixture()
    with pytest.raises(ValueError):
        evaluate_crowd_consistency(image, image, faces, **kwargs)
