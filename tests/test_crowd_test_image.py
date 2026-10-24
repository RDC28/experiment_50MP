"""Synthetic crowd fixture and its source-scale processing integration."""

import numpy as np
import pytest

from synthetic_raw_lab.color_management import srgb_to_linear
from synthetic_raw_lab.crowd_processing import enhance_regions
from synthetic_raw_lab.face_preservation import ManualFaceDetector, evaluate_face_identity, face_tier
from synthetic_raw_lab.human_pipeline import lanczos_stand_in
from synthetic_raw_lab.region_masks import estimate_masks
from synthetic_raw_lab.test_images import make_crowd_test_image


def test_crowd_fixture_supports_all_tiers_and_processing():
    encoded, boxes = make_crowd_test_image()
    assert encoded.shape == (256, 384, 3)
    assert encoded.dtype == np.uint8
    assert [face_tier(face) for face in boxes] == [1, 2, 3]
    source = srgb_to_linear(encoded.astype(np.float32) / 255)
    detections = ManualFaceDetector(boxes, primary_index=0)(source)
    masks = estimate_masks(source, detections)
    for name in ("skin", "hair", "clothing", "primary_face", "background_faces", "bokeh"):
        assert masks.masks[name].max() > 0
    for face in boxes:
        hand_y = min(encoded.shape[0], face.y + face.height * 2) - max(2, face.height // 8)
        assert masks.masks["skin"][hand_y, face.x + face.width // 2] > 0
    upscaled = lanczos_stand_in(source, 576, 384, "conservative")
    result = enhance_regions(upscaled.linear, masks)
    metrics = evaluate_face_identity(source, result.linear, detections, skin_mask=masks.skin_samples)
    assert [row["face_count"] for row in metrics["per_tier"]] == [1, 1, 1]
    assert all(np.isfinite(row["ssim"]) and -1 <= row["ssim"] <= 1 for row in metrics["faces"])
    assert all(row["skin_tone_shift_ciede2000"] is not None for row in metrics["faces"])
    # The empty corner is intentionally smooth for bokeh protection checks.
    assert source[:20, :20].std(axis=(0, 1)).max() < 0.01


def test_crowd_fixture_is_reproducible_and_seed_changes_clothing_only():
    first, boxes = make_crowd_test_image()
    repeated, repeated_boxes = make_crowd_test_image()
    changed, changed_boxes = make_crowd_test_image(seed=43)
    np.testing.assert_array_equal(first, repeated)
    assert boxes == repeated_boxes == changed_boxes
    assert not np.array_equal(first, changed)
    for face in boxes:
        region = np.s_[face.y : face.y + face.height, face.x : face.x + face.width]
        np.testing.assert_array_equal(first[region], changed[region])


@pytest.mark.parametrize("width,height", [(256, 256), (512, 256), (384, 512)])
def test_crowd_boxes_fit_other_aspect_ratios(width, height):
    image, boxes = make_crowd_test_image(width, height)
    assert image.shape == (height, width, 3)
    for face in boxes:
        face.validate_bounds(width, height)


@pytest.mark.parametrize("kwargs", [{"width": 255}, {"height": True}, {"width": 300.5}, {"seed": -1}, {"seed": True}])
def test_crowd_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        make_crowd_test_image(**kwargs)


@pytest.mark.parametrize("width,height,counts", [(384, 256, [1, 1, 1]), (3840, 2160, [1, 3, 10])])
def test_crowd_source_pixel_tiers_and_non_overlapping_faces(width, height, counts):
    image, boxes = make_crowd_test_image(width, height)
    assert image.shape == (height, width, 3)
    assert [sum(face_tier(face) == tier for face in boxes) for tier in (1, 2, 3)] == counts
    for index, face in enumerate(boxes):
        face.validate_bounds(width, height)
        for other in boxes[index + 1 :]:
            assert (
                face.x + face.width <= other.x
                or other.x + other.width <= face.x
                or face.y + face.height <= other.y
                or other.y + other.height <= face.y
            )


def test_crowd_sizes_follow_configured_source_thresholds():
    from synthetic_raw_lab.face_preservation import FaceTierConfig

    config = FaceTierConfig(tier1_min_pixels=72, tier2_min_pixels=24)
    _, boxes = make_crowd_test_image(tier_config=config)
    assert [face_tier(face, config) for face in boxes] == [1, 2, 3]
    assert boxes[0].source_size >= config.tier1_min_pixels * 4 / 3
