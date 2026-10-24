"""Source-scale region estimates, manual precedence and visualization contracts."""

import json

import numpy as np
import pytest

from synthetic_raw_lab.face_preservation import FaceBox, ManualFaceDetector
from synthetic_raw_lab.pipeline import MEASUREMENT
from synthetic_raw_lab.region_masks import REGIONS, estimate_masks, mask_overlay


def scene():
    image = np.full((100, 160, 3), [0.5, 0.25, 0.12], np.float32)
    faces = ManualFaceDetector((FaceBox(10, 15, 30, 30), FaceBox(100, 20, 20, 20)), 0)(image)
    return image, faces


def test_masks_cover_all_regions_with_feathered_source_scale_weights():
    image, faces = scene()
    before = image.copy()
    result = estimate_masks(image, faces)
    assert tuple(result.masks) == REGIONS
    for name, mask in result.masks.items():
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.float32
        assert np.isfinite(mask).all() and mask.min() >= 0 and mask.max() <= 1
        if name not in ("hands", "hair"):
            assert mask.max() > 0
    assert result.skin_samples[30, 25] > 0.99
    assert result.masks["primary_face"][30, 25] > 0.99
    assert result.masks["background_faces"][30, 110] > 0.99
    assert result.masks["primary_face"][30, 110] == 0
    assert np.any((result.masks["primary_face"] > 0) & (result.masks["primary_face"] < 1))
    assert not result.masks["hands"].any()
    np.testing.assert_array_equal(image, before)
    record = result.stage_record().to_dict()
    assert record["classification"] == MEASUREMENT
    assert "manual-only" in json.dumps(record)


def test_manual_masks_replace_exactly_and_update_background():
    image, faces = scene()
    hands = np.ones(image.shape[:2], np.float32)
    result = estimate_masks(image, faces, overrides={"hands": hands})
    np.testing.assert_array_equal(result.masks["hands"], hands)
    assert not result.masks["background"].any()
    assert not result.masks["bokeh"].any()
    hands[:] = 0
    assert result.masks["hands"].min() == 1
    explicit = estimate_masks(image, faces, overrides={"background": hands, "bokeh": hands})
    assert not explicit.masks["bokeh"].any()
    assert explicit.overridden == ("background", "bokeh")


def test_no_faces_and_texture_rejects_bokeh():
    image = np.full((40, 40, 3), 0.2, np.float32)
    empty = ManualFaceDetector()(image)
    masks = estimate_masks(image, empty).masks
    np.testing.assert_allclose(masks["background"] + masks["bokeh"], 1)
    assert 0 < masks["bokeh"].min() <= masks["bokeh"].max() < 1
    image[::2] = 0.9
    assert estimate_masks(image, empty).masks["bokeh"].max() == 0


def test_boundary_faces_and_unassigned_primary():
    image = np.zeros((8, 8, 3), np.float32)
    faces = ManualFaceDetector((FaceBox(0, 0, 1, 1), FaceBox(6, 6, 2, 2)))(image)
    masks = estimate_masks(image, faces, feather_pixels=0).masks
    assert masks["primary_face"].max() == 0
    assert masks["background_faces"][0, 0] == 1
    assert masks["background_faces"][7, 7] == 1


@pytest.mark.parametrize("value", [-1, 33, float("nan"), True])
def test_invalid_feather_rejected(value):
    with pytest.raises(ValueError):
        estimate_masks(*scene(), feather_pixels=value)


@pytest.mark.parametrize("value", [0, -1, 2, float("inf"), True])
def test_invalid_variance_rejected(value):
    with pytest.raises(ValueError):
        estimate_masks(*scene(), bokeh_variance=value)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2), np.float32),
        np.zeros((100, 160)),
        np.full((100, 160), np.nan, np.float32),
        np.full((100, 160), 1.1, np.float32),
    ],
)
def test_invalid_overrides_rejected(mask):
    with pytest.raises(ValueError, match="mask"):
        estimate_masks(*scene(), overrides={"skin": mask})


def test_unknown_override_rejected():
    with pytest.raises(ValueError, match="unknown"):
        estimate_masks(*scene(), overrides={"typo": np.zeros((100, 160), np.float32)})


def test_every_mask_can_be_visualized_without_mutating_inputs():
    image, faces = scene()
    for mask in estimate_masks(image, faces).masks.values():
        before = mask.copy()
        overlay = mask_overlay(image, mask)
        assert overlay.shape == image.shape and overlay.dtype == np.uint8
        np.testing.assert_array_equal(mask, before)
    black = np.zeros((2, 2, 3), np.float32)
    np.testing.assert_array_equal(
        mask_overlay(black, np.ones((2, 2), np.float32)), np.full((2, 2, 3), [128, 0, 128], np.uint8)
    )


def test_colour_constraints_and_partition_reproduce_host_mask_spill():
    from synthetic_raw_lab.color_management import srgb_to_linear

    encoded = np.full((100, 100, 3), 0.8, np.float32)
    encoded[15:22, 40:60] = 0.05  # hair against a light background
    encoded[22:42, 40:60] = [0.65, 0.4, 0.25]
    encoded[42:75, 40:60] = [0.1, 0.2, 0.6]  # narrow garment
    image = srgb_to_linear(encoded).astype(np.float32)
    faces = ManualFaceDetector((FaceBox(40, 22, 20, 20),), 0)(image)
    masks = estimate_masks(image, faces).masks
    assert masks["hair"][18, 50] > 0
    assert masks["hair"][18, 38] == 0  # old halo included background
    assert masks["clothing"][50, 50] > 0
    assert masks["clothing"][50, 35] == 0  # old ellipse spilled sideways
    assert np.max(sum(masks.values())) <= 1.000001
    person = np.maximum.reduce([masks[name] for name in REGIONS[:6]]) > 0
    assert not masks["bokeh"][person].any()
    assert not np.array_equal(masks["bokeh"], masks["background"])
    assert masks["skin"][32, 50] == 0  # face takes precedence


def test_manual_capacity_wins_and_incompatible_overrides_are_rejected():
    image, faces = scene()
    manual = np.full(image.shape[:2], 0.75, np.float32)
    masks = estimate_masks(image, faces, overrides={"clothing": manual}).masks
    np.testing.assert_array_equal(masks["clothing"], manual)
    assert masks["primary_face"][30, 25] == 0.25
    assert np.max(sum(masks.values())) <= 1.000001
    with pytest.raises(ValueError, match="sum to at most 1"):
        estimate_masks(image, faces, overrides={"hair": manual, "clothing": manual})


def test_connected_hair_covers_cap_inside_face_box_without_claiming_skin():
    from synthetic_raw_lab.color_management import srgb_to_linear

    encoded = np.full((100, 100, 3), 0.8, np.float32)
    encoded[22:62, 30:70] = [0.65, 0.4, 0.25]
    encoded[17:34, 30:70] = [0.08, 0.05, 0.04]
    encoded[38:42, 75:80] = [0.08, 0.05, 0.04]  # disconnected dark object
    image = srgb_to_linear(encoded).astype(np.float32)
    faces = ManualFaceDetector((FaceBox(30, 22, 40, 40),), 0)(image)
    masks = estimate_masks(image, faces, feather_pixels=0).masks
    assert masks["hair"][30, 50] == 1  # well below the old shallow head band
    assert masks["primary_face"][30, 50] == 0
    assert masks["primary_face"][45, 50] == 1
    assert masks["hair"][40, 77] == 0
    assert masks["hair"][16, 50] == 0
    assert np.max(sum(masks.values())) <= 1.000001


def test_exposed_skin_matches_face_chroma_with_protective_precedence():
    from synthetic_raw_lab.color_management import srgb_to_linear

    encoded = np.full((120, 120, 3), [0.1, 0.3, 0.6], np.float32)
    encoded[20:40, 45:65] = [0.72, 0.45, 0.3]
    encoded[40:46, 51:59] = [0.72, 0.45, 0.3]  # neck
    encoded[52:67, 30:35] = [0.576, 0.36, 0.24]  # shaded arm, same chroma
    encoded[70:74, 53:57] = [0.72, 0.45, 0.3]  # disconnected hand
    encoded[55:60, 70:75] = [0.7, 0.65, 0.1]  # warm non-skin colour
    encoded[100:105, 53:57] = [0.72, 0.45, 0.3]  # outside local body window
    image = srgb_to_linear(encoded).astype(np.float32)
    faces = ManualFaceDetector((FaceBox(45, 20, 20, 20),), 0)(image)
    masks = estimate_masks(image, faces, feather_pixels=0).masks
    for row, col in ((43, 55), (60, 32), (72, 55)):
        assert masks["skin"][row, col] == 1
    for row, col in ((30, 55), (57, 72), (102, 55), (50, 55)):
        assert masks["skin"][row, col] == 0
    np.testing.assert_allclose(sum(masks.values()), 1, atol=1e-6)
