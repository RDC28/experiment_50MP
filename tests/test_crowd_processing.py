"""Small-image checks for controlled, protective region blending."""

from dataclasses import fields, replace

import numpy as np
import pytest

from synthetic_raw_lab.crowd_processing import RegionDetailConfig, enhance_regions
from synthetic_raw_lab.region_masks import REGIONS, RegionMasks


def masks(**weights):
    return RegionMasks(
        {name: np.full((8, 10), weights.get(name, 0), np.float32) for name in REGIONS},
        (),
        2,
        0.0004,
    )


def image():
    source = np.zeros((16, 20, 3), np.float32)
    source[:, 10:] = [0.8, 0.6, 0.4]
    return source


def test_detail_is_bounded_and_does_not_mutate_inputs():
    source = image()
    original = source.copy()
    regions = masks(background=1)
    result = enhance_regions(source, regions)
    assert result.linear.dtype == np.float32
    assert np.any(result.linear != source)
    assert np.max(np.abs(result.linear - source)) <= 0.005001
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(regions.masks["background"], np.ones((8, 10)))
    record = result.stage_record().to_dict()
    assert record["classification"] == "Simulation"
    assert "no recovered detail" in record["notes"][0]
    assert record["parameters"]["background_faces"] == 0


@pytest.mark.parametrize("region", ["primary_face", "background_faces", "bokeh"])
def test_protected_regions_override_overlapping_high_detail(region):
    source = image()
    result = enhance_regions(source, masks(background=1, clothing=1, hair=1, **{region: 1}))
    np.testing.assert_array_equal(result.linear, source)
    assert not result.strength_map.any()


@pytest.mark.parametrize("region", ["skin", "hands"])
def test_low_detail_protective_caps(region):
    result = enhance_regions(image(), masks(hair=1, **{region: 1}))
    np.testing.assert_allclose(result.strength_map, 0.05)


def test_feathered_weights_remain_continuous_after_resize():
    regions = masks(background=1)
    regions.masks["primary_face"][:] = np.linspace(0, 1, 10)
    result = enhance_regions(image(), regions)
    row = result.strength_map[8]
    assert row[0] == 0.25 and row[-1] == 0
    assert np.all(np.diff(row) <= 0)
    assert np.max(np.abs(np.diff(row))) < 0.02


def test_zero_strength_and_constant_input_are_preserved():
    config = RegionDetailConfig(**dict.fromkeys(REGIONS, 0.0))
    np.testing.assert_array_equal(enhance_regions(image(), masks(background=1), config).linear, image())
    constant = np.full((16, 20, 3), 1.2, np.float32)
    np.testing.assert_allclose(enhance_regions(constant, masks(background=1)).linear, constant, atol=1e-7)


def test_overlap_is_normalized_and_uncovered_pixels_use_background():
    config = replace(RegionDetailConfig(), hair=0.4, clothing=0.2)
    np.testing.assert_allclose(enhance_regions(image(), masks(hair=1, clothing=1), config).strength_map, 0.3)
    np.testing.assert_allclose(enhance_regions(image(), masks(), config).strength_map, 0.25)


@pytest.mark.parametrize("field", fields(RegionDetailConfig))
@pytest.mark.parametrize("invalid", [True, float("nan"), -1, 9, "low"])
def test_config_rejects_invalid_values(field, invalid):
    with pytest.raises(ValueError, match=field.name):
        RegionDetailConfig(**{field.name: invalid})


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_invalid_masks_are_rejected(bad):
    regions = masks()
    regions.masks["hands"][0, 0] = bad
    with pytest.raises(ValueError, match="mask must"):
        enhance_regions(image(), regions)


def test_missing_mask_is_not_silently_ignored():
    regions = masks()
    del regions.masks["skin"]
    with pytest.raises(ValueError, match="exactly"):
        enhance_regions(image(), regions)
