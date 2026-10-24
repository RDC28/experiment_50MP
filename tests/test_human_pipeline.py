"""Small-image tests for the human upscaler boundary and honest provenance."""

import json

import numpy as np
import pytest

from synthetic_raw_lab import baselines, human_pipeline
from synthetic_raw_lab.pipeline import INTERPOLATION


@pytest.mark.parametrize("mode", ["conservative", "balanced"])
def test_stand_in_matches_baseline_and_records_provenance(mode):
    source = np.random.default_rng(42).uniform(-0.1, 1.1, (12, 17, 3)).astype(np.float32)
    original = source.copy()
    result = human_pipeline.get_upscaler()(source, 31, 25, mode)
    np.testing.assert_array_equal(result.linear, baselines.resample(source, 31, 25, "lanczos"))
    np.testing.assert_array_equal(source, original)
    assert result.linear.dtype == np.float32
    record = result.stage_record().to_dict()
    assert record["classification"] == INTERPOLATION
    assert record["notes"] == [human_pipeline.STAND_IN_DESCRIPTION]
    assert record["parameters"] == {"sr_backend": "lanczos_stand_in", "mode": mode, "width": 31, "height": 25}
    assert "Phase 2 SR not yet available" in json.dumps(record)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.float32),
        np.full((4, 4, 3), np.nan, dtype=np.float32),
        np.full((4, 4, 3), np.inf, dtype=np.float32),
    ],
)
def test_stand_in_rejects_invalid_images(source):
    with pytest.raises(ValueError, match="linear float32"):
        human_pipeline.get_upscaler()(source, 8, 8, "conservative")


@pytest.mark.parametrize("width,height", [(0, 8), (8, -1), (True, 8), (8, 2.5)])
def test_stand_in_rejects_invalid_dimensions(width, height):
    with pytest.raises(ValueError, match="positive integers"):
        human_pipeline.get_upscaler()(np.zeros((4, 4, 3), np.float32), width, height, "balanced")


def test_backend_and_mode_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match="unknown sr_backend"):
        human_pipeline.get_upscaler("missing")
    with pytest.raises(ValueError, match="mode must"):
        human_pipeline.get_upscaler()(np.zeros((4, 4, 3), np.float32), 8, 8, "aggressive")
    replacement = object()
    monkeypatch.setitem(human_pipeline.UPSCALERS, "phase2_adapter", replacement)
    assert human_pipeline.get_upscaler("phase2_adapter") is replacement


def test_general_sr_blend_endpoints_and_midpoint():
    from synthetic_raw_lab.human_pipeline import blend_upscale

    baseline = np.full((4, 5, 3), 0.2, np.float32)
    selected = np.full_like(baseline, 0.6)
    np.testing.assert_array_equal(blend_upscale(baseline, selected, 1), selected)
    np.testing.assert_array_equal(blend_upscale(baseline, selected, 0), baseline)
    np.testing.assert_allclose(blend_upscale(baseline, selected, 0.5), 0.4)


def test_edge_sharpening_neutral_bounded_and_face_excluded():
    from synthetic_raw_lab.human_pipeline import sharpen_edges

    image = np.random.default_rng(19).uniform(-0.1, 1.1, (8, 9, 3)).astype(np.float32)
    before = image.copy()
    faces = np.zeros(image.shape[:2], bool)
    faces[2:6, 3:7] = True
    np.testing.assert_array_equal(sharpen_edges(image, 0, faces), image)
    output = sharpen_edges(image, 1, faces)
    np.testing.assert_array_equal(output[faces], image[faces])
    assert np.max(np.abs(output - image)) <= 0.0200001
    assert np.any(output[~faces] != image[~faces])
    np.testing.assert_array_equal(image, before)


def test_tonal_contrast_pivot_direction_and_bounded_gain():
    from synthetic_raw_lab.human_pipeline import adjust_tonal_contrast

    levels = np.array([0, 1e-8, 0.045, 0.18, 0.72, 4], np.float32)
    source = np.repeat(levels[None, :, None], 3, axis=2)
    original = source.copy()
    increased = adjust_tonal_contrast(source, 0.5)
    decreased = adjust_tonal_contrast(source, -0.5)
    np.testing.assert_allclose(increased[0, 3], source[0, 3], atol=1e-7)
    assert increased[0, 2, 0] < source[0, 2, 0] < decreased[0, 2, 0]
    assert decreased[0, 4, 0] < source[0, 4, 0] < increased[0, 4, 0]
    assert np.all(increased >= source * 0.5)
    assert np.all(increased <= source * 2)
    np.testing.assert_array_equal(source, original)
    assert increased.dtype == np.float32 and np.isfinite(increased).all()


def test_tonal_contrast_preserves_chromatic_ratios_and_nonpositive_luminance():
    from synthetic_raw_lab.human_pipeline import adjust_tonal_contrast

    source = np.array([[[0.3, 0.2, 0.1], [-0.1, -0.1, -0.1]]], np.float32)
    result = adjust_tonal_contrast(source, 0.5)
    np.testing.assert_allclose(result[0, 0] / source[0, 0], np.repeat(result[0, 0, 0] / 0.3, 3), rtol=2e-7)
    np.testing.assert_array_equal(result[0, 1], source[0, 1])


def test_colour_balance_linear_channel_gains_without_clipping():
    from synthetic_raw_lab.human_pipeline import balance_colour

    source = np.array([[[0.8, 0.4, -0.1]]], np.float32)
    result = balance_colour(source, (1.5, 1, 0.5))
    np.testing.assert_allclose(result, [[[1.2, 0.4, -0.05]]])
    np.testing.assert_array_equal(source, np.array([[[0.8, 0.4, -0.1]]], np.float32))
    assert result.dtype == np.float32


def test_tone_and_colour_neutral_defaults_are_bit_identical():
    from synthetic_raw_lab.human_pipeline import adjust_tonal_contrast, balance_colour

    source = np.array([[[-0.0, -0.125, 1.125], [1e-8, 0.18, 0.5]]], np.float32)
    for result in (adjust_tonal_contrast(source, 0), balance_colour(source, (1, 1, 1))):
        assert result.tobytes() == source.tobytes()
        assert not np.shares_memory(result, source)


def test_grain_seed_bounds_luminance_and_face_suppression():
    from synthetic_raw_lab.human_pipeline import add_grain

    image = np.full((32, 32, 3), 0.5, np.float32)
    faces = np.zeros((32, 32), bool)
    faces[:4] = True
    original = image.copy()
    np.testing.assert_array_equal(add_grain(image, 0, faces, 42), image)
    output = add_grain(image, 1, faces, 42)
    np.testing.assert_array_equal(output, add_grain(image, 1, faces, 42))
    np.testing.assert_array_equal(output[faces], image[faces])
    assert not np.array_equal(output, add_grain(image, 1, faces, 43))
    assert np.max(np.abs(output - image)) <= 0.020001
    np.testing.assert_array_equal(output[..., 0], output[..., 1])
    dark = add_grain(image * 0.1, 1, faces, 42)
    assert np.std(dark - image * 0.1) < np.std(output - image)
    np.testing.assert_array_equal(image, original)
    assert output.dtype == np.float32 and np.isfinite(output).all()


@pytest.mark.parametrize("value", [-1, 1.1, True, float("nan"), "1"])
def test_grain_invalid_strength(value):
    from synthetic_raw_lab.human_config import from_dict
    from synthetic_raw_lab.human_pipeline import add_grain

    with pytest.raises(ValueError):
        from_dict({"grain": value})
    with pytest.raises(ValueError):
        add_grain(np.zeros((2, 2, 3), np.float32), value, np.zeros((2, 2), bool), 42)


def test_skin_smoothing_neutral_mask_weight_and_face_caps():
    from synthetic_raw_lab.human_pipeline import smooth_skin

    image = np.random.default_rng(9).uniform(0.3, 0.34, (16, 16, 3)).astype(np.float32)
    image[0, 0, 0] = -0.0
    before = image.copy()
    skin = np.ones((16, 16), np.float32)
    limits = np.ones_like(skin)
    limits[:4] = 0
    limits[4:8] = 0.1
    skin[:, :4] = 0
    assert smooth_skin(image, 0, skin, limits).tobytes() == image.tobytes()
    full = smooth_skin(image, 1, np.ones_like(skin), np.ones_like(skin))
    output = smooth_skin(image, 1, skin, limits)
    assert output[:4].tobytes() == image[:4].tobytes()
    assert output[:, :4].tobytes() == image[:, :4].tobytes()
    np.testing.assert_allclose(output[4:8, 4:] - image[4:8, 4:], 0.1 * (full[4:8, 4:] - image[4:8, 4:]), atol=3e-8)
    assert np.max(np.abs(output - image)) <= 0.0200001
    assert np.var(output[8:, 4:]) < np.var(image[8:, 4:])
    np.testing.assert_array_equal(image, before)
    assert output.dtype == np.float32 and np.isfinite(output).all()


def test_skin_smoothing_preserves_strong_edge():
    from synthetic_raw_lab.human_pipeline import smooth_skin

    image = np.full((12, 12, 3), 0.1, np.float32)
    image[:, 6:] = 0.9
    weights = np.ones((12, 12), np.float32)
    np.testing.assert_allclose(smooth_skin(image, 1, weights, weights), image, atol=1e-6)


@pytest.mark.parametrize("value", [-1, 1.1, True, float("nan"), "1"])
def test_skin_smoothing_invalid_strength(value):
    from synthetic_raw_lab.human_config import from_dict

    with pytest.raises(ValueError):
        from_dict({"skin_texture_smoothing": value})
    with pytest.raises(ValueError):
        human_pipeline.smooth_skin(
            np.zeros((2, 2, 3), np.float32), value, np.ones((2, 2), np.float32), np.ones((2, 2), np.float32)
        )
