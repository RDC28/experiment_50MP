from __future__ import annotations

import numpy as np
import pytest

from prd_v6 import colour, constants, pipeline


def test_srgb_transfer_round_trips():
    values = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.allclose(colour.srgb_encode(colour.srgb_decode(values)), values, atol=1e-6)


def test_camera_transform_inverts_exactly():
    rng = np.random.default_rng(0)
    linear = rng.uniform(0, 1, (8, 8, 3)).astype(np.float32)
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    camera = colour.linear_srgb_to_camera(linear, neutral, pipeline.camera_matrix(), constants.SIGNAL_16)
    back = colour.camera_to_linear_srgb(camera, neutral, pipeline.camera_matrix(), constants.SIGNAL_16)
    assert np.allclose(back, linear, atol=1e-5)


def test_render_white_maps_to_clip_in_the_largest_channel():
    """LibRaw's convention: render white is the neutral scaled so its largest channel reaches clip."""
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    white = pipeline.to_camera(np.ones((1, 1, 3), dtype=np.float32), neutral)[0, 0]
    assert np.allclose(white, neutral / neutral.max() * constants.SIGNAL_16, rtol=1e-4)


def test_prior_sits_between_tungsten_and_daylight():
    cam = pipeline.camera_matrix()
    prior = colour.prior_neutral(cam)
    daylight = colour.daylight_neutral(cam)
    assert prior[1] == pytest.approx(1.0)
    assert prior == pytest.approx([0.503, 1.0, 0.550], abs=0.01)
    assert daylight == pytest.approx([0.3744, 1.0, 0.7401], abs=0.01)


def test_neutral_from_libraw_multipliers():
    assert colour.neutral_from_multipliers([2.0, 1.0, 1.8, 1.0]) == pytest.approx([0.5, 1.0, 1 / 1.8])


def test_uint16_encoding_uses_the_full_range():
    encoded = np.array([0.0, 0.5, 1.0, 1.2], dtype=np.float32)
    assert colour.to_uint16(encoded).tolist() == [0, 32768, 65535, 65535]


@pytest.mark.parametrize("value", [None, "", "  ", "estimated", "Estimated"])
def test_no_declaration_parses_to_none(value):
    assert colour.parse_declared_cct(value) is None


@pytest.mark.parametrize(
    "value, kelvin", [(4300, 4300.0), (4300.0, 4300.0), ("4300", 4300.0), (" 4300 K ", 4300.0), ("5600k", 5600.0)]
)
def test_a_declared_cct_parses_to_kelvin(value, kelvin):
    assert colour.parse_declared_cct(value) == kelvin


@pytest.mark.parametrize("value", ["warm", "4300 mired", True, float("nan"), float("inf"), 1000, 30000, "-4300"])
def test_a_bad_declaration_is_rejected(value):
    with pytest.raises(ValueError, match="ILLUMINANT_CCT_K"):
        colour.parse_declared_cct(value)


def test_default_illuminant_is_exactly_the_estimated_prior():
    cam = pipeline.camera_matrix()
    for none in (None, "", "estimated"):
        illuminant = colour.resolve_illuminant(cam, none)
        assert illuminant["source"] == "estimated"
        assert illuminant["cct_k"] == constants.PRIOR_CCT_K
        assert illuminant["neutral"] == colour.prior_neutral(cam).tolist()
        assert "estimated prior" in illuminant["label"] and "declared" not in illuminant["label"]


def test_a_declared_illuminant_goes_through_the_planckian_path_and_is_labelled_declared():
    cam = pipeline.camera_matrix()
    illuminant = colour.resolve_illuminant(cam, "4300 K")
    assert illuminant["source"] == "declared" and illuminant["cct_k"] == 4300.0
    assert illuminant["neutral"] == colour.neutral_for_cct(4300.0, cam).tolist()
    assert illuminant["neutral"] != colour.prior_neutral(cam).tolist()
    assert illuminant["label"] == "user-declared, 4300 K"
    # Declaring the prior's own CCT reproduces the prior neutral, but is still labelled declared.
    same = colour.resolve_illuminant(cam, constants.PRIOR_CCT_K)
    assert same["neutral"] == pytest.approx(colour.prior_neutral(cam).tolist()) and same["source"] == "declared"


def test_a_warmer_declaration_raises_red_and_lowers_blue():
    cam = pipeline.camera_matrix()
    warm, cool = (colour.resolve_illuminant(cam, k)["neutral"] for k in (3200, 6500))
    assert warm[0] > cool[0] and warm[2] < cool[2] and warm[1] == cool[1] == 1.0
