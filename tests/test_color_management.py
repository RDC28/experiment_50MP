import numpy as np
import pytest

from helpers import build_icc_profile, d50_colorants
from synthetic_raw_lab import color_management as cm


def test_srgb_transfer_known_values_and_roundtrip():
    assert cm.srgb_to_linear(np.float32(0.5)) == pytest.approx(0.214041, abs=1e-6)
    assert cm.linear_to_srgb(np.float32(0.18)) == pytest.approx(0.461356, abs=1e-6)
    assert cm.srgb_to_linear(np.float32(0.02)) == pytest.approx(0.02 / 12.92, rel=1e-6)
    x = np.linspace(0, 1, 10001, dtype=np.float32)
    assert np.max(np.abs(cm.linear_to_srgb(cm.srgb_to_linear(x)) - x)) < 2e-6


def test_encode_uint16_clips_and_counts():
    linear = np.array([[[-0.1, 0.0, 0.5], [1.0, 1.2, np.nan]]], dtype=np.float32)
    encoded, stats = cm.encode_srgb_uint16(linear, chunk_rows=1)
    assert encoded.dtype == np.uint16
    assert encoded.ravel().tolist() == [0, 0, round(float(cm.linear_to_srgb(np.float32(0.5))) * 65535), 65535, 65535, 0]
    assert (stats.below_zero, stats.above_one, stats.nonfinite) == (1, 1, 1)
    assert stats.to_dict()["above_one_fraction"] == pytest.approx(1 / 6)


def test_uint16_lut_inverts_encoding():
    linear = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    encoded, _ = cm.encode_srgb_uint16(linear)
    assert np.max(np.abs(cm.srgb_uint16_to_linear(encoded) - linear)) < 5e-5


def test_uint16_to_uint8_rounds():
    codes = np.array([0, 257, 65535, 128], dtype=np.uint16)
    assert cm.srgb_uint16_to_uint8(codes).tolist() == [0, 1, 255, 0]


def test_srgb_primaries_map_to_identity():
    assert np.allclose(cm.matrix_to_working(cm.SRGB_PRIMARIES, cm.D65_XY), np.eye(3), atol=1e-12)


def test_wide_gamut_red_falls_outside_srgb():
    p3 = cm.named_color_space("display_p3", "user_override")
    red = cm.to_working_space(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32), p3)[0, 0]
    assert red[0] > 1 and red[1] < 0 and red[2] < 0
    white = cm.to_working_space(np.ones((1, 1, 3), dtype=np.float32), p3)[0, 0]
    assert np.allclose(white, 1, atol=1e-5)


def test_littlecms_srgb_profile_is_recognised_as_srgb():
    space = cm.color_space_from_icc(cm.srgb_icc_profile())
    assert space.name == "srgb"
    assert space.origin == "icc_profile"
    assert space.matrix is None


def test_table_curve_srgb_profile_is_recognised_as_srgb():
    table = cm.srgb_to_linear(np.linspace(0, 1, 1024, dtype=np.float32))
    space = cm.color_space_from_icc(build_icc_profile(curve=("table", table)))
    assert space.name == "srgb"


def test_display_p3_icc_matches_named_display_p3():
    params = (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)
    profile = build_icc_profile(d50_colorants(cm.DISPLAY_P3_PRIMARIES), ("para", 3, params), description="P3 test")
    space = cm.color_space_from_icc(profile)
    assert space.name == "icc:P3 test"
    samples = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
    from_icc = cm.to_working_space(samples, space)
    named = cm.to_working_space(samples, cm.named_color_space("display_p3", "user_override"))
    assert np.max(np.abs(from_icc - named)) < 2e-3


def test_gamma_profile_applies_its_curve():
    space = cm.color_space_from_icc(build_icc_profile(curve=("gamma", 2.2)))
    assert space.transfer == "ICC gamma 2.19922/gamma 2.19922/gamma 2.19922"
    value = cm.to_working_space(np.full((1, 1, 3), 0.5, dtype=np.float32), space)[0, 0]
    assert np.allclose(value, 0.5 ** (563 / 256), atol=1e-3)


@pytest.mark.parametrize("function", [0, 1, 2, 3, 4])
def test_parametric_curve_types(function):
    params = {
        0: (2.2,),
        1: (2.2, 1.0, 0.0),
        2: (2.2, 1.0, 0.0, 0.0),
        3: (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045),
        4: (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045, 0.0, 0.0),
    }[function]
    space = cm.color_space_from_icc(build_icc_profile(curve=("para", function, params)))
    x = np.linspace(0, 1, 11, dtype=np.float32)[None, :, None].repeat(3, axis=2)
    out = cm.to_working_space(x, space)[0, :, 0]
    expected = cm.srgb_to_linear(x[0, :, 0]) if function >= 3 else x[0, :, 0] ** 2.2
    assert np.allclose(out, expected, atol=2e-3)


def test_grey_profile_is_replicated():
    profile = build_icc_profile(curve=("gamma", 1.0), color_space=b"GRAY")
    space = cm.color_space_from_icc(profile)
    out = cm.to_working_space(np.full((2, 2, 1), 0.25, dtype=np.float32), space)
    assert out.shape == (2, 2, 3)
    assert np.allclose(out, 0.25, atol=1e-3)


def test_unsupported_profiles_raise():
    with pytest.raises(cm.UnsupportedColorProfile):
        cm.color_space_from_icc(b"not a profile")
    lut_only = bytearray(build_icc_profile())
    lut_only[16:20] = b"CMYK"
    with pytest.raises(cm.UnsupportedColorProfile):
        cm.color_space_from_icc(bytes(lut_only))


def test_resolution_order():
    assert cm.resolve_source_color_space("auto", None).origin == "untagged_default"
    assert cm.resolve_source_color_space("auto", None).name == "srgb"
    assert cm.resolve_source_color_space("auto", None, is_linear_container=True).name == "linear_srgb"
    override = cm.resolve_source_color_space("adobe_rgb", cm.srgb_icc_profile())
    assert (override.name, override.origin) == ("adobe_rgb", "user_override")
    with pytest.raises(ValueError):
        cm.resolve_source_color_space("prophoto", None)


def test_signal_labels_never_claim_scene_referred():
    for name in ("srgb", "linear_srgb", "display_p3", "adobe_rgb"):
        signal = cm.named_color_space(name, "user_override").signal
        assert "scene-referred" not in signal or "not scene-referred" in signal
    assert "not scene-referred" in cm.SIGNAL_LINEAR_DISPLAY


def test_exr_chromaticities():
    rec709 = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
    assert cm.linear_color_space_from_chromaticities(rec709).name == "linear_srgb"
    p3 = (0.68, 0.32, 0.265, 0.69, 0.15, 0.06, 0.3127, 0.3290)
    space = cm.linear_color_space_from_chromaticities(p3)
    assert space.name == "linear_custom"
    assert np.allclose(space.matrix, cm.matrix_to_working(cm.DISPLAY_P3_PRIMARIES, cm.D65_XY))


def test_icc_description_parses_desc_tag():
    assert cm.icc_description(build_icc_profile(description="Hello")) == "Hello"
    assert cm.icc_description(cm.srgb_icc_profile())
    assert cm.icc_description(None) is None
