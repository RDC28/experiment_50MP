"""Tiny numerical and disk-backed inverse-rendering tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from synthraw.colour_space import resolve_colour_space
from synthraw.hdr import HDRReconstructionEngine, inverse_smoothstep, inverse_transfer


@pytest.mark.parametrize("transfer", ["SRGB", "DISPLAY_P3", "ADOBE_RGB", "REC709"])
def test_transfer_endpoints_monotonic_and_dtype(transfer):
    decoded = inverse_transfer(np.linspace(0, 1, 257), transfer)
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded[[0, -1]], [0, 1], atol=1e-7)
    assert np.all(np.diff(decoded) > 0)


@pytest.mark.parametrize(
    ("transfer", "encoded", "expected"),
    [
        ("SRGB", 0.04045, 0.0031308),
        ("DISPLAY_P3", 0.5, 0.21404114),
        ("ADOBE_RGB", 0.5, 0.2177555),
        ("REC709", 0.045, 0.01),
        ("REC709", 0.5, 0.2595894),
    ],
)
def test_reference_transfer_values(transfer, encoded, expected):
    np.testing.assert_allclose(inverse_transfer(np.array([encoded]), transfer), [expected], atol=1e-7)


def test_inverse_tone_roundtrip():
    original = np.linspace(0, 1, 101, dtype=np.float32)
    rendered = 3 * original**2 - 2 * original**3
    np.testing.assert_allclose(inverse_smoothstep(rendered), original, atol=2e-6)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.01, 1.01])
@pytest.mark.parametrize("operation", [inverse_transfer, inverse_smoothstep])
def test_invalid_radiance_rejected(invalid, operation):
    with pytest.raises(ValueError, match="finite and normalized"):
        operation(np.array([invalid]))


def test_strip_pipeline_matches_reference(tmp_path):
    source = np.arange(7 * 5 * 3, dtype=np.uint8).reshape(7, 5, 3)
    before = source.copy()
    output = HDRReconstructionEngine().linearize(source, tmp_path / "linear.npy", strip_rows=2)
    assert isinstance(output, np.memmap)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(source, before)
    np.testing.assert_allclose(output, inverse_smoothstep(inverse_transfer(source / 255)))
    np.testing.assert_array_equal(np.load(tmp_path / "linear.npy"), output)


def test_memmap_input_and_optional_tone(tmp_path):
    path = tmp_path / "input.npy"
    np.save(path, np.full((3, 2, 3), 0.5, dtype=np.float32))
    source = np.load(path, mmap_mode="r")
    engine = HDRReconstructionEngine()
    with pytest.raises(ValueError, match="overwrite"):
        engine.linearize(source[:], path)
    result = engine.linearize(source, tmp_path / "output.npy", undo_tone=False, transfer="ADOBE_RGB")
    np.testing.assert_allclose(result, 0.2177555, atol=1e-7)


def test_failed_stage_cleanup_and_existing_output_protection(tmp_path):
    path = tmp_path / "out.npy"
    engine = HDRReconstructionEngine()
    with pytest.raises(ValueError):
        engine.linearize(np.full((2, 2, 3), np.nan), path)
    assert not path.exists()
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        engine.linearize(np.zeros((2, 2, 3)), path)
    assert path.read_bytes() == b"existing"


@pytest.mark.parametrize("rows", [0, -1, True, 1.5])
def test_invalid_strip_size(tmp_path, rows):
    with pytest.raises(ValueError, match="strip_rows"):
        HDRReconstructionEngine().linearize(np.zeros((2, 2, 3)), tmp_path / "out.npy", strip_rows=rows)


def test_hdr_preserves_supported_and_bounds_extrapolation(tmp_path):
    encoded = np.array([[[128] * 3, [253] * 3, [2] * 3, [255] * 3, [0] * 3]], dtype=np.uint8)
    linear = inverse_transfer(encoded / 255)
    result = HDRReconstructionEngine().expand(encoded, linear, tmp_path, strip_rows=1)
    arrays = result["arrays"]
    hdr = arrays["04_hdr_radiance"]
    np.testing.assert_array_equal(hdr[:, :1], linear[:, :1])
    assert np.all(hdr[:, 1] > linear[:, 1])
    assert np.all(hdr[:, 2] < linear[:, 2])
    np.testing.assert_allclose(hdr[:, 3], 8)
    np.testing.assert_array_equal(hdr[:, 4], 0)
    np.testing.assert_array_equal(arrays["source_supported_mask"], [[True, False, False, False, False]])
    np.testing.assert_array_equal(arrays["highlight_fabricated_mask"], [[False, True, False, True, False]])
    np.testing.assert_array_equal(arrays["shadow_fabricated_mask"], [[False, False, True, False, True]])
    np.testing.assert_array_equal(arrays["hdr_confidence"][:, [0, 3, 4]], [[1, 0, 0]])
    assert result["generated_hdr_area_percentage"] == 80
    assert all(isinstance(array, np.memmap) for array in arrays.values())
    assert np.isfinite(hdr).all() and np.all(hdr >= 0)


@pytest.mark.parametrize("backend", ["DETERMINISTIC", "EXPANDDIFF", "VID2HDR"])
def test_hdr_strip_equivalence_and_fallback(tmp_path, backend):
    encoded = np.linspace(0, 1, 81, dtype=np.float32).reshape(9, 3, 3)
    linear = inverse_transfer(encoded)
    engine = HDRReconstructionEngine()
    first = engine.expand(encoded, linear, tmp_path / "a", strip_rows=1, backend=backend)
    second = engine.expand(encoded, linear, tmp_path / "b", strip_rows=9)
    for name, array in first["arrays"].items():
        np.testing.assert_array_equal(array, second["arrays"][name])
        np.testing.assert_array_equal(np.load(tmp_path / "a" / f"{name}.npy"), array)
    assert bool(first["fallback_reason"]) == (backend != "DETERMINISTIC")


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1])
def test_hdr_invalid_linear_cleans_outputs(tmp_path, invalid):
    with pytest.raises(ValueError, match="radiance"):
        HDRReconstructionEngine().expand(np.ones((2, 2, 3)), np.full((2, 2, 3), invalid), tmp_path)
    assert not list(tmp_path.glob("*.npy"))


def test_hdr_existing_stage_is_protected(tmp_path):
    path = tmp_path / "highlight_fabricated_mask.npy"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        HDRReconstructionEngine().expand(np.ones((2, 2, 3)), np.ones((2, 2, 3)), tmp_path)
    assert path.read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("ev", [-1, np.nan, np.inf, 17])
def test_hdr_invalid_ev(tmp_path, ev):
    with pytest.raises(ValueError, match="EV"):
        HDRReconstructionEngine().expand(np.ones((2, 2, 3)), np.ones((2, 2, 3)), tmp_path, highlight_ev=ev)


@pytest.mark.parametrize("strip_rows", [1, 3])
def test_wide_gamut_linearize_to_hdr_with_explicit_mapping(tmp_path, strip_rows):
    encoded = np.array([[[255, 0, 0], [128, 128, 128]]] * 3, dtype=np.uint8)
    space = resolve_colour_space(SimpleNamespace(icc_present=True, icc_description="Display P3"))
    engine = HDRReconstructionEngine()
    linear = engine.linearize(encoded, tmp_path / "linear.npy", colour_space=space, strip_rows=strip_rows)
    before = linear.copy()
    assert np.any(linear < 0)
    with pytest.raises(ValueError, match="nonnegative"):
        engine.expand(encoded, linear, tmp_path / "strict")
    result = engine.expand(encoded, linear, tmp_path / "mapped", negative_policy="CLIP", strip_rows=strip_rows)
    hdr = result["arrays"]["04_hdr_radiance"]
    assert np.isfinite(hdr).all() and np.all(hdr >= 0)
    assert np.all(hdr[:, 0, 0] > 1)  # Preserve positive wide-gamut headroom.
    np.testing.assert_array_equal(hdr[:, 0, 1:], 0)
    np.testing.assert_array_equal(hdr[:, 1], linear[:, 1])
    np.testing.assert_array_equal(linear, before)
    np.testing.assert_array_equal(np.load(tmp_path / "linear.npy"), before)
    audit = result["gamut_mapping"]
    assert audit["policy"] == "CLIP" and audit["provenance"] == "I"
    assert audit["negative_channel_count"] == 6
    assert audit["affected_pixel_count"] == 3
    assert audit["affected_pixel_percentage"] == 50
    assert audit["minimum_input_channel"] == float(before.min())


def test_gamut_mapping_excludes_unsupported_colour_without_fabricating_hdr_masks(tmp_path):
    encoded = np.full((1, 2, 3), 0.5, dtype=np.float32)
    linear = np.array([[[-0.1, 0.2, 1.5], [0.2, 0.3, 0.4]]], dtype=np.float32)
    result = HDRReconstructionEngine().expand(encoded, linear, tmp_path, negative_policy="CLIP")
    arrays = result["arrays"]
    np.testing.assert_array_equal(arrays["source_supported_mask"], [[False, True]])
    np.testing.assert_array_equal(arrays["hdr_confidence"], [[0, 1]])
    np.testing.assert_array_equal(arrays["highlight_fabricated_mask"], False)
    np.testing.assert_array_equal(arrays["shadow_fabricated_mask"], False)
    np.testing.assert_allclose(arrays["04_hdr_radiance"], [[[0, 0.2, 1.5], [0.2, 0.3, 0.4]]])
    assert result["generated_hdr_area_percentage"] == 0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_gamut_mapping_still_rejects_nonfinite_radiance(tmp_path, invalid):
    with pytest.raises(ValueError, match="finite"):
        HDRReconstructionEngine().expand(
            np.full((1, 1, 3), 0.5), np.full((1, 1, 3), invalid), tmp_path, negative_policy="CLIP"
        )
    assert not list(tmp_path.glob("*.npy"))


def test_invalid_gamut_policy_creates_no_outputs(tmp_path):
    with pytest.raises(ValueError, match="negative_policy"):
        HDRReconstructionEngine().expand(
            np.ones((1, 1, 3)), np.ones((1, 1, 3)), tmp_path / "hdr", negative_policy="IGNORE"
        )
    assert not (tmp_path / "hdr").exists()
