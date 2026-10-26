"""Tiny standard-primary vectors and audited, disk-backed colour conversion."""

from types import SimpleNamespace

import numpy as np
import pytest

from synthraw.colour_space import convert_linear_primaries, resolve_colour_space
from synthraw.hdr import HDRReconstructionEngine, inverse_smoothstep, inverse_transfer
from synthraw.provenance import ProvenanceWriter


def audit(description, present=True):
    return SimpleNamespace(icc_description=description, icc_present=present)


@pytest.mark.parametrize(
    ("description", "transfer"),
    [
        ("sRGB IEC61966-2.1", "SRGB"),
        ("Display P3", "DISPLAY_P3"),
        ("Adobe RGB (1998)", "ADOBE_RGB"),
        ("Rec.709", "REC709"),
        ("BT.709", "REC709"),
    ],
)
def test_profile_resolution(description, transfer):
    result = resolve_colour_space(audit(description))
    assert result.transfer == transfer
    assert result.assumption is None


@pytest.mark.parametrize(
    "description,present", [(None, False), (None, True), ("Custom LUT", True), ("Display P3", False)]
)
def test_unknown_profile_records_assumption(tmp_path, description, present):
    result = resolve_colour_space(audit(description, present))
    assert result.transfer == "SRGB" and result.assumption
    ProvenanceWriter().write(result.records(), tmp_path / "colour.json")
    assert result.records()[0].to_dict()["provenance"] == "I"
    assert result.records()[1].to_dict()["provenance"] == "D"
    assert result.assumption in (tmp_path / "colour.json").read_text()


@pytest.mark.parametrize(
    ("description", "expected"),
    [("Display P3", [1.224745, -0.042058, -0.019642]), ("Adobe RGB (1998)", [1.398283, 0, 0])],
)
def test_wide_gamut_red_reference_and_no_clipping(description, expected):
    result = convert_linear_primaries(np.array([[1, 0, 0]]), resolve_colour_space(audit(description)))
    np.testing.assert_allclose(result[0], expected, atol=2e-6)
    assert result.dtype == np.float32


@pytest.mark.parametrize("description", ["sRGB", "Rec.709", "Display P3", "Adobe RGB"])
def test_neutral_white_and_black(description):
    result = convert_linear_primaries(np.array([[0, 0, 0], [1, 1, 1]]), resolve_colour_space(audit(description)))
    np.testing.assert_allclose(result, [[0, 0, 0], [1, 1, 1]], atol=3e-4)


def test_strip_linearization_resolves_transfer_then_tone_then_primaries(tmp_path):
    source = np.array([[[180, 80, 40], [128, 128, 128]]] * 3, dtype=np.uint8)
    resolution = resolve_colour_space(audit("Display P3"))
    result = HDRReconstructionEngine().linearize(source, tmp_path / "linear.npy", colour_space=resolution, strip_rows=1)
    expected = convert_linear_primaries(inverse_smoothstep(inverse_transfer(source / 255, "DISPLAY_P3")), resolution)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(np.load(tmp_path / "linear.npy"), result)


@pytest.mark.parametrize("values", [np.array([1, 2]), np.array([[np.nan, 0, 0]])])
def test_invalid_rgb(values):
    with pytest.raises(ValueError, match="finite linear RGB"):
        convert_linear_primaries(values, resolve_colour_space(audit("sRGB")))
