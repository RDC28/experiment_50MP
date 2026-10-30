"""Known sampled edges verify response without a model or full-sized raster."""

import json

import numpy as np
import pytest
from scipy.special import erf

from prd_v5.mtf import edge_mtf

EVIDENCE = {"edge": "synthetic isolated straight edge", "linearization": "analytic linear intensity"}


@pytest.mark.parametrize("sigma", [1.5, 3, 6])
def test_gaussian_edge_has_expected_bandwidth(sigma):
    spacing = 0.125
    x = np.arange(-64, 64, spacing)
    esf = 0.5 * (1 + erf(x / (np.sqrt(2) * sigma)))
    result = edge_mtf(esf, sample_spacing_px=spacing, evidence=EVIDENCE)
    for level, name in [(0.5, "mtf50"), (0.1, "mtf10")]:
        expected = np.sqrt(-2 * np.log(level)) / (2 * np.pi * sigma)
        assert result[name]["frequency"] == pytest.approx(expected, rel=0.002)
    assert max(result["frequencies"]) <= 0.5
    assert result["response"][0] == 1
    json.dumps(result, allow_nan=False)


def test_sharp_edge_has_no_invented_crossing():
    result = edge_mtf(np.r_[np.zeros(32), np.ones(32)], evidence=EVIDENCE)
    np.testing.assert_allclose(result["response"], 1)
    assert result["mtf50"]["frequency"] is None
    assert "Nyquist" in result["mtf10"]["reason"]


def test_polarity_gain_offset_and_input_immutability():
    profile = erf(np.linspace(-8, 8, 128))
    original = profile.copy()
    first = edge_mtf(profile, evidence=EVIDENCE)
    second = edge_mtf(10 - 3 * profile, evidence=EVIDENCE)
    np.testing.assert_allclose(first["response"], second["response"], atol=1e-14)
    np.testing.assert_array_equal(profile, original)


@pytest.mark.parametrize("profile,evidence", [(None, None), (np.zeros(32), None), (np.ones(32), EVIDENCE)])
def test_missing_measurement_stays_not_measured(profile, evidence):
    result = edge_mtf(profile, evidence=evidence)
    assert result["status"] == "not measured"
    assert "mtf50" not in result


@pytest.mark.parametrize(
    "profile",
    [
        np.zeros((16, 2)),
        np.zeros(15),
        np.zeros(65537),
        [np.nan] * 32,
        [np.inf] * 32,
        np.ones(32, complex),
        np.ones(32, bool),
    ],
)
def test_invalid_profiles(profile):
    with pytest.raises(ValueError, match="profile"):
        edge_mtf(profile, evidence=EVIDENCE)


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, True, 1j, "1", None])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        edge_mtf(np.arange(32), sample_spacing_px=spacing, evidence=EVIDENCE)


def test_evidence_required():
    with pytest.raises(ValueError, match="evidence"):
        edge_mtf(np.arange(32), evidence={"edge": "unknown"})
