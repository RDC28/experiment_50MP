from __future__ import annotations

import numpy as np
import pytest

from prd_v6 import constants, metrics


def _variance(x, plane):
    return metrics.noise_variance(x, plane=plane)


def _scene(hw=(64, 64)):
    rng = np.random.default_rng(1)
    x = rng.uniform(2000, 40000, hw + (3,))
    return x, np.ones(hw, dtype=bool)


def test_self_score_is_perfect():
    x, mask = _scene()
    result = metrics.score(x, x, None, mask, variance=_variance, candidate_noisy=False)
    assert result["k"] == pytest.approx(1.0)
    assert result["min_A"] == pytest.approx(1.0)
    assert result["min_P2"] == 1.0
    assert result["passes"]


def test_global_exposure_is_fitted_once():
    x, mask = _scene()
    result = metrics.score(x, x / 2, None, mask, variance=_variance, candidate_noisy=False)
    assert result["k"] == pytest.approx(2.0)
    assert result["min_A"] == pytest.approx(1.0)
    assert result["planes"]["G"]["A_unfitted"] == pytest.approx(0.5)


def test_a_is_one_minus_relative_rms_error():
    x, mask = _scene()
    y = x.copy()
    y[..., 0] *= 1.1
    result = metrics.score(x, y, None, mask, variance=_variance, candidate_noisy=False, k=1.0)
    assert result["planes"]["R"]["A"] == pytest.approx(0.9)
    assert result["planes"]["G"]["A"] == pytest.approx(1.0)
    assert not result["passes"]
    assert result["worst_plane"] == "R"


def test_p2_tolerance_is_five_percent_for_bright_values():
    x = np.full(1000, 50000.0)
    assert metrics.p2(x, x * 1.04, 1.0, np.zeros_like(x), 0.0) == 1.0
    assert metrics.p2(x, x * 1.06, 1.0, np.zeros_like(x), 0.0) == 0.0


def test_p2_tolerance_is_three_sigma_for_dark_values():
    x = np.full(1000, 100.0)
    var = metrics.noise_variance(x)
    sigma = float(np.sqrt(var[0]))
    assert metrics.p2(x, x + 2.9 * sigma, 1.0, var, 0.0) == 1.0
    assert metrics.p2(x, x + 3.1 * sigma, 1.0, var, 0.0) == 0.0


def test_cfa_planes_pool_both_greens():
    layout = np.tile(np.array([[0, 1], [1, 2]], dtype=np.int8), (4, 4))
    values = layout.astype(np.float64) + 1
    planes = metrics.split_planes(values, layout, np.ones_like(layout, dtype=bool))
    assert [p.size for p in planes] == [16, 32, 16]


def test_noise_model_and_binning():
    expected = 16 * ((4 / constants.GAIN_E_PER_DN14) ** 2 + 1 / 12)
    assert metrics.noise_variance(np.array([0.0]))[0] == pytest.approx(expected)
    assert metrics.binned_counts(4.5) == pytest.approx((5.0625, 10.125, 5.0625))
    assert metrics.binned_counts(1.0) == (1.0, 1.0, 1.0)
