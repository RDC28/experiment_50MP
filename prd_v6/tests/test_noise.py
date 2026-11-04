"""50 MP noise: the camera model is topped up, not added on top of noise the render already carries (5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from prd_v6 import colour, constants, pipeline
from synthraw.realism import photon_transfer_fit

HW = (1024, 1536)  # 32-photosite blocks per CFA plane, 24 grey levels with 16 blocks each


def _layout(hw):
    return np.tile(np.array([[0, 1], [1, 2]], dtype=np.int8), (hw[0] // 2, hw[1] // 2))


def _grey_steps(hw=HW, *, noise=0.0, seed=0) -> np.ndarray:
    """Linear grey in 64-pixel columns from 0.3% to 50% of white, plus render-like noise (variance ~ level)."""
    levels = np.repeat(np.geomspace(0.003, 0.5, hw[1] // 64), 64).astype(np.float32)
    linear = np.broadcast_to(levels[None, :, None], hw + (3,)).copy()
    if noise:
        linear += np.random.default_rng(seed).normal(size=linear.shape).astype(np.float32) * np.sqrt(noise * linear)
    return np.clip(linear, 0.0, 1.0)


def _green_slope(samples: np.ndarray) -> float:
    signal = pipeline.stored_to_signal(samples)
    slopes = [photon_transfer_fit(signal[r::2, c::2], black=0, white=constants.SIGNAL_16) for r, c in ((0, 1), (1, 0))]
    return float(np.mean([fit["dn2_per_dn"] for fit in slopes]))


def test_camera_noise_model_on_the_16_bit_scale():
    slope, read_variance = pipeline.camera_noise_dn16()
    assert slope == pytest.approx(constants.SCALE_16 / constants.GAIN_E_PER_DN14)  # 1.923 DN16 per DN16
    assert read_variance == pytest.approx((constants.READ_NOISE_E * slope) ** 2)  # 59.2 DN16^2


@pytest.mark.parametrize(
    ("existing", "added"),
    [(None, (1.923, 59.17)), ((1.0, 30.0), (0.923, 29.17)), ((-0.5, -10.0), (1.923, 59.17))],
)
def test_add_noise_adds_the_remainder_up_to_the_camera_model(existing, added):
    level = 20_000.0
    flat = np.full(400_000, level, dtype=np.float32)
    dark = np.zeros(400_000, dtype=np.float32)
    noisy_flat = pipeline.add_noise(flat, 1, existing=existing)
    noisy_dark = pipeline.add_noise(dark, 2, existing=existing)
    assert noisy_flat.mean() == pytest.approx(level, rel=1e-3)
    assert noisy_dark.var() == pytest.approx(added[1], rel=0.03)
    assert noisy_flat.var() - noisy_dark.var() == pytest.approx(added[0] * level, rel=0.03)


def test_add_noise_adds_nothing_when_the_signal_already_carries_the_camera_noise():
    signal = np.linspace(0, constants.SIGNAL_16, 1000, dtype=np.float32)
    assert np.array_equal(pipeline.add_noise(signal, 3, existing=(2.5, 100.0)), signal)


def test_existing_noise_measures_each_cfa_position():
    rng = np.random.default_rng(4)
    signal = np.repeat(np.geomspace(50, 30_000, HW[1] // 64), 64)[None, :] * np.ones((HW[0], 1))
    signal = signal + rng.normal(size=signal.shape) * np.sqrt(0.8 * signal + 40)
    measured = pipeline.existing_noise(signal.astype(np.float32))
    assert set(measured) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for slope, read_variance in measured.values():
        assert slope == pytest.approx(0.8, rel=0.25)
        assert read_variance == pytest.approx(40, abs=40)
    assert pipeline.existing_noise(np.zeros((8, 8), np.float32))[(0, 0)] == (0.0, 0.0)  # no fit: noise-free


def test_cfa_samples_top_up_matches_a_clean_scene_with_camera_noise():
    """Regression (run 20261104T064058Z): the 50 MP G gain read 0.64x the camera's.

    The SR mosaic already carried about 59% of the camera's shot variance, and the full model went
    on top. Measured the way the noise gate measures, a render that carries noise must now come out
    like a clean scene with the camera model, within the gate's 15%.
    """
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    layout = _layout(HW)
    _, model = pipeline.cfa_samples(_grey_steps(), neutral, layout, noise_seed=6)
    noisy_render = _grey_steps(noise=2e-5, seed=1)
    clean, topped_up = pipeline.cfa_samples(noisy_render, neutral, layout, noise_seed=6)

    existing = _green_slope(clean)
    assert 0.3 * _green_slope(model) < existing < 0.9 * _green_slope(model)  # the render carries real noise
    assert _green_slope(topped_up) / _green_slope(model) == pytest.approx(1.0, abs=constants.NOISE_GATE_TOLERANCE)
    double_counted = pipeline.quantize(pipeline.add_noise(pipeline.stored_to_signal(clean), 6))
    assert _green_slope(double_counted) / _green_slope(model) > 1 + constants.NOISE_GATE_TOLERANCE


def test_cfa_samples_noise_is_reproducible_and_seed_dependent():
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    linear, layout = _grey_steps((64, 128)), _layout((64, 128))
    first = pipeline.cfa_samples(linear, neutral, layout, noise_seed=6)[1]
    assert np.array_equal(first, pipeline.cfa_samples(linear, neutral, layout, noise_seed=6)[1])
    assert not np.array_equal(first, pipeline.cfa_samples(linear, neutral, layout, noise_seed=7)[1])


def test_cfa_samples_accepts_a_seed_sequence_as_video_v1_passes_it():
    """video_v1 gives every frame its own SeedSequence; an int seed and the same sequence must agree."""
    linear = np.full((8, 8, 3), 0.3, dtype=np.float32)
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    layout = np.tile(np.array([[0, 1], [1, 2]], dtype=np.int8), (4, 4))
    _, from_int = pipeline.cfa_samples(linear, neutral, layout, noise_seed=7)
    _, from_sequence = pipeline.cfa_samples(linear, neutral, layout, noise_seed=np.random.SeedSequence(7))
    assert np.array_equal(from_int, from_sequence)
