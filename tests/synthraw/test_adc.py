"""Tiny ADC contract and applied temporal noise checks for PRD_v3."""

import numpy as np
import pytest

from synthraw.adc import ADCSimulator
from synthraw.cfa import CFAMosaicer
from synthraw.config import ExperimentConfig
from synthraw.sensor import SensorNoiseSimulator


def simulate(values, config=None, **kwargs):
    return ADCSimulator().simulate(
        np.asarray(values, dtype=float),
        config or ExperimentConfig(),
        **({"conversion_gain_e_per_dn": 2.0, "read_noise_e": 3.0} | kwargs),
    )


def test_gain_pedestal_rounding_and_clipping():
    result = simulate([[-1e300, -1, 0, 1, 2, 1e300]], analogue_gain=2, black_level=10)
    np.testing.assert_array_equal(result.codes, [[0, 9, 10, 11, 12, 16383]])
    assert result.codes.dtype == np.uint16
    np.testing.assert_array_equal(simulate([[1, 3, 5]], black_level=0).codes, [[0, 2, 2]])


@pytest.mark.parametrize("bits", [14, 16])
def test_adc_precision(bits):
    result = simulate([[0, 1e6]], ExperimentConfig(ADC_BITS=bits))
    assert result.codes.max() == (1 << bits) - 1
    assert result.applied_parameters["adc_bits"] == bits


def test_noise_profile_uses_gain_range_and_toggles():
    result = simulate([[100]], analogue_gain=4, black_level=100, white_level=1100)
    quantisation = 1 / (12 * 1000**2)
    assert result.noise_profile == pytest.approx((0.002, 0.006**2 + quantisation) * 3)
    # Rounding to integer DN remains in the stored codes when read noise is disabled.
    assert simulate([[0]], ExperimentConfig(READ_NOISE=False)).noise_profile[1] == pytest.approx(1 / (12 * 15871**2))
    assert simulate([[0]], ExperimentConfig(SHOT_NOISE=False)).noise_profile is None
    assert simulate([[0]], ExperimentConfig(READ_NOISE=False)).applied_parameters["read_noise_e"] == 0


def test_profile_matches_applied_flat_variance():
    config = ExperimentConfig()
    for signal in (300, 1200):
        expected = np.full((160, 160, 3), float(signal))
        noisy = SensorNoiseSimulator().simulate(expected, config, read_noise_e=8)
        cfa = CFAMosaicer().mosaic(noisy)
        result = simulate(cfa, config, read_noise_e=8, analogue_gain=2, white_level=8000)
        normalised = (result.codes.astype(float) - 512) / (8000 - 512)
        slope, offset = result.noise_profile[:2]
        assert np.var(normalised) == pytest.approx(slope * np.mean(normalised) + offset, rel=0.04)


def test_chunk_invariance_and_memmap(tmp_path):
    values = np.arange(77, dtype=float).reshape(11, 7) - 30
    target = np.lib.format.open_memmap(tmp_path / "codes.npy", mode="w+", dtype=np.uint16, shape=values.shape)
    result = simulate(values, strip_rows=3, out=target)
    assert result.codes is target
    np.testing.assert_array_equal(result.codes, simulate(values, strip_rows=1).codes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"conversion_gain_e_per_dn": 0},
        {"conversion_gain_e_per_dn": float("nan")},
        {"read_noise_e": -1},
        {"read_noise_e": 1e308, "analogue_gain": 1e308},
        {"analogue_gain": 0},
        {"black_level": -1},
        {"white_level": 16384},
        {"white_level": 512},
        {"white_level": True},
        {"strip_rows": 0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        simulate([[0]], **kwargs)


@pytest.mark.parametrize("values", [[], [1, 2], [[float("nan")]], [[float("inf")]]])
def test_invalid_input(values):
    with pytest.raises(ValueError):
        simulate(values)


def test_validation_precedes_output_write():
    values = np.zeros((3, 2))
    values[-1, -1] = np.nan
    out = np.full((3, 2), 99, dtype=np.uint16)
    with pytest.raises(ValueError, match="finite"):
        simulate(values, out=out, strip_rows=1)
    assert np.all(out == 99)
    with pytest.raises(ValueError, match="out"):
        simulate(np.zeros((3, 2)), out=np.zeros((3, 2)))


def test_float32_input_uses_float64_adc_arithmetic():
    values = np.array([[0, 1, 2]], dtype=np.float32)
    result = ADCSimulator().simulate(
        values, ExperimentConfig(), conversion_gain_e_per_dn=1, read_noise_e=0, analogue_gain=1e40
    )
    np.testing.assert_array_equal(result.codes, [[512, 16383, 16383]])
