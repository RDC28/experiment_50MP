"""Tiny expected-electron fixtures for the PRD_v3 sensor milestone."""

from dataclasses import replace

import numpy as np
import pytest

from synthraw.config import ExperimentConfig
from synthraw.sensor import ElectronSignalSimulator, SensorNoiseSimulator


@pytest.mark.parametrize("rows", [1, 3, 32])
def test_expected_electrons_preserve_hdr_and_apply_linear_exposure(rows):
    source = np.tile([-0.1, 0.5, 3.0], (7, 5, 1))
    result = ElectronSignalSimulator().simulate(source, 52000, exposure_scale=0.25, strip_rows=rows)
    np.testing.assert_array_equal(result, np.tile([0, 6500, 39000], (7, 5, 1)))
    np.testing.assert_array_equal(source[0, 0], [-0.1, 0.5, 3])


def test_npy_memmap_and_chunk_invariance(tmp_path):
    source = np.random.default_rng(3).uniform(-1, 4, (13, 11, 3)).astype(np.float32)
    out = np.lib.format.open_memmap(tmp_path / "07_expected_electrons.npy", mode="w+", shape=source.shape)
    simulator = ElectronSignalSimulator()
    assert simulator.simulate(source, 52000, out=out, strip_rows=2) is out
    out.flush()
    np.testing.assert_array_equal(np.load(tmp_path / "07_expected_electrons.npy"), simulator.simulate(source, 52000))


def test_zero_exposure_is_dark():
    np.testing.assert_array_equal(ElectronSignalSimulator().simulate(np.ones((2, 3, 3)), 52000, exposure_scale=0), 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"full_well_electrons": 0},
        {"full_well_electrons": True},
        {"full_well_electrons": float("inf")},
        {"exposure_scale": -1},
        {"exposure_scale": float("nan")},
        {"strip_rows": 0},
        {"strip_rows": True},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ElectronSignalSimulator().simulate(np.ones((2, 3, 3)), **({"full_well_electrons": 52000} | kwargs))


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 4), (0, 3, 3)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError, match="RGB"):
        ElectronSignalSimulator().simulate(np.ones(shape), 52000)


@pytest.mark.parametrize("bad", [np.nan, np.inf, np.finfo(float).max])
def test_validation_is_atomic(bad):
    source = np.ones((7, 5, 3))
    source[-1, -1, -1] = bad
    out = np.full_like(source, -10)
    with pytest.raises(ValueError, match="finite"):
        ElectronSignalSimulator().simulate(source, 52000, out=out, strip_rows=2)
    np.testing.assert_array_equal(out, -10)


def test_output_contract_and_integer_input():
    source = np.ones((2, 3, 3))
    for out in (source, np.empty(source.shape, dtype=np.float32), np.empty((1, 3, 3))):
        with pytest.raises(ValueError, match="out"):
            ElectronSignalSimulator().simulate(source, 52000, out=out)
    with pytest.raises(ValueError, match="floating"):
        ElectronSignalSimulator().simulate(source.astype(int), 52000)


def test_flat_poisson_read_statistics():
    values = SensorNoiseSimulator().simulate(np.full((128, 128, 3), 100.0), ExperimentConfig(), read_noise_e=4)
    assert abs(values.mean() - 100) < 0.2
    assert abs(values.var() - 116) < 3


def test_noise_chunk_invariance_and_memmap(tmp_path):
    config = ExperimentConfig(PRNU=True, DSNU=True, ROW_FPN=True, COLUMN_FPN=True, HOT_PIXELS=True, DEAD_PIXELS=True)
    expected = np.full((17, 19, 3), 40.0)
    kwargs = dict(
        read_noise_e=3,
        prnu_sigma=0.02,
        dsnu_e=2,
        row_fpn_e=1,
        column_fpn_e=2,
        hot_pixel_fraction=0.1,
        dead_pixel_fraction=0.1,
    )
    simulator = SensorNoiseSimulator()
    reference = simulator.simulate(expected, config, **kwargs, strip_rows=1)
    out = np.lib.format.open_memmap(tmp_path / "noise.npy", mode="w+", dtype=np.float64, shape=expected.shape)
    assert simulator.simulate(expected, config, **kwargs, strip_rows=7, out=out) is out
    np.testing.assert_array_equal(reference, out)


def test_body_and_noise_seed_semantics():
    simulator = SensorNoiseSimulator()
    expected = np.full((19, 17, 3), 100.0)
    config = ExperimentConfig(
        SHOT_NOISE=False,
        READ_NOISE=False,
        PRNU=True,
        DSNU=True,
        ROW_FPN=True,
        COLUMN_FPN=True,
        HOT_PIXELS=True,
        DEAD_PIXELS=True,
    )
    kwargs = dict(
        read_noise_e=3,
        prnu_sigma=0.1,
        dsnu_e=2,
        row_fpn_e=1,
        column_fpn_e=2,
        hot_pixel_fraction=0.1,
        dead_pixel_fraction=0.1,
    )
    fixed = simulator.simulate(expected, config, **kwargs)
    np.testing.assert_array_equal(fixed, simulator.simulate(expected, replace(config, SENSOR_NOISE_SEED=8), **kwargs))
    assert not np.array_equal(fixed, simulator.simulate(expected, replace(config, SENSOR_BODY_SEED=8), **kwargs))
    temporal = replace(config, SHOT_NOISE=True, READ_NOISE=True)
    assert not np.array_equal(
        simulator.simulate(expected, temporal, **kwargs),
        simulator.simulate(expected, replace(temporal, SENSOR_NOISE_SEED=8), **kwargs),
    )


def test_disabled_toggles_and_dark_defects():
    simulator = SensorNoiseSimulator()
    expected = np.full((3, 5, 3), 10.0)
    config = ExperimentConfig(SHOT_NOISE=False, READ_NOISE=False)
    kwargs = dict(
        read_noise_e=10,
        dark_current_e_per_s=2,
        exposure_time_s=3,
        hot_pixel_fraction=1,
        hot_pixel_e_per_s=5,
        dead_pixel_fraction=1,
    )
    np.testing.assert_array_equal(simulator.simulate(expected, config, **kwargs), expected)
    np.testing.assert_array_equal(simulator.simulate(expected, replace(config, DARK_CURRENT=True), **kwargs), 16)
    np.testing.assert_array_equal(simulator.simulate(expected, replace(config, HOT_PIXELS=True), **kwargs), 25)
    np.testing.assert_array_equal(
        simulator.simulate(expected, replace(config, HOT_PIXELS=True, DEAD_PIXELS=True), **kwargs), 0
    )


@pytest.mark.parametrize(
    "kwargs", [{"read_noise_e": -1}, {"prnu_sigma": np.nan}, {"strip_rows": 0}, {"hot_pixel_fraction": 1.1}]
)
def test_noise_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SensorNoiseSimulator().simulate(np.ones((2, 3, 3)), ExperimentConfig(), **({"read_noise_e": 2} | kwargs))


def test_noise_invalid_input_is_atomic():
    expected = np.ones((3, 5, 3))
    expected[-1, -1, -1] = -1
    out = np.full_like(expected, -99)
    with pytest.raises(ValueError):
        SensorNoiseSimulator().simulate(expected, ExperimentConfig(), read_noise_e=2, out=out, strip_rows=1)
    np.testing.assert_array_equal(out, -99)
