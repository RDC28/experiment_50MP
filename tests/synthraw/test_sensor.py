"""Tiny expected-electron fixtures for the PRD_v3 sensor milestone."""

import numpy as np
import pytest

from synthraw.sensor import ElectronSignalSimulator


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
