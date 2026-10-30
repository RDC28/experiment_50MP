import json

import numpy as np
import pytest

from prd_v5.analytic import AnalyticISP
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.sensor import BayerSensor


def output(tmp_path, shape, name="adc"):
    return np.memmap(tmp_path / name, mode="w+", dtype=np.uint16, shape=shape)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_cfa_and_adc(tmp_path, pattern):
    camera = CameraHypothesis(cfa_pattern=pattern, shot_noise=False, read_noise=False)
    rgb = np.broadcast_to([0.1, 0.2, 0.3], (3, 5, 3)).copy()
    dest = output(tmp_path, (3, 5))
    report = BayerSensor(camera).process(rgb, dest)
    tile = np.array(["RGB".index(c) for c in pattern]).reshape(2, 2)
    expected = [[round(rgb[y, x, tile[y % 2, x % 2]] * 15871 + 512) for x in range(5)] for y in range(3)]
    np.testing.assert_array_equal(dest, expected)
    json.dumps(report, allow_nan=False)


def test_tiny_analytic_to_sensor(tmp_path):
    encoded = np.full((3, 5, 3), 0.5, dtype=np.float32)
    clean = np.memmap(tmp_path / "clean", mode="w+", dtype=np.float32, shape=encoded.shape)
    AnalyticISP().process(encoded, clean, strip_rows=1)
    dest = output(tmp_path, (3, 5))
    BayerSensor(CameraHypothesis(shot_noise=False, read_noise=False)).process(clean, dest)
    np.testing.assert_array_equal(dest, round(0.21404114 * 15871 + 512))


def test_strip_and_source_resolution_invariance(tmp_path):
    rgb = np.full((9, 11, 3), 0.2, dtype=np.float32)
    dest = output(tmp_path, rgb.shape[:2])
    sensor = BayerSensor()
    sensor.process(rgb, dest, strip_rows=1)
    first = dest.copy()
    sensor.process(rgb.copy(), dest, strip_rows=4)
    np.testing.assert_array_equal(dest, first)
    sensor.process(rgb, dest, strip_rows=20)
    np.testing.assert_array_equal(dest, first)


def test_gain_and_clipping(tmp_path):
    source = np.broadcast_to(np.array([0, 0.25, 2])[:, None, None], (3, 4, 3)).astype(float)
    dest = output(tmp_path, (3, 4))
    report = BayerSensor(CameraHypothesis(iso=200, shot_noise=False, read_noise=False)).process(source, dest)
    np.testing.assert_array_equal(dest[:, 0], [512, round(512 + 0.25 * 15871), 16383])
    assert report["codes_clipped_high"] == 4


def test_below_base_iso_keeps_full_well_exposure(tmp_path):
    camera = CameraHypothesis(iso=50, shot_noise=False, read_noise=False)
    source = np.broadcast_to(np.array([0, 0.6, 1])[:, None, None], (3, 4, 3)).astype(float)
    dest = output(tmp_path, (3, 4))
    report = BayerSensor(camera).process(source, dest)
    np.testing.assert_array_equal(dest[:, 0], [512, round(512 + 0.6 * 0.5 * 15871), camera.white_level])
    assert camera.white_level == 512 + round(0.5 * 15871)
    assert report["codes_clipped_high"] == 0


@pytest.mark.parametrize("signal", [0.0, 0.1, 0.5])
def test_temporal_variance(tmp_path, signal):
    # Small flat fixtures test shot slope and read intercept including quantization.
    camera = CameraHypothesis(read_noise_e=20)
    rgb = np.full((128, 128, 3), signal, dtype=np.float32)
    dest = output(tmp_path, rgb.shape[:2])
    BayerSensor(camera).process(rgb, dest)
    gain = camera.analogue_gain / camera.conversion_gain_e_per_dn
    expected_variance = (signal * camera.full_well_e + camera.read_noise_e**2) * gain**2 + 1 / 12
    assert np.var(dest.astype(float)) == pytest.approx(expected_variance, rel=0.05)
    assert np.mean(dest) == pytest.approx(512 + signal * 15871, abs=0.6)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 1e20])
def test_invalid_pixels(tmp_path, value):
    with pytest.raises(ValueError):
        BayerSensor().process(np.full((1, 1, 3), value), output(tmp_path, (1, 1)))


def test_buffer_validation_and_noop(tmp_path):
    rgb = np.zeros((2, 3, 3), dtype=float)
    sensor = BayerSensor(CameraHypothesis(read_noise_e=0))
    dest = output(tmp_path, (2, 3))
    assert sensor.process(rgb, dest)["read_noise_zero_effect"]
    with pytest.raises(ValueError, match="memmap"):
        sensor.process(rgb, np.zeros((2, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="dimensions"):
        sensor.process(rgb, output(tmp_path, (1, 3), "wrong"))
    with pytest.raises(ValueError, match="strip_rows"):
        sensor.process(rgb, dest, strip_rows=0)
