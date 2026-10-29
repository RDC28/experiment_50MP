import json

import numpy as np
import pytest

from prd_v5.analytic import AnalyticISP


def output(tmp_path, shape):
    return np.memmap(tmp_path / "camera.bin", mode="w+", shape=shape, dtype=np.float32)


def test_srgb_transfer_and_clipping(tmp_path):
    source = np.array([[[0, 0.04045, 1], [0.5, 0.5, 0.5]]], dtype=np.float32)
    dest = output(tmp_path, source.shape)
    report = AnalyticISP().process(source, dest, strip_rows=1)
    np.testing.assert_allclose(dest[0, 0], [0, 0.04045 / 12.92, 1], atol=1e-7)
    np.testing.assert_allclose(dest[0, 1], 0.21404114, atol=1e-7)
    assert report["source_clipped_pixels"] == 1
    json.dumps(report, allow_nan=False)


def test_forward_inverse_roundtrip_and_strip_invariance(tmp_path):
    camera = np.random.default_rng(42).uniform(0.01, 0.15, (7, 5, 3))
    matrix = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    wb = np.array([2, 1, 1.5])
    linear = (camera * wb * 1.2) @ matrix.T
    toned = 3 * linear**2 - 2 * linear**3
    encoded = np.where(toned <= 0.0031308, 12.92 * toned, 1.055 * toned ** (1 / 2.4) - 0.055)
    isp = AnalyticISP(matrix, wb, 1.2, True)
    dest = output(tmp_path, camera.shape)
    report = isp.process(encoded, dest, strip_rows=2)
    assert report["forward_consistency"]["status"] == "passed"
    assert report["forward_consistency"]["rmse"] < 1e-6
    np.testing.assert_allclose(dest, camera, atol=2e-7)
    first = dest.copy()
    isp.process(encoded, dest, strip_rows=7)
    np.testing.assert_array_equal(first, dest)
    matrix[:] = 0
    assert isp.camera_to_srgb[0][0] == 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_to_srgb": ((0, 0, 0),) * 3},
        {"white_balance": (1, 0, 1)},
        {"exposure_gain": float("nan")},
        {"exposure_gain": -1},
        {"inverse_smoothstep": 1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        AnalyticISP(**kwargs)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_pixels(tmp_path, value):
    source = np.full((1, 1, 3), value)
    with pytest.raises(ValueError, match="normalized"):
        AnalyticISP().process(source, output(tmp_path, source.shape))


def test_buffer_contract(tmp_path):
    source = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="memmap"):
        AnalyticISP().process(source, source.copy())
    dest = output(tmp_path, source.shape)
    with pytest.raises(ValueError, match="overlap"):
        AnalyticISP().process(dest, dest)
    with pytest.raises(ValueError, match="strip_rows"):
        AnalyticISP().process(source, dest, strip_rows=0)


def test_out_of_gamut_report(tmp_path):
    source = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    isp = AnalyticISP(((1, 0.5, 0), (0, 1, 0), (0, 0, 1)))
    dest = output(tmp_path, source.shape)
    report = isp.process(source, dest)
    assert report["negative_camera_samples_floored"] == 1
    np.testing.assert_array_equal(dest, [[[0, 1, 0]]])
    consistency = report["forward_consistency"]
    assert consistency["status"] == "failed"
    expected_error = 1.055 * 0.5 ** (1 / 2.4) - 0.055
    assert consistency["max_absolute_error"] == pytest.approx(expected_error)
    assert consistency["rmse"] == pytest.approx(expected_error / np.sqrt(3))


def test_forward_consistency_endpoints_and_strip_metrics(tmp_path):
    source = np.random.default_rng(3).uniform(0, 1, (9, 7, 3)).astype(np.float32)
    source[0, 0] = [0, 0.04045, 1]
    dest = output(tmp_path, source.shape)
    isp = AnalyticISP(white_balance=(2, 1, 1.5), exposure_gain=0.7)
    first = isp.process(source, dest, strip_rows=1)["forward_consistency"]
    second = isp.process(source, dest, strip_rows=9)["forward_consistency"]
    assert first["status"] == "passed"
    assert first["max_absolute_error"] < 1e-6
    assert first["max_absolute_error"] == second["max_absolute_error"]
    assert first["rmse"] == pytest.approx(second["rmse"], abs=1e-15)
    json.dumps(first, allow_nan=False)
