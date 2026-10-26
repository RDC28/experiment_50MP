"""Tiny camera-colour round trips and DNG matrix conventions."""

import numpy as np
import pytest

from synthraw.camera_colour import CameraColourModel, InverseISPEngine
from synthraw.provenance import FieldRecord, Provenance


def test_nonidentity_round_trip():
    matrix = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0], [0.05, 0.1, 0.85]])
    model = CameraColourModel(FieldRecord("camera_to_working", matrix.tolist(), "M", "Supplied test fit"))
    camera = np.arange(18).reshape(2, 3, 3) / 5
    gains = np.array([2, 1, 1.5])
    rendered = (camera * gains * 1.7) @ matrix.T
    result = InverseISPEngine().reconstruct(rendered, model, white_balance_gains=tuple(gains), digital_gain=1.7)
    np.testing.assert_allclose(result.camera_rgb, camera, atol=1e-14)
    np.testing.assert_allclose(result.records["AsShotNeutral"].value, 1 / gains)
    assert result.records["camera_to_working"].provenance == Provenance.MEASURED
    assert result.records["inverse_isp_digital_gain"].user_override


def test_dng_matrix_direction_and_white_adaptation():
    records = CameraColourModel().metadata()
    for key, white in [("ColorMatrix1", [1.09850, 1, 0.35585]), ("ColorMatrix2", [0.95047, 1.0000001, 1.08883])]:
        matrix = np.array(records[key].value)
        np.testing.assert_allclose(matrix @ white, np.ones(3), atol=2e-7)
        np.testing.assert_allclose(np.linalg.inv(matrix) @ matrix, np.eye(3), atol=1e-14)
        assert records[key + "_condition_number"].value == pytest.approx(np.linalg.cond(matrix))
    assert records["CalibrationIlluminant1"].value == 17
    assert records["CalibrationIlluminant2"].value == 21
    assert records["camera_to_working"].provenance == Provenance.FABRICATED


def test_preserves_signed_gamut_and_hdr():
    rgb = np.array([[[-0.1, 2, 8]]])
    np.testing.assert_array_equal(InverseISPEngine().reconstruct(rgb, CameraColourModel()).camera_rgb, rgb)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.eye(2), np.diag([1, 1, 1e-8]), np.full((3, 3), np.nan)])
def test_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        CameraColourModel(FieldRecord("camera_to_working", matrix.tolist(), "F", "Test prior"))


@pytest.mark.parametrize("gains", [(0, 1, 1), (-1, 1, 1), (1, np.nan, 1), (1, 1)])
def test_invalid_wb(gains):
    with pytest.raises(ValueError, match="White-balance"):
        InverseISPEngine().reconstruct(np.ones((2, 2, 3)), CameraColourModel(), white_balance_gains=gains)


@pytest.mark.parametrize("gain", [0, -1, np.nan, np.inf])
def test_invalid_digital_gain(gain):
    with pytest.raises(ValueError, match="digital_gain"):
        InverseISPEngine().reconstruct(np.ones((2, 2, 3)), CameraColourModel(), digital_gain=gain)


@pytest.mark.parametrize("rgb", [np.zeros((0, 1, 3)), np.ones((2, 3)), np.full((1, 1, 3), np.inf)])
def test_invalid_rgb(rgb):
    with pytest.raises(ValueError, match="RGB"):
        InverseISPEngine().reconstruct(rgb, CameraColourModel())


def test_model_snapshots_input_and_metadata():
    source = FieldRecord("camera_to_working", np.eye(3).tolist(), "F", "Test prior")
    model = CameraColourModel(source)
    source.value[0][0] = 2
    metadata = model.metadata()
    metadata["camera_to_working"].value[0][0] = 3
    assert model.metadata()["camera_to_working"].value[0][0] == 1
    with pytest.raises(ValueError):
        model.camera_to_working[0, 0] = 2
