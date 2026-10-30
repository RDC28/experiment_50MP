"""Tiny camera-colour round trips and DNG matrix conventions."""

import numpy as np
import pytest

from synthraw.camera_colour import CameraColourModel, InverseISPEngine
from synthraw.experimental import CAMERA_DOMAIN
from synthraw.provenance import FieldRecord, Provenance


def test_nonidentity_round_trip():
    matrix = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0], [0.05, 0.1, 0.85]])
    model = CameraColourModel(FieldRecord("camera_to_working", matrix.tolist(), "M", "Supplied test fit"))
    camera = np.arange(18).reshape(2, 3, 3) / 5
    gains = np.array([2, 1, 1.5])
    rendered = (camera * gains * 1.7) @ matrix.T
    result = InverseISPEngine().reconstruct(
        rendered, model, white_balance_gains=tuple(gains), digital_gain=1.7, user_override=True
    )
    np.testing.assert_allclose(result.camera_rgb, camera, atol=1e-14)
    np.testing.assert_allclose(result.records["AsShotNeutral"].value, 1 / gains)
    assert result.records["camera_to_working"].provenance == Provenance.MEASURED
    assert result.records["inverse_isp_digital_gain"].user_override
    assert not result.records["AsShotNeutral"].user_override
    inferred = InverseISPEngine().reconstruct(rendered, model, white_balance_gains=tuple(gains))
    assert not inferred.records["inverse_isp_white_balance_gains"].user_override


def test_measured_xyz_matrix_gives_sony_neutrals_and_round_trips():
    from synthraw.camera_profile import ILCE1M2_DAYLIGHT_NEUTRAL, ILCE1M2_XYZ_TO_CAMERA_D65

    record = FieldRecord("xyz_to_camera_d65", ILCE1M2_XYZ_TO_CAMERA_D65, "M", "Sample fit", disclaimer="One frame")
    model = CameraColourModel.from_xyz_to_camera(record)
    written = np.array(model.metadata()["ColorMatrix2"].value)
    measured = np.array(ILCE1M2_XYZ_TO_CAMERA_D65)
    np.testing.assert_allclose(written, measured * written[1, 1] / measured[1, 1], atol=1e-12)
    np.testing.assert_allclose(np.max(written @ [0.95047, 1.0, 1.08883]), 1, atol=1e-4)
    # A D65 grey seen by the measured camera is the LibRaw daylight neutral: R and B below green.
    neutral = model.camera_neutral(np.ones(3))
    np.testing.assert_allclose(neutral, ILCE1M2_DAYLIGHT_NEUTRAL, atol=2e-3)
    assert neutral[0] < 1 and neutral[2] < 1
    gains = (1.2, 1.0, 0.8)
    result = InverseISPEngine().reconstruct(np.ones((1, 1, 3)), model, white_balance_gains=gains)
    written = result.camera_rgb[0, 0] / result.camera_rgb[0, 0, 1]
    np.testing.assert_allclose(result.records["AsShotNeutral"].value, written)
    assert result.records["camera_to_working"].provenance == Provenance.MEASURED


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


@pytest.mark.filterwarnings("ignore:.*licence restrictions:UserWarning")
@pytest.mark.parametrize("raw_flow,spiraldiff", [(False, False), (True, False), (False, True), (True, True)])
def test_learned_gates_preserve_physics_and_report_domain(raw_flow, spiraldiff):
    rgb = np.arange(18).reshape(2, 3, 3) / 4
    model = CameraColourModel()
    baseline = InverseISPEngine().reconstruct(rgb, model, white_balance_gains=(2, 1, 3), digital_gain=1.5)
    result = InverseISPEngine().reconstruct(
        rgb,
        model,
        white_balance_gains=(2, 1, 3),
        digital_gain=1.5,
        use_raw_flow=raw_flow,
        use_spiraldiff=spiraldiff,
    )
    np.testing.assert_array_equal(result.camera_rgb, baseline.camera_rgb)
    records = result.records
    assert records["inverse_isp_backend"].value == "PHYSICS"
    expected = [name for name, flag in (("RAW_FLOW", raw_flow), ("SPIRALDIFF", spiraldiff)) if flag]
    assert records["inverse_isp_requested_priors"].value == expected
    if expected:
        assert "unavailable" in records["inverse_isp_fallback_reason"].value
        assert records["inverse_isp_camera_domain"].value == CAMERA_DOMAIN
    else:
        assert records["inverse_isp_fallback_reason"].value is None
        assert records["inverse_isp_camera_domain"].value == "virtual matrix camera"
    for key in (
        "inverse_isp_backend",
        "inverse_isp_requested_priors",
        "inverse_isp_fallback_reason",
        "inverse_isp_camera_domain",
    ):
        records[key].validate()
        assert records[key].provenance == Provenance.DETERMINISTIC


@pytest.mark.parametrize("flag", ["yes", 1, None])
def test_learned_gates_reject_nonboolean(flag):
    for key in ("use_raw_flow", "use_spiraldiff"):
        with pytest.raises(ValueError, match="booleans"):
            InverseISPEngine().reconstruct(np.ones((1, 1, 3)), CameraColourModel(), **{key: flag})
