"""Tiny-fixture camera geometry hypotheses and optional model gates."""

import math
from unittest.mock import Mock

import pytest

from synthraw.camera_geometry import CameraGeometryEstimator
from synthraw.camera_profile import CameraProfileManager
from synthraw.provenance import ProvenanceWriter


@pytest.fixture
def profile():
    return CameraProfileManager().build()


def test_default_rectilinear_geometry_and_serialization(profile, tmp_path):
    result = CameraGeometryEstimator().estimate(profile)
    focal = 35.9 / (2 * math.tan(math.radians(30)))
    assert result.backend == "PROFILE_DEFAULT" and result.fallback_reason is None
    assert result.records["FocalLength"].value == pytest.approx(focal)
    assert result.records["FocalLengthIn35mmFormat"].value == round(focal * math.hypot(36, 24) / math.hypot(35.9, 24))
    assert result.records["LensInfo"].value == [focal, focal, 2.8, 2.8]
    assert all(r.provenance == "F" and r.confidence is None for r in result.records.values())
    ProvenanceWriter().write(list(result.records.values()), tmp_path / "geometry.json")


def test_manual_precedence_and_optional_gate(profile):
    adapter = Mock(side_effect=AssertionError("must remain lazy"))
    estimator = CameraGeometryEstimator()
    estimator.estimate(profile, geocalib_adapter=adapter, image=object())
    manual = estimator.estimate(profile, manual_fov_degrees=90, use_geocalib=True, geocalib_adapter=adapter)
    adapter.assert_not_called()
    assert manual.backend == "MANUAL"
    assert manual.records["FocalLength"].value == pytest.approx(35.9 / 2)
    assert manual.records["FocalLength"].user_override


def test_mocked_geocalib(profile):
    image = [[[0.2, 0.3, 0.4]]]
    adapter = Mock(return_value=75.0)
    result = CameraGeometryEstimator().estimate(profile, use_geocalib=True, geocalib_adapter=adapter, image=image)
    adapter.assert_called_once_with(image)
    assert result.backend == "GEOCALIB" and result.fallback_reason is None
    assert result.records["FocalLength"].provenance == "I"
    assert result.records["LensInfo"].provenance == "F"


@pytest.mark.parametrize("error", [ImportError("missing"), RuntimeError("weights absent"), ValueError("bad output")])
def test_model_failure_falls_back(profile, error):
    result = CameraGeometryEstimator().estimate(
        profile,
        use_geocalib=True,
        geocalib_adapter=Mock(side_effect=error),
        image=object(),
        profile_fov_degrees=70,
    )
    assert result.backend == "PROFILE_DEFAULT"
    assert type(error).__name__ in result.fallback_reason
    assert result.records["horizontal_fov_degrees"].value == 70


@pytest.mark.parametrize("value", [0, 180, -1, float("nan"), float("inf"), True, "60"])
def test_invalid_manual_fov(profile, value):
    with pytest.raises(ValueError):
        CameraGeometryEstimator().estimate(profile, manual_fov_degrees=value)


def test_invalid_model_output_and_unconfigured_fallback(profile):
    estimator = CameraGeometryEstimator()
    for kwargs in ({}, {"geocalib_adapter": Mock(return_value=180), "image": object()}):
        result = estimator.estimate(profile, use_geocalib=True, **kwargs)
        assert result.backend == "PROFILE_DEFAULT" and result.fallback_reason


@pytest.mark.parametrize("kwargs", [{"maximum_aperture": 0}, {"use_geocalib": 1}, {"profile_fov_degrees": 180}])
def test_invalid_configuration(profile, kwargs):
    with pytest.raises(ValueError):
        CameraGeometryEstimator().estimate(profile, **kwargs)
