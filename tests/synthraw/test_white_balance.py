"""Tiny illuminant and override checks for the PRD_v3 camera milestone."""

import numpy as np
import pytest

from synthraw.camera_profile import CameraProfileManager
from synthraw.provenance import Provenance
from synthraw.white_balance import WhiteBalanceEstimator


@pytest.fixture
def profile():
    return CameraProfileManager().build()


@pytest.mark.parametrize("method", ["GREY_WORLD", "WHITE_PATCH", "COMBINED"])
@pytest.mark.parametrize("temperature,tint", [(2500, 0), (4200, 0.12), (6500, -0.1), (9900, 0)])
def test_inferred_neutral_and_temperature_round_trip(profile, method, temperature, tint):
    estimator = WhiteBalanceEstimator()
    manual = estimator.estimate(None, profile, manual_cct=temperature, manual_tint=tint)
    neutral = np.array(manual.records["AsShotNeutral"].value)
    image = np.linspace(0.1, 2, 48).reshape(6, 8, 1) * neutral
    result = estimator.estimate(image, profile, method=method)
    assert result.records["CCT"].value == pytest.approx(temperature, abs=4)
    assert result.records["tint"].value == pytest.approx(tint, abs=0.002)
    assert result.records["AsShotNeutral"].value == pytest.approx(neutral)
    np.testing.assert_allclose(image * result.gains, np.broadcast_to(image[..., 1:2], image.shape))
    for record in result.records.values():
        assert record.provenance == Provenance.INFERRED
        assert 0 <= record.confidence <= 0.5
        assert not record.user_override


def test_manual_override_does_not_depend_on_pixels(profile):
    result = WhiteBalanceEstimator().estimate(None, profile, manual_cct=3200, manual_tint=0.2)
    assert result.records["LightSource"].value == 3
    for record in result.records.values():
        assert record.user_override
        assert record.provenance == Provenance.FABRICATED
        assert record.confidence is None
        record.to_dict()
    assert result.records["AsShotNeutral"].value[1] == 1


def test_white_patch_uses_joint_bright_pixels_and_combined_agreement(profile):
    image = np.full((10, 10, 3), [0.4, 0.2, 0.1])
    image[0] = [1, 1, 1]
    estimator = WhiteBalanceEstimator()
    grey = estimator.estimate(image, profile, method="GREY_WORLD")
    white = estimator.estimate(image, profile, method="WHITE_PATCH")
    combined = estimator.estimate(image, profile)
    assert white.records["AsShotNeutral"].value == [1, 1, 1]
    np.testing.assert_allclose(
        combined.records["AsShotNeutral"].value,
        np.sqrt(grey.records["AsShotNeutral"].value),
    )
    assert combined.records["CCT"].confidence < 0.5


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2, 3)),
        np.ones((2, 3)),
        np.empty((0, 2, 3)),
        np.full((2, 2, 3), np.nan),
        np.full((2, 2, 3), -1),
        np.full((2, 2, 3), [1, 0, 1]),
    ],
)
def test_unusable_images_fail_explicitly(profile, image):
    with pytest.raises(ValueError):
        WhiteBalanceEstimator().estimate(image, profile)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"manual_cct": 2499},
        {"manual_cct": 9901},
        {"manual_cct": np.nan},
        {"manual_cct": 5000, "manual_tint": np.inf},
        {"manual_tint": 0.1},
        {"method": "MAGIC"},
    ],
)
def test_invalid_settings_rejected(profile, kwargs):
    with pytest.raises(ValueError):
        WhiteBalanceEstimator().estimate(np.ones((2, 2, 3)), profile, **kwargs)


def test_inferred_temperature_stays_within_profile(profile):
    result = WhiteBalanceEstimator().estimate(np.full((2, 2, 3), [100, 1, 0.01]), profile)
    assert result.records["CCT"].value == 2500
