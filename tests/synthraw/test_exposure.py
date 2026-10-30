"""Arithmetic-only exposure posterior tests for PRD_v3."""

import math

import pytest

from synthraw.camera_profile import CameraProfileManager
from synthraw.exposure import (
    APERTURE_STOPS,
    ISO_STOPS,
    MAX_DEVELOPMENT_SHIFT_EV,
    MIN_DEVELOPMENT_SHIFT_EV,
    SHUTTER_STOPS,
    ExposurePosteriorEstimator,
    exposure_and_gain,
    placement_clamp_note,
)
from synthraw.provenance import ProvenanceWriter


@pytest.fixture
def profile():
    return CameraProfileManager().build()


def test_joint_posterior_and_serialization(profile, tmp_path):
    result = ExposurePosteriorEstimator().estimate(profile)
    assert len(result.candidates) == 8
    assert sum(c.weight for c in result.candidates) == pytest.approx(1)
    assert all(0 < c.weight < 1 for c in result.candidates)
    assert list(c.weight for c in result.candidates) == sorted((c.weight for c in result.candidates), reverse=True)
    for c in result.candidates:
        assert c.exposure_time in SHUTTER_STOPS and c.exposure_time >= 1 / 8000
        assert c.f_number in APERTURE_STOPS
        assert c.iso in ISO_STOPS and 100 <= c.iso <= 32000
    records = result.records
    assert records["ExposureTime"].provenance == "I"
    assert len(records["ExposureTime"].alternatives) == 7
    assert records["ExposureTime"].confidence is None
    assert records["BaselineExposure"].provenance == "F"
    assert len(records["exposure_joint_candidates"].value) == 8
    ProvenanceWriter().write(list(records.values()), tmp_path / "metadata.json")


def test_ev_hypothesis_changes_posterior(profile):
    estimator = ExposurePosteriorEstimator()

    def ev(candidate):
        return math.log2(candidate.f_number**2 / candidate.exposure_time) - math.log2(candidate.iso / 100)

    bright = estimator.estimate(profile, scene_ev100=16)
    dark = estimator.estimate(profile, scene_ev100=3)
    assert ev(bright.candidates[0]) > ev(dark.candidates[0]) + 8
    assert estimator.estimate(profile) == estimator.estimate(profile)


@pytest.mark.parametrize("mode", ["MANUAL", "PROFILE_DEFAULT"])
def test_explicit_hypotheses(profile, mode):
    kwargs = {"manual": (1 / 125, 5.6, 100)} if mode == "MANUAL" else {}
    result = ExposurePosteriorEstimator().estimate(profile, mode=mode, **kwargs)
    assert len(result.candidates) == 1
    for name in ("ExposureTime", "FNumber", "PhotographicSensitivity"):
        record = result.records[name]
        assert record.provenance == "F" and record.confidence is None
        assert record.user_override == (mode == "MANUAL")


def test_electronic_and_expanded_limits(profile):
    estimator = ExposurePosteriorEstimator()
    triple = (1 / 32000, 1.4, 102400)
    with pytest.raises(ValueError, match="profile limits"):
        estimator.estimate(profile, mode="MANUAL", manual=triple)
    result = estimator.estimate(profile, mode="MANUAL", manual=triple, electronic_shutter=True, expanded_iso=True)
    assert result.candidates[0].iso == 102400


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "UNKNOWN"},
        {"mode": "MANUAL"},
        {"manual": (1 / 125, 5.6, 100)},
        {"scene_ev100": float("nan")},
        {"baseline_exposure": 6},
        {"exposure_bias": -6},
        {"candidate_count": 1},
        {"candidate_count": True},
        {"candidate_count": 2.5},
        {"electronic_shutter": 1},
        {"expanded_iso": "yes"},
        {"mode": "MANUAL", "manual": (0.01, 5.6, 100)},
        {"mode": "MANUAL", "manual": (True, 5.6, 100)},
    ],
)
def test_invalid_requests(profile, kwargs):
    with pytest.raises(ValueError):
        ExposurePosteriorEstimator().estimate(profile, **kwargs)


@pytest.mark.parametrize(
    ("peak", "applied_ev", "clamped"),
    ((1.0, 0.5, False), (8.0, MAX_DEVELOPMENT_SHIFT_EV, True), (0.01, MIN_DEVELOPMENT_SHIFT_EV, True)),
)
def test_exposure_placement_stays_developable(profile, peak, applied_ev, clamped):
    from synthraw.config import ExperimentConfig

    exposure = ExposurePosteriorEstimator().estimate(profile).records
    records = exposure_and_gain(ExperimentConfig(), exposure, profile.camera, hdr_peak=peak)
    assert records["applied_baseline_exposure_ev"].value == pytest.approx(applied_ev)
    gain = records["applied_analogue_gain"].value
    assert records["applied_exposure_scale"].value == pytest.approx(2.0**-applied_ev / gain)
    note = placement_clamp_note(records)
    assert (note is not None) == clamped
    if clamped:
        assert "clamped" in note
