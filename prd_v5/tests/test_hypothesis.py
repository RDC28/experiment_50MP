"""Lightweight MVP camera-state contracts, without inference or image allocation."""

import json
import random
from dataclasses import FrozenInstanceError, replace

import pytest

from prd_v5.hypothesis import CameraField, CameraHypothesis


def test_shared_draw_and_streams_are_immutable_and_resolution_independent():
    before = random.getstate()
    camera = CameraHypothesis.sample(42)
    assert random.getstate() == before
    assert camera == CameraHypothesis.sample(42)
    assert camera != CameraHypothesis.sample(43)
    assert 20000 <= camera.full_well_e <= 100000
    assert 1 <= camera.read_noise_e <= 5
    seeds_2k = {stage: camera.stream_seed(stage) for stage in ("shot", "read", "body")}
    seeds_4k = {stage: camera.stream_seed(stage) for stage in ("body", "shot", "read")}
    assert seeds_2k == seeds_4k
    assert len(set(seeds_2k.values())) == 3
    with pytest.raises(FrozenInstanceError):
        camera.iso = 200


def test_electrons_adc_and_noise_metadata_describe_same_exposure():
    camera = CameraHypothesis(iso=400)
    span = camera.white_level - camera.black_level
    dn_per_e = camera.analogue_gain / camera.conversion_gain_e_per_dn
    assert camera.saturation_e * dn_per_e == pytest.approx(span)
    assert camera.black_level + camera.saturation_e * dn_per_e == pytest.approx(16383)
    # Poisson variance plus read variance and uniform one-DN rounding variance.
    electrons = 1000
    variance_dn = (electrons + camera.read_noise_e**2) * dn_per_e**2 + 1 / 12
    signal_normalized = electrons * dn_per_e / span
    slope, offset = camera.noise_profile[:2]
    assert slope * signal_normalized + offset == pytest.approx(variance_dn / span**2)
    assert camera.noise_profile == (slope, offset) * 3
    assert replace(camera, shot_noise=False).noise_profile is None
    no_read = replace(camera, read_noise=False)
    assert no_read.noise_profile[1] == pytest.approx(1 / (12 * span**2))


def test_registry_is_json_serializable_honest_and_detached():
    camera = CameraHypothesis.sample()
    registry = json.loads(json.dumps(camera.registry(), allow_nan=False))
    for name, field in registry.items():
        assert field["name"] == name
        assert field["method"] and field["validation"]
        assert field["confidence"] is None
        assert field["posterior"] is None
    for name in ("gps", "serial_number", "timestamp", "shutter_count"):
        assert registry[name]["value"] is None
        assert registry[name]["mode"] == "USER"
        assert registry[name]["provenance"] == "UNSPECIFIED"
    registry["iso"]["value"] = 32000
    assert camera.registry()["iso"]["value"] == camera.iso == 100
    assert registry["noise_profile"]["provenance"] == "DERIVED"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**64},
        {"adc_bits": 17},
        {"black_level": 16383},
        {"black_level": -1},
        {"black_level": False},
        {"iso": 0},
        {"iso": 100.0},
        {"full_well_e": 0},
        {"full_well_e": float("nan")},
        {"read_noise_e": -1},
        {"shutter_s": float("inf")},
        {"aperture": False},
        {"cfa_pattern": "RGB"},
        {"shot_noise": 1},
        {"read_noise": "false"},
    ],
)
def test_invalid_hypotheses_fail_before_execution(kwargs):
    with pytest.raises(ValueError):
        CameraHypothesis(**kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "INFER"},
        {"provenance": "RECOVERED"},
        {"confidence": float("nan")},
        {"confidence": 1.1},
        {"method": ""},
        {"value": float("inf")},
    ],
)
def test_invalid_registry_records_are_rejected(changes):
    values = dict(
        name="iso",
        value=100,
        units="ISO",
        mode="AUTO",
        provenance="SYNTHETIC",
        confidence=None,
        posterior=None,
        method="Explicit hypothesis",
        validation="positive",
    )
    values.update(changes)
    with pytest.raises(ValueError):
        CameraField(**values)
