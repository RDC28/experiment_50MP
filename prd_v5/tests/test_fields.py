"""Field mode selection uses supplied evidence, never an implicit estimator."""

import pytest

from prd_v5.fields import USER_ONLY_FIELDS, resolve_field
from prd_v5.hypothesis import CameraField, CameraHypothesis


def evidence(provenance, value=200, name="iso", units=None):
    return CameraField(name, value, units, "AUTO", provenance, None, None, "Test evidence", "Domain checked by caller")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("AUTO", "USER_SET"), ("USER", "USER_SET"), ("ESTIMATE", "ESTIMATED"), ("FABRICATE", "PRIOR_SAMPLED")],
)
def test_requested_mode_selects_eligible_evidence(mode, expected):
    candidates = [evidence(p) for p in ("SYNTHETIC", "PRIOR_SAMPLED", "ESTIMATED", "USER_SET")]
    result = resolve_field("iso", None, mode, candidates)
    assert result.provenance == expected
    assert result.mode == mode
    # Selected values can drive actual coupled electronics, not merely labels.
    camera = CameraHypothesis(iso=result.value)
    assert camera.analogue_gain == 2
    assert camera.saturation_e == camera.full_well_e / 2


def test_unavailable_estimate_is_null_without_prior_fallback():
    result = resolve_field("iso", None, "ESTIMATE", [evidence("PRIOR_SAMPLED")])
    assert result.value is None
    assert result.provenance == "UNSPECIFIED"
    assert result.confidence is None and result.posterior is None
    assert result.mode == "ESTIMATE"


@pytest.mark.parametrize("mode", ["AUTO", "FABRICATE"])
def test_unobservable_fabrication_requires_explicit_gate(mode):
    candidate = evidence("SYNTHETIC", 25, "sensor_temperature", "C")
    assert resolve_field("sensor_temperature", "C", mode, [candidate], unobservable_metadata=True).value is None
    enabled = resolve_field(
        "sensor_temperature",
        "C",
        mode,
        [candidate],
        unobservable_metadata=True,
        fabricate_unobservable_metadata=True,
    )
    assert enabled.value == 25 and enabled.provenance == "SYNTHETIC"


@pytest.mark.parametrize("name", sorted(USER_ONLY_FIELDS))
def test_identity_is_user_only_even_with_fabrication_enabled(name):
    with pytest.raises(ValueError, match="requires USER"):
        resolve_field(name, None, "FABRICATE", [evidence("SYNTHETIC", name=name)], fabricate_unobservable_metadata=True)
    assert resolve_field(name, None, "USER", [evidence("SYNTHETIC", name=name)]).value is None
    assert resolve_field(name, None, "USER", [evidence("USER_SET", "supplied", name)]).value == "supplied"


def test_auto_prioritizes_observation_and_preserves_uncertainty_without_aliasing():
    from dataclasses import replace

    observed = replace(evidence("OBSERVED", [100]), confidence=0.8, posterior={"interval": [90, 110]})
    result = resolve_field("iso", None, "AUTO", [evidence("SYNTHETIC"), observed])
    assert result.provenance == "OBSERVED" and result.confidence == 0.8
    result.value.append(300)
    result.posterior["interval"].append(120)
    assert observed.value == [100] and observed.posterior == {"interval": [90, 110]}


@pytest.mark.parametrize(
    "candidates", [[evidence("OBSERVED"), evidence("OBSERVED")], [evidence("OBSERVED", name="aperture")], [object()]]
)
def test_ambiguous_or_mismatched_evidence_is_rejected(candidates):
    with pytest.raises(ValueError):
        resolve_field("iso", None, "AUTO", candidates)


def test_invalid_mode_and_nonboolean_gate_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        resolve_field("iso", None, "UNKNOWN")
    with pytest.raises(ValueError, match="booleans"):
        resolve_field("iso", None, "AUTO", fabricate_unobservable_metadata=1)


def test_camera_resolution_recomputes_electronics_and_preserves_base():
    from prd_v5.fields import resolve_camera

    base = CameraHypothesis()
    camera, records = resolve_camera(base, {"iso": "USER"}, {"iso": [evidence("USER_SET", 800, units="ISO")]})
    assert camera.iso == 800 and camera.analogue_gain == 8
    assert camera.saturation_e == base.full_well_e / 8
    assert camera.noise_profile[0] == 8 / base.full_well_e
    assert base.iso == 100
    selected = {record.name: record for record in records}
    assert selected["iso"].mode == "USER" and selected["iso"].provenance == "USER_SET"
    assert selected["full_well_e"].provenance == "SYNTHETIC"
    assert selected["full_well_e"].value == base.full_well_e


@pytest.mark.parametrize("mode", ["USER", "ESTIMATE"])
def test_camera_resolution_never_silently_reuses_missing_evidence(mode):
    from prd_v5.fields import resolve_camera

    with pytest.raises(ValueError, match="Unresolved camera input: iso"):
        resolve_camera(CameraHypothesis(), {"iso": mode})


@pytest.mark.parametrize("name", ["white_level", "gps", "misspelled"])
def test_camera_resolution_rejects_unknown_and_derived_inputs(name):
    from prd_v5.fields import resolve_camera

    for modes, candidates in [({name: "AUTO"}, {}), ({}, {name: []})]:
        with pytest.raises(ValueError, match="Unknown or derived"):
            resolve_camera(CameraHypothesis(), modes, candidates)


def test_camera_resolution_validates_selected_domains_and_explicit_empty_candidates():
    from prd_v5.fields import resolve_camera

    with pytest.raises(ValueError, match="ISO"):
        resolve_camera(CameraHypothesis(), {"iso": "USER"}, {"iso": [evidence("USER_SET", 50, units="ISO")]})
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_camera(CameraHypothesis(), {}, {"iso": []})
    camera, records = resolve_camera(CameraHypothesis(), {})
    assert camera == CameraHypothesis() and len(records) == 11


def test_resolved_camera_drives_tiny_adc(tmp_path):
    import numpy as np

    from prd_v5.fields import resolve_camera
    from prd_v5.sensor import BayerSensor

    camera, _ = resolve_camera(
        CameraHypothesis(shot_noise=False, read_noise=False),
        {"iso": "ESTIMATE"},
        {"iso": [evidence("ESTIMATED", 400, units="ISO")]},
    )
    dest = np.memmap(tmp_path / "adc", mode="w+", dtype=np.uint16, shape=(2, 2))
    try:
        BayerSensor(camera).process(np.full((2, 2, 3), 0.125), dest)
        np.testing.assert_array_equal(dest, round(512 + 0.5 * (16383 - 512)))
    finally:
        dest._mmap.close()
