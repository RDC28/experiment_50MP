"""PRD_v3 determinism, seed, metadata dependency and optional RawTherapee validation rows."""

import json
import subprocess
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.pipeline import run_pipeline
from synthraw.sensor import SensorNoiseSimulator
from synthraw.validation import (
    MetadataDependencyValidator,
    RawTherapeeRenderer,
    SensorSeedValidator,
    hard_failures,
)


def _by_name(checks):
    return {check.name: check for check in checks}


def test_seed_checks_pass_for_the_sensor_simulator():
    checks = _by_name(SensorSeedValidator().validate(ExperimentConfig()))
    assert checks["determinism"].passed and checks["determinism"].measured["max_abs_difference_e"] == 0
    noise = checks["noise_seed_structure"]
    assert noise.passed and noise.measured["realisation_changed"]
    assert noise.measured["structure_correlation_reseeded"] > 0.98
    body = checks["body_seed_signature"]
    assert body.passed and body.measured["max_abs_difference_e"] == 0 and body.measured["body_seed_changes_map"]
    assert body.measured["dead_photosites"] > 0


def test_body_signature_fails_when_body_effects_follow_the_noise_seed(monkeypatch):
    original = SensorNoiseSimulator.simulate

    def leaky(self, expected_e, config, **kwargs):
        # A broken simulator that derives the body map from the noise seed.
        from dataclasses import replace

        return original(self, expected_e, replace(config, SENSOR_BODY_SEED=config.SENSOR_NOISE_SEED), **kwargs)

    monkeypatch.setattr(SensorNoiseSimulator, "simulate", leaky)
    checks = _by_name(SensorSeedValidator().validate(ExperimentConfig()))
    assert checks["body_seed_signature"].passed is False
    assert checks["body_seed_signature"].measured["max_abs_difference_e"] > 0


def test_noise_seed_check_fails_when_noise_swamps_structure():
    checks = _by_name(SensorSeedValidator().validate(ExperimentConfig(), max_signal_e=20.0))
    assert checks["noise_seed_structure"].passed is False


def test_compare_runs_reports_difference_and_shape_mismatch():
    a = {"raw": np.zeros((4, 4), np.uint16), "scene": np.ones((2, 2, 3))}
    same = SensorSeedValidator.compare_runs(a, {k: v.copy() for k, v in a.items()})
    assert same.passed
    changed = {"raw": a["raw"] + 1, "scene": np.ones((3, 2, 3))}
    check = SensorSeedValidator.compare_runs(a, changed)
    assert check.passed is False
    assert check.measured["max_abs_difference"] == {"raw": 1.0, "scene": "missing or shape mismatch"}


def _run(tmp_path, name):
    rows, cols = np.mgrid[0:20, 0:48]
    pixels = np.stack([255 * cols / 47, 255 * rows / 19, 128 + 0 * cols], axis=-1)
    source = tmp_path / "source.png"
    Image.fromarray(np.round(pixels).astype(np.uint8)).save(source)
    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    return run_pipeline(
        source, config, tmp_path / name, timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=96
    ).root


@pytest.fixture(scope="module")
def pipeline_runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("determinism")
    return _run(tmp_path, "first"), _run(tmp_path, "second")


def test_pipeline_is_deterministic_and_metadata_consistent(pipeline_runs):
    first, second = pipeline_runs
    names = (
        "raw/sensor_mosaic_uint16.npy",
        "intermediates/03_scene_linear.npy",
        "intermediates/08_noisy_electrons.npy",
    )
    load = lambda root: {name: np.load(root / name, mmap_mode="r") for name in names}  # noqa: E731
    assert SensorSeedValidator.compare_runs(load(first), load(second)).passed
    groups = json.loads((first / "metadata" / "metadata.json").read_text())["groups"]
    check = MetadataDependencyValidator().validate(groups)
    assert check.passed, check.measured["violations"]
    assert hard_failures([check]) == []


def _mutate(groups, group, field, value):
    for record in groups[group]:
        if record["field"] == field:
            record["value"] = value
    return groups


@pytest.mark.parametrize(
    ("group", "field", "value", "message"),
    [
        ("dng", "ShutterSpeedValue", 3.0, "APEX"),
        ("dng", "BlackLevel", 20000, "levels"),
        ("dng", "ActiveArea", [0, 0, 999, 10], "ActiveArea"),
        ("dng", "CFAPattern", [2, 1, 1, 0], "applied phase"),
        ("dng", "NoiseProfile", [-1.0, 0.0], "S > 0"),
        ("dng", "AsShotNeutral", [1.0, 0.0, 1.0], "AsShotNeutral"),
        ("dng", "Make", "SONY", "Make"),
        ("exposure_posterior", "FNumber", 8.0, "disagrees"),
        ("white_balance", "AsShotNeutral", [0.5, 1.0, 0.5], "disagrees"),
    ],
)
def test_impossible_combinations_fail_the_hard_gate(pipeline_runs, group, field, value, message):
    groups = json.loads((pipeline_runs[0] / "metadata" / "metadata.json").read_text())["groups"]
    check = MetadataDependencyValidator().validate(_mutate(groups, group, field, value))
    assert check.passed is False and check.hard
    assert any(message in violation for violation in check.measured["violations"])


def test_dangling_dependency_and_fabricated_confidence_are_violations():
    groups = {
        "a": [
            {"field": "x", "value": 1, "provenance": "I", "dependencies": ["missing"]},
            {"field": "y", "value": 2, "provenance": "F", "confidence": 0.5, "disclaimer": "d"},
            {"field": "GPSLatitude", "value": 1.0, "provenance": "F", "disclaimer": "d"},
            {"field": "z", "value": 3, "provenance": "Q"},
        ]
    }
    violations = MetadataDependencyValidator().validate(groups).measured["violations"]
    assert len(violations) == 4
    assert "unknown field 'missing'" in violations[0]


def test_rawtherapee_is_report_only(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: None)
    missing = RawTherapeeRenderer().render(tmp_path / "x.dng", tmp_path)
    assert missing.passed is None and missing.measured["available"] is False

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/rawtherapee-cli")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        (tmp_path / "out" / "x_rawtherapee.tif").write_bytes(b"tiff")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rendered = RawTherapeeRenderer().render(tmp_path / "x.dng", tmp_path / "out")
    assert rendered.passed is None and rendered.measured["rendered"]
    assert calls[0][:2] == ["/usr/bin/rawtherapee-cli", "-o"] and calls[0][-2:] == ["-c", str(tmp_path / "x.dng")]

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    failed = RawTherapeeRenderer().render(tmp_path / "y.dng", tmp_path / "out")
    assert failed.passed is None and "boom" in failed.detail
    assert hard_failures([missing, rendered, failed]) == []
