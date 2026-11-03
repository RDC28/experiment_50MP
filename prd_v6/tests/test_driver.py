"""Driver steps 3-6 with the heavy work replaced: gating, guarding and run state (section 9.2)."""

from __future__ import annotations

import json

import pytest

from prd_v6 import benchmark, delivery, driver, preflight, production, sweeps

PRODUCTION_RUN = production.run


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def record(name, result=None):
        def fake(*args, **kwargs):
            seen.append(name)
            return result

        return fake

    monkeypatch.setattr(
        preflight,
        "run",
        record("preflight", {"arw": "ref.arw", "model": None, "dng_validate": "x", "source_errors": {}}),
    )
    monkeypatch.setattr(benchmark, "run", record("benchmark", {"status": "PASS"}))
    monkeypatch.setattr(production, "run", record("production"))
    monkeypatch.setattr(sweeps, "run", record("sweeps"))
    return seen


def statuses(run_dir):
    return {s["step"]: s["status"] for s in delivery.load_state(run_dir)["steps"]}


def test_benchmark_only_run_skips_production(tmp_path, calls):
    driver.run(tmp_path, {"PRESET": "max"}, log=lambda m: None)
    assert calls == ["preflight", "benchmark", "sweeps"]
    assert statuses(tmp_path) == {3: "finished", 4: "finished", 6: "finished"}
    assert (tmp_path / "archives").is_dir()


def test_production_failure_still_runs_sweeps(tmp_path, calls, monkeypatch):
    monkeypatch.setattr(production, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("bad source")))
    driver.run(tmp_path, {"PRESET": "max", "SOURCE_4K_URL": "https://example.invalid/a.jpg"}, log=lambda m: None)
    assert calls == ["preflight", "benchmark", "sweeps"]
    assert statuses(tmp_path) == {3: "finished", 4: "finished", 5: "failed", 6: "finished"}


def test_a_bad_own_image_is_reported_next_to_the_verdict(tmp_path, calls, monkeypatch):
    """PRD 9.2 and 6.6: a SOURCE_4K_URL that preflight could not decode fails step 5 only; the PASS stands."""
    error = "SOURCE_4K_URL: the download (132597 bytes) is not a decodable image (UnidentifiedImageError)"
    inputs = {"arw": "ref.arw", "model": None, "dng_validate": "x", "source_errors": {"SOURCE_4K_URL": error}}
    monkeypatch.setattr(preflight, "run", lambda *a, **k: inputs)
    monkeypatch.setattr(production, "run", PRODUCTION_RUN)

    def passed(config):
        calls.append("benchmark")
        (tmp_path / "verdict.json").write_text(json.dumps({"status": "PASS", "text": "PASS.", "outputs": {}}))

    monkeypatch.setattr(benchmark, "run", passed)
    driver.run(
        tmp_path, {"PRESET": "max", "SOURCE_4K_URL": "https://drive.google.com/file/d/a/view"}, log=lambda m: None
    )
    assert calls == ["benchmark", "sweeps"]
    assert statuses(tmp_path) == {3: "finished", 4: "finished", 5: "failed", 6: "finished"}
    record = json.loads((tmp_path / "production" / "production.json").read_text())
    assert record["sources"] == {"4k_source": {"error": error}}
    verdict = delivery.finalize_verdict(tmp_path)
    assert verdict["status"] == "PASS"
    assert verdict["completed_with_errors"] == [f"step 5 (Production) raised RuntimeError: {error}"]


def test_preflight_failure_stops_before_the_benchmark(tmp_path, calls, monkeypatch):
    def reject(*args, **kwargs):
        raise preflight.PreflightError("Tesla T4 is rejected")

    monkeypatch.setattr(preflight, "run", reject)
    with pytest.raises(preflight.PreflightError):
        driver.run(tmp_path, {"PRESET": "max"}, log=lambda m: None)
    assert calls == []
    assert statuses(tmp_path) == {3: "failed"}
    verdict = delivery.finalize_verdict(tmp_path)
    assert verdict["status"] == "INVALID" and "Tesla T4 is rejected" in verdict["text"]


def test_only_the_max_preset_exists(tmp_path, calls):
    with pytest.raises(ValueError, match="only the 'max' preset"):
        driver.run(tmp_path, {"PRESET": "fast"}, log=lambda m: None)
    assert not (tmp_path / "state.json").exists() or json.loads((tmp_path / "state.json").read_text())["steps"] == []
