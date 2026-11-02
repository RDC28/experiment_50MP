"""Run state, final verdict, archives and recovery (sections 6.6, 9.2, 10) on synthetic run folders."""

from __future__ import annotations

import json
import zipfile

import pytest

from prd_v6 import delivery

OUTPUT = {"label": "2K", "passes": True, "planes": {p: {"A": 0.97, "P2": 0.98} for p in "RGB"}}


def make_run(tmp_path, *, steps, verdict_status=None, settings=None):
    run_dir = tmp_path / "20261102T000000Z_abcdef0"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"settings": settings or {}}))
    for step, outcome in steps:
        delivery.mark_step_started(run_dir, step)
        if outcome == "finished":
            delivery.mark_step_finished(run_dir, step)
        elif outcome == "failed":
            try:
                raise RuntimeError(f"boom in {step}")
            except RuntimeError as error:
                delivery.mark_step_finished(run_dir, step, error)
    if verdict_status:
        verdict = {
            "status": verdict_status,
            "text": f"{verdict_status}: A = 97.0%",
            "outputs": {"2k": OUTPUT},
            "invalid_reasons": [],
            "warnings": [],
        }
        (run_dir / "verdict.json").write_text(json.dumps(verdict))
        folder = run_dir / "benchmark" / "2k"
        folder.mkdir(parents=True)
        (folder / "exp50v6_2k.dng").write_bytes(b"\0" * 1000)
        (folder / "metrics.json").write_text("{}")
    return run_dir


def names(archives):
    return sorted(a.name.split("_", 2)[2] for a in archives)


def test_pass_run_packages_summary_and_data_archives(tmp_path, capsys):
    run_dir = make_run(tmp_path, steps=[(s, "finished") for s in (2, 3, 4, 6)], verdict_status="PASS")
    verdict, archives = delivery.finalize_run(run_dir, download=True)
    assert verdict["status"] == "PASS" and verdict["final"]
    assert verdict["completed_with_errors"] == []
    assert names(archives) == ["01_summary.zip", "02_benchmark_2k.zip"]
    with zipfile.ZipFile(archives[0]) as summary:
        listed = summary.namelist()
    assert {"verdict.json", "state.json", "archives.json", "manifest.json"} <= set(listed)
    assert "benchmark/2k/metrics.json" in listed
    assert not any(name.endswith(".dng") for name in listed)
    with zipfile.ZipFile(archives[1]) as data:
        assert "benchmark/2k/exp50v6_2k.dng" in data.namelist()
    out = capsys.readouterr().out
    assert "PRD v6 RUN 20261102T000000Z_abcdef0: PASS" in out
    assert "download_results" in out
    downloads = json.loads((run_dir / "downloads.json").read_text())
    assert [d["status"] for d in downloads] == ["not in Colab: use the file on disk"] * 2


def test_gating_step_failure_is_invalid(tmp_path):
    run_dir = make_run(tmp_path, steps=[(2, "finished"), (3, "failed")])
    verdict, archives = delivery.finalize_run(run_dir, {"DRIVE_MIRROR": False}, download=False)
    assert verdict["status"] == "INVALID"
    assert "step 3 (Preflight) raised RuntimeError: boom in 3" in verdict["text"]
    assert names(archives) == ["01_summary.zip"]


def test_exception_outside_marked_steps_is_invalid(tmp_path):
    run_dir = make_run(tmp_path, steps=[])
    verdict, _ = delivery.finalize_run(run_dir, {}, {"step": None, "error": "ValueError: bad"}, download=False)
    assert verdict["status"] == "INVALID"


def test_production_failure_keeps_the_benchmark_verdict(tmp_path):
    steps = [(2, "finished"), (3, "finished"), (4, "finished"), (5, "failed"), (6, "finished")]
    run_dir = make_run(tmp_path, steps=steps, verdict_status="MISSED")
    verdict, _ = delivery.finalize_run(run_dir, download=False)
    assert verdict["status"] == "MISSED"
    assert "Completed with errors: step 5 (Production) raised RuntimeError: boom in 5" in verdict["text"]
    state = json.loads((run_dir / "state.json").read_text())
    assert "Traceback" in next(s for s in state["steps"] if s["step"] == 5)["traceback"]


def test_process_lost_before_the_verdict_is_invalid(tmp_path):
    run_dir = make_run(tmp_path, steps=[(2, "finished"), (3, "started")])
    verdict, archives = delivery.download_results(run_dir, download=False)
    assert verdict["status"] == "INVALID"
    assert verdict["text"] == "INVALID: process lost during step 3"
    assert json.loads((run_dir / "run_status.json").read_text())["by"] == "download_results"
    assert names(archives) == ["01_summary.zip"]


def test_process_lost_after_the_verdict_keeps_it_once(tmp_path):
    steps = [(2, "finished"), (3, "finished"), (4, "finished"), (6, "started")]
    run_dir = make_run(tmp_path, steps=steps, verdict_status="PASS")
    first, _ = delivery.download_results(run_dir, download=False)
    second, _ = delivery.download_results(run_dir, download=False)
    assert first["status"] == second["status"] == "PASS"
    assert second["text"].count("process lost during step 6") == 1


def test_archive_of_an_unfinished_step_is_partial(tmp_path):
    run_dir = make_run(tmp_path, steps=[(2, "finished"), (3, "finished"), (4, "started")], verdict_status=None)
    folder = run_dir / "benchmark" / "2k"
    folder.mkdir(parents=True)
    (folder / "exp50v6_2k.dng").write_bytes(b"\0" * 10)
    verdict, archives = delivery.download_results(run_dir, download=False)
    assert verdict["status"] == "INVALID"
    assert names(archives) == ["01_summary.zip", "02_benchmark_2k_partial.zip"]


def test_download_all_adds_the_everything_archive_without_inputs(tmp_path):
    steps = [(s, "finished") for s in (2, 3, 4, 6)]
    run_dir = make_run(tmp_path, steps=steps, verdict_status="PASS", settings={"DOWNLOAD_ALL": True})
    (run_dir / "inputs").mkdir()
    (run_dir / "inputs" / "reference.arw").write_bytes(b"\0" * 100)
    (run_dir / "inputs" / "setup.json").write_text("{}")
    _, archives = delivery.finalize_run(run_dir, download=False)
    assert names(archives)[-1] == "07_all.zip"
    with zipfile.ZipFile(archives[-1]) as everything:
        listed = everything.namelist()
    assert "inputs/setup.json" in listed and "inputs/reference.arw" not in listed


def test_checkpoint_never_raises(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path, steps=[(2, "finished")])
    monkeypatch.setattr(delivery, "build_summary", lambda _: (_ for _ in ()).throw(OSError("disk full")))
    delivery.checkpoint(run_dir)
    assert "checkpoint failed: OSError: disk full" in (run_dir / "run.log").read_text()


@pytest.mark.parametrize("clone", [delivery.CLONE_DIR])
def test_recovery_line_imports_from_the_clone(clone):
    line = delivery.recovery_line("/content/exp50mp_v6/run", clone)
    assert line == (
        f'import sys; sys.path.insert(0, "{clone}"); '
        'from prd_v6.delivery import download_results; download_results("/content/exp50mp_v6/run")'
    )
