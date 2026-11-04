"""Verdicts, split archives, recovery, and a whole tiny run through the driver (steps 3-5)."""

from __future__ import annotations

import json
import stat
import zipfile

import pytest

from prd_v5.dng_validate import ENV_VAR
from prd_v6.install import DNG_VALIDATE_BUILD
from video_v1 import delivery, driver, preflight
from video_v1.tests.conftest import make_clip, needs_ffmpeg

FRAMES = 3
RUN_ID = "20261104T120000Z_test"


def _names(path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run folder after steps 1-2: config, a dng_validate build record, and a stand-in validator."""
    monkeypatch.setenv(ENV_VAR, "")
    monkeypatch.setattr(preflight, "gpu_name", lambda: "NVIDIA A100-SXM4-40GB")
    tool = tmp_path / "bin" / "dng_validate"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\necho 'Validating \"$2\"...'\nexit 0\n")  # what dng_validate -v prints on a pass
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    run = tmp_path / "content" / RUN_ID
    (run / DNG_VALIDATE_BUILD).parent.mkdir(parents=True)
    (run / DNG_VALIDATE_BUILD).write_text(json.dumps({"status": "built", "binary": str(tool)}))
    for step in (1, 2):
        delivery.mark_step_started(run, step)
        delivery.mark_step_finished(run, step)
    return run


@needs_ffmpeg
def test_a_whole_tiny_run_is_complete_and_split_into_archives(run_dir, tmp_path, monkeypatch, capsys):
    pytest.importorskip("rawpy")
    monkeypatch.setattr(delivery, "PART_MAX_BYTES", 16_000)  # two 64x48 frames (about 6.5 kB each) per part
    clip = make_clip(tmp_path / "beach.mkv", FRAMES)
    settings = {"VIDEO": str(clip), "MAX_FRAMES": None, "DRIVE_MIRROR": False}
    driver.run(run_dir, settings, log=lambda message: None)
    verdict, archives = delivery.finalize_run(run_dir, settings, download=False)
    assert verdict["status"] == "COMPLETE", verdict["text"]
    assert verdict["text"].endswith(delivery.NO_ACCURACY_CLAIM)
    assert [a.name for a in archives] == [
        f"{RUN_ID}_01_summary.zip",
        f"{RUN_ID}_02_dng_part1of2.zip",
        f"{RUN_ID}_03_dng_part2of2.zip",
    ]
    summary = _names(archives[0])
    expected_files = ["report.html", "summary.txt", "verdict.json", "convert.json", "check.json", "timings.csv"]
    expected_files += ["check/preview_side_by_side.mp4", "check/stability.png", "dng/beach/beach_manifest.json"]
    for expected in expected_files:
        assert expected in summary, expected
    assert not any(name.endswith(".dng") for name in summary)
    assert not any(name.startswith("inputs/") and not name.endswith(".json") for name in summary)
    part1, part2 = _names(archives[1]), _names(archives[2])
    assert [n for n in part1 if n.endswith(".dng")] == ["dng/beach/beach_000000.dng", "dng/beach/beach_000001.dng"]
    assert [n for n in part2 if n.endswith(".dng")] == ["dng/beach/beach_000002.dng"]
    assert "dng/beach/beach_manifest.json" in part1 and "dng/beach/beach_manifest.json" in part2
    report = (run_dir / "report.html").read_text()
    assert "No accuracy claim" in report and "4K" in report
    assert f"VIDEO v1 RUN {RUN_ID}: COMPLETE" in capsys.readouterr().out


def _converted(run_dir, *, check_passed=True, failures=0):
    for step in (3, 4):
        delivery.mark_step_started(run_dir, step)
        delivery.mark_step_finished(run_dir, step)
    convert = {"frames_source": 3, "frames_converted": 3, "validation_failures": failures, "failed_frames": [],
               "size": [48, 64]}  # fmt: skip
    (run_dir / "convert.json").write_text(json.dumps(convert))
    (run_dir / "check.json").write_text(json.dumps({"passed": check_passed, "metrics": {"level": {"passed": False}}}))


def test_a_stability_failure_is_flagged_not_invalid(run_dir):
    _converted(run_dir, check_passed=False)
    delivery.mark_step_started(run_dir, 5)
    delivery.mark_step_finished(run_dir, 5)
    verdict = delivery.finalize_verdict(run_dir)
    assert verdict["status"] == "FLAGGED" and "stability check failed on level" in verdict["text"]


def test_a_validation_failure_or_a_step_5_error_is_flagged(run_dir):
    _converted(run_dir, failures=1)
    delivery.mark_step_started(run_dir, 5)
    delivery.mark_step_finished(run_dir, 5, RuntimeError("the preview encoder failed (exit 1)"))
    verdict = delivery.finalize_verdict(run_dir)
    assert verdict["status"] == "FLAGGED"
    assert any("failed LibRaw or dng_validate" in r for r in verdict["reasons"])
    assert any("preview encoder" in r for r in verdict["reasons"])


def test_a_gating_failure_is_invalid(run_dir):
    delivery.mark_step_started(run_dir, 3)
    delivery.mark_step_finished(run_dir, 3, preflight.PreflightError("free disk 1.0 GB < 2.0 GB"))
    verdict = delivery.finalize_verdict(run_dir)
    assert verdict["status"] == "INVALID" and "free disk" in verdict["text"]


def test_recovery_after_a_lost_process_packages_a_partial_sequence(run_dir, capsys):
    delivery.mark_step_started(run_dir, 3)
    delivery.mark_step_finished(run_dir, 3)
    delivery.mark_step_started(run_dir, 4)
    frames = run_dir / "dng" / "beach"
    frames.mkdir(parents=True)
    (frames / "beach_000000.dng").write_bytes(b"\0" * 100)
    (frames / "beach_manifest.json").write_text("{}")
    verdict, archives = delivery.download_results(run_dir, download=False)
    assert verdict["status"] == "INVALID" and "process lost during step 4" in verdict["text"]
    assert [a.name for a in archives] == [f"{RUN_ID}_01_summary.zip", f"{RUN_ID}_02_dng_part1of1_partial.zip"]
    again, _ = delivery.download_results(run_dir, download=False)
    assert again["status"] == "INVALID" and again["finalized_at"] == verdict["finalized_at"]
    assert "Recovery, in a new cell" in capsys.readouterr().out


def test_split_parts_keeps_frame_order_and_gives_a_large_frame_its_own_part(tmp_path):
    sizes = [4, 4, 9, 2, 2]
    entries = []
    for index, size in enumerate(sizes):
        path = tmp_path / f"f{index}.dng"
        path.write_bytes(b"\0" * size)
        entries.append((path, path.relative_to(tmp_path)))
    parts = delivery.split_parts(entries, 8)
    assert [[p.name for p, _ in part] for part in parts] == [["f0.dng", "f1.dng"], ["f2.dng"], ["f3.dng", "f4.dng"]]


def test_recovery_line_names_this_package():
    line = delivery.recovery_line("/content/exp50mp_video_v1/x")
    assert "from video_v1.delivery import download_results" in line and '"/content/exp50mp_video_v1/x"' in line
