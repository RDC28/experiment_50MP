"""Driver step 3: runtime tier, tools, the clip's location, its probe, and free disk."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from prd_v5.dng_validate import ENV_VAR
from prd_v6.install import DNG_VALIDATE_BUILD
from video_v1 import preflight
from video_v1.tests.conftest import TINY_HW, make_clip, needs_ffmpeg


def test_classify_warns_without_a_gpu_and_stops_on_low_ram():
    problems, warnings = preflight.classify(None, 83.0)
    assert not problems and "CPU" in warnings[0]
    assert preflight.classify("NVIDIA A100-SXM4-40GB", 83.0) == ([], [])
    assert preflight.classify("NVIDIA L4", 53.0) == ([], [])
    problems, warnings = preflight.classify("Tesla T4", 4.0)
    assert problems and warnings


def test_disk_estimate_covers_the_outputs_and_their_zips():
    pixels = 1080 * 1920
    one_frame = pixels * 2 + pixels * 6 + pixels * 14 // 8  # DNG, uncompressed TIFF, 14-bit MLV
    need = preflight.disk_needed_gb((1080, 1920), 150)
    assert need > one_frame * 150 * 2 / 1e9 + preflight.DISK_RESERVE_GB
    assert preflight.disk_needed_gb((1080, 1920), 150, write_dng=False) < need


def test_video_dest_names():
    inputs = Path("/r/inputs")
    assert preflight.video_dest(inputs, "https://example.org/a/My%20clip.mp4?x=1").name == "My_clip.mp4"
    drive = "https://drive.google.com/file/d/1AbC_d-E/view?usp=sharing"
    assert preflight.video_dest(inputs, drive).name == "source_video"


def test_resolve_video_sources(tmp_path):
    with pytest.raises(preflight.PreflightError, match="no file was uploaded"):
        preflight.resolve_video(tmp_path, None)
    clip = tmp_path / "inputs" / "clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"x" * 10)
    (tmp_path / preflight.UPLOAD_RECORD).write_text(json.dumps({"path": str(clip), "name": "clip.mp4", "bytes": 10}))
    assert preflight.resolve_video(tmp_path, None) == (clip, {"how": "uploaded", "path": str(clip), "name": "clip.mp4",
                                                               "bytes": 10})  # fmt: skip
    assert preflight.resolve_video(tmp_path, str(clip))[1]["how"] == "runtime path"
    with pytest.raises(preflight.PreflightError, match="Drive mounted"):
        preflight.resolve_video(tmp_path, "/content/drive/MyDrive/missing.mp4")


@pytest.fixture
def setup_done(tmp_path, monkeypatch):
    """A run folder as step 2 leaves it: a dng_validate build record pointing at an executable."""
    monkeypatch.setenv(ENV_VAR, "")  # preflight sets it; monkeypatch restores it
    monkeypatch.setattr(preflight, "gpu_name", lambda: "NVIDIA L4")
    tool = tmp_path / "bin" / "dng_validate"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    build = tmp_path / "run" / DNG_VALIDATE_BUILD
    build.parent.mkdir(parents=True)
    build.write_text(json.dumps({"status": "built", "binary": str(tool)}))
    return tmp_path / "run"


@needs_ffmpeg
def test_preflight_probes_the_clip_and_records_it(setup_done, tmp_path):
    clip = make_clip(tmp_path / "clip.mkv", 3)
    inputs = preflight.run(setup_done, {"VIDEO": str(clip)}, log=lambda message: None)
    assert inputs["clip"].frames == 3 and inputs["dng_validate"].endswith("dng_validate")
    record = json.loads((setup_done / "source.json").read_text())
    assert record["frames"] == 3 and record["output_size"] == list(TINY_HW)
    assert record["origin"]["how"] == "runtime path" and record["audio"] == "none in the source"
    environment = json.loads((setup_done / "environment.json").read_text())
    assert environment["tier"]["gpu"] == "NVIDIA L4" and environment["disk"]["frames"] == 3


@needs_ffmpeg
def test_preflight_stops_on_low_disk(setup_done, tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "DISK_RESERVE_GB", 1e9)
    clip = make_clip(tmp_path / "clip.mkv", 2)
    with pytest.raises(preflight.PreflightError, match="free disk"):
        preflight.run(setup_done, {"VIDEO": str(clip)}, log=lambda message: None)


@needs_ffmpeg
def test_preflight_needs_dng_validate(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "gpu_name", lambda: None)
    with pytest.raises(preflight.PreflightError, match="dng_validate"):
        preflight.run(tmp_path, {"VIDEO": None}, log=lambda message: None)
