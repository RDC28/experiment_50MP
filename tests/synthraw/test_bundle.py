import json
import sys
import types
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from synthraw.bundle import (
    FINAL_IMAGE_FILES,
    FINAL_RUN_FILES,
    bundle_run,
    copy_run_to_drive,
    deliver_bundle,
    deliver_final_image,
    drive_run_folder,
    sync_to_drive,
)
from synthraw.pipeline import open_execution_log
from synthraw.run_layout import RunLayout


@pytest.fixture
def run(tmp_path):
    layout = RunLayout.create(tmp_path / "runs", timestamp=datetime(2026, 1, 2, 3, 4, 5))
    np.save(layout.directory("raw") / "sensor_mosaic_uint16.npy", np.zeros((2, 2), np.uint16))
    np.save(layout.directory("intermediates") / "04_hdr_radiance.npy", np.zeros((2, 2, 3), np.float32))
    np.save(layout.directory("confidence") / "hdr_confidence.npy", np.ones((2, 2), np.float32))
    (layout.directory("confidence") / "notes.json").write_text("{}", "utf-8")
    (layout.directory("metadata") / "provenance.json").write_text("{}", "utf-8")
    return layout


def _names(bundle, root):
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        return {name.removeprefix(f"{root.name}/") for name in archive.namelist()}


def test_bundle_includes_whole_tree_by_default(run):
    bundle = bundle_run(run.root)
    assert bundle == run.root.with_name("run_20260102_030405.zip")
    assert _names(bundle, run.root) == {
        "raw/sensor_mosaic_uint16.npy",
        "intermediates/04_hdr_radiance.npy",
        "confidence/hdr_confidence.npy",
        "confidence/notes.json",
        "metadata/provenance.json",
        "logs/bundle_manifest.json",
    }
    manifest = json.loads((run.directory("logs") / "bundle_manifest.json").read_text())
    assert manifest["omitted_large_arrays"] == []
    assert not list(bundle.parent.glob(".*.partial"))


def test_bundle_toggle_omits_only_large_arrays(run, tmp_path):
    bundle = bundle_run(run.root, exclude_large_arrays=True, destination=tmp_path / "out.zip")
    names = _names(bundle, run.root)
    # The RAW mosaic is a required PRD deliverable, never a droppable intermediate.
    assert "raw/sensor_mosaic_uint16.npy" in names
    assert "confidence/notes.json" in names
    assert not {"intermediates/04_hdr_radiance.npy", "confidence/hdr_confidence.npy"} & names
    manifest = json.loads((run.directory("logs") / "bundle_manifest.json").read_text())
    assert manifest["omitted_large_arrays"] == ["confidence/hdr_confidence.npy", "intermediates/04_hdr_radiance.npy"]


def test_bundle_rejects_non_run_folders_and_inner_destinations(run, tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a run folder"):
        bundle_run(tmp_path)
    with pytest.raises(ValueError, match="outside the run folder"):
        bundle_run(run.root, destination=run.directory("logs") / "self.zip")


def test_failed_bundle_leaves_no_partial_zip(run, monkeypatch):
    def fail(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail)
    with pytest.raises(OSError, match="disk full"):
        bundle_run(run.root)
    assert sorted(path.name for path in run.root.parent.iterdir()) == [run.root.name]


def test_delivery_without_colab_is_reported(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    bundle = tmp_path / "run.zip"
    bundle.write_bytes(b"zip")
    assert deliver_bundle(bundle)["download"] == "unavailable"


def test_delivery_offers_download(tmp_path, monkeypatch):
    downloads = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    bundle = tmp_path / "run.zip"
    bundle.write_bytes(b"zip")
    status = deliver_bundle(bundle)
    assert downloads == [str(bundle)]
    assert status["download"] == "offered"


def test_large_zip_skips_browser_download_and_drive_keeps_the_full_run(run, tmp_path, monkeypatch):
    downloads = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    drive = tmp_path / "drive"
    drive.mkdir()
    (run.directory("logs") / "execution.log").write_text("stage done\n", "utf-8")
    assert copy_run_to_drive(run.root, drive) == (drive / run.root.name, None)
    # Written after the run folder copy: the report, the manifest and the end of the log.
    (run.directory("validation") / "run_report.json").write_text('{"status": "SUCCESS"}', "utf-8")
    bundle = bundle_run(run.root, exclude_large_arrays=True)
    with (run.directory("logs") / "execution.log").open("a", encoding="utf-8") as log:
        log.write("Bundle written\n")
    assert sync_to_drive(run.root, drive, bundle) is None
    size = bundle.stat().st_size
    status = deliver_bundle(bundle, max_download_bytes=size - 1)
    assert downloads == [] and status["download"].startswith("skipped")
    assert (drive / bundle.name).read_bytes() == bundle.read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        bundled = set(archive.namelist())
    copied = {path.relative_to(drive / run.root.name) for path in (drive / run.root.name).rglob("*") if path.is_file()}
    assert copied == {path.relative_to(run.root) for path in run.root.rglob("*") if path.is_file()}
    for name in FINAL_RUN_FILES:
        assert (drive / run.root.name / name).read_bytes() == (run.root / name).read_bytes()
    assert any(f"{run.root.name}/{name.as_posix()}" not in bundled for name in copied)
    assert deliver_bundle(bundle, max_download_bytes=size)["download"] == "offered"
    assert downloads == [str(bundle)]


def _colab_drive(monkeypatch, mount):
    calls = []

    def record(path):
        calls.append(path)
        mount(Path(path))

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=record)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return calls


def test_drive_is_mounted_for_the_intermediates_when_not_mounted(tmp_path, monkeypatch):
    calls = _colab_drive(monkeypatch, lambda path: (path / "MyDrive").mkdir(parents=True))
    folder, warning = drive_run_folder(tmp_path / "drive", mount=True)
    assert calls == [str(tmp_path / "drive")] and warning is None
    assert folder == tmp_path / "drive" / "MyDrive" / "synthraw_runs" and folder.is_dir()


def test_already_mounted_drive_is_used_without_a_prompt(tmp_path, monkeypatch):
    (tmp_path / "drive" / "MyDrive").mkdir(parents=True)
    calls = _colab_drive(monkeypatch, lambda path: pytest.fail("Drive is already mounted"))
    for mount in (True, False):
        folder, warning = drive_run_folder(tmp_path / "drive", mount=mount)
        assert folder == tmp_path / "drive" / "MyDrive" / "synthraw_runs" and warning is None
    assert calls == []


def test_declined_drive_mount_warns_and_never_raises(tmp_path, monkeypatch):
    def decline(path):
        raise ValueError("mount failed: credential propagation was unsuccessful")

    calls = _colab_drive(monkeypatch, decline)
    folder, warning = drive_run_folder(tmp_path / "drive", mount=True)
    assert folder is None and len(calls) == 1
    assert "credential propagation" in warning and "exist only on this Colab runtime" in warning
    monkeypatch.setitem(sys.modules, "google.colab", None)
    folder, warning = drive_run_folder(tmp_path / "drive", mount=True)
    assert folder is None and warning.startswith("Google Drive was not mounted (ModuleNotFoundError")
    assert drive_run_folder(tmp_path / "drive", mount=False) == (None, None)


def _no_space(*args, **kwargs):
    raise OSError(28, "No space left on device")


def test_failed_drive_run_copy_returns_a_warning(run, tmp_path, monkeypatch):
    monkeypatch.setattr("synthraw.bundle.shutil.copytree", _no_space)
    folder, warning = copy_run_to_drive(run.root, tmp_path)
    assert folder is None
    assert "No space left on device" in warning and "exist only on this Colab runtime" in warning


def test_failed_final_drive_sync_returns_a_warning(run, tmp_path, monkeypatch):
    (run.directory("validation") / "run_report.json").write_text("{}", "utf-8")
    (run.directory("logs") / "execution.log").write_text("log\n", "utf-8")
    bundle = bundle_run(run.root, exclude_large_arrays=True)
    monkeypatch.setattr("synthraw.bundle.shutil.copy2", _no_space)
    warning = sync_to_drive(run.root, tmp_path / "drive", bundle)
    assert "No space left on device" in warning and bundle.name in warning


def test_execution_log_rebinding_closes_previous_run(tmp_path):
    first = RunLayout.create(tmp_path, timestamp=datetime(2026, 1, 1))
    second = RunLayout.create(tmp_path, timestamp=datetime(2026, 1, 2))
    open_execution_log(first, "synthraw.test_rebind").info("first")
    logger = open_execution_log(second, "synthraw.test_rebind")
    logger.info("second")
    assert len(logger.handlers) == 1
    assert "first" in (first.directory("logs") / "execution.log").read_text()
    assert "second" not in (first.directory("logs") / "execution.log").read_text()
    assert "second" in (second.directory("logs") / "execution.log").read_text()
    logger.handlers[0].close()


def _final_image(run):
    run.directory("masters").joinpath("developed_50mp_16bit.tif").write_bytes(b"tiff" * 10)
    run.directory("raw").joinpath("synthetic_50mp.dng").write_bytes(b"dng" * 4)
    return {kind: run.root / name for kind, name in FINAL_IMAGE_FILES.items()}


def _colab_files(monkeypatch):
    downloads = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return downloads


def test_final_image_is_offered_for_download_and_kept_on_drive(run, tmp_path, monkeypatch):
    """Regression: the run left the final image only inside the zip."""
    files = _final_image(run)
    downloads = _colab_files(monkeypatch)
    drive = tmp_path / "drive"
    messages = []
    result = deliver_final_image(run.root, drive_dir=drive, log=messages.append)
    assert downloads == [str(files["tiff16"]), str(files["dng"])]
    assert result["downloads"] == {"tiff16": "offered", "dng": "offered"}
    assert result["files"] == {kind: str(path) for kind, path in files.items()}
    for kind, path in files.items():
        copied = drive / run.root.name / path.relative_to(run.root)
        assert result["drive"][kind] == str(copied) and copied.read_bytes() == path.read_bytes()
        assert any(str(path) in message for message in messages)
        assert any(str(copied) in message for message in messages)
    assert result["missing"] == [] and result["warnings"] == []


def test_final_image_already_on_drive_is_not_copied_again(run, tmp_path, monkeypatch):
    _final_image(run)
    drive = tmp_path / "drive"
    copy_run_to_drive(run.root, drive)
    monkeypatch.setattr("synthraw.bundle.shutil.copy2", lambda *args: pytest.fail("already on Drive"))
    result = deliver_final_image(run.root, drive_dir=drive, download=False, log=lambda message: None)
    assert set(result["drive"]) == {"tiff16", "dng"} and result["downloads"] == {}


def test_final_image_outside_colab_prints_paths_only(run, monkeypatch):
    files = _final_image(run)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    messages = []
    result = deliver_final_image(run.root, log=messages.append)
    assert result["downloads"] == {} and result["drive"] == {}
    assert all(any(str(path) in message for message in messages) for path in files.values())
    assert any("Not in Colab" in message for message in messages)


def test_final_image_reports_missing_files_large_downloads_and_drive_failures(run, tmp_path, monkeypatch):
    run.directory("masters").joinpath("developed_50mp_16bit.tif").write_bytes(b"x" * 100)
    downloads = _colab_files(monkeypatch)
    monkeypatch.setattr("synthraw.bundle.shutil.copy2", _no_space)
    messages = []
    result = deliver_final_image(run.root, drive_dir=tmp_path / "drive", max_download_bytes=99, log=messages.append)
    assert result["missing"] == ["raw/synthetic_50mp.dng"]
    assert downloads == [] and result["downloads"]["tiff16"].startswith("skipped")
    assert result["drive"] == {} and "No space left on device" in result["warnings"][0]
    assert any(message.startswith("WARNING: ") for message in messages)
