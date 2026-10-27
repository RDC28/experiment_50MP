import json
import sys
import types
import zipfile
from datetime import datetime

import numpy as np
import pytest

from synthraw.bundle import bundle_run, deliver_bundle
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


def test_delivery_without_colab_or_drive_is_reported(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    bundle = tmp_path / "run.zip"
    bundle.write_bytes(b"zip")
    status = deliver_bundle(bundle, drive_dir=tmp_path / "unmounted")
    assert status["download"] == "unavailable"
    assert status["drive_copy"].startswith("skipped")
    assert deliver_bundle(bundle)["drive_copy"] is None


def test_delivery_offers_download_and_copies_to_drive(tmp_path, monkeypatch):
    downloads = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    bundle = tmp_path / "run.zip"
    bundle.write_bytes(b"zip")
    drive = tmp_path / "drive"
    drive.mkdir()
    status = deliver_bundle(bundle, drive_dir=drive)
    assert downloads == [str(bundle)]
    assert status["download"] == "offered"
    assert (drive / "run.zip").read_bytes() == b"zip"


def test_large_zip_skips_browser_download_and_drive_keeps_the_full_run(run, tmp_path, monkeypatch):
    downloads = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    bundle = bundle_run(run.root, exclude_large_arrays=True)
    drive = tmp_path / "drive"
    drive.mkdir()
    size = bundle.stat().st_size
    status = deliver_bundle(bundle, drive_dir=drive, run_root=run.root, max_download_bytes=size - 1)
    assert downloads == [] and status["download"].startswith("skipped")
    assert (drive / bundle.name).read_bytes() == bundle.read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        bundled = set(archive.namelist())
    copied = {path.relative_to(drive / run.root.name) for path in (drive / run.root.name).rglob("*") if path.is_file()}
    assert copied == {path.relative_to(run.root) for path in run.root.rglob("*") if path.is_file()}
    assert any(f"{run.root.name}/{name.as_posix()}" not in bundled for name in copied)
    assert deliver_bundle(bundle, drive_dir=drive, run_root=run.root, max_download_bytes=size)["download"] == "offered"
    assert downloads == [str(bundle)]


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
