"""Real tiny source-to-independent-TIFF orchestration checks."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from prd_v5.hypothesis import CameraHypothesis
from prd_v5.pipeline import run_analytic


def test_complete_bundle(tmp_path):
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    output = tmp_path / "run"
    report = run_analytic(
        source, output, CameraHypothesis(), target_megapixels=0.001024, assume_unprofiled_srgb=True, strip_rows=3
    )
    assert json.loads((output / "manifest.json").read_text()) == json.loads(json.dumps(report))
    assert set(p.name for p in output.iterdir()) == {
        "capture.dng",
        "developed.tiff",
        "source_support.npy",
        "manifest.json",
    }
    support = np.load(output / "source_support.npy", mmap_mode="r", allow_pickle=False)
    try:
        assert support.dtype == np.bool_
        assert list(support.shape) == report["masks"]["source_support"]["shape"]
        assert support.all()
        assert int(support.sum()) == report["masks"]["source_support"]["supported_pixels"]
    finally:
        support._mmap.close()
    for name, record in report["artifacts"].items():
        assert record["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()
    assert report["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
    assert report["registry"]["gps"]["value"] is None
    with pytest.raises(FileExistsError):
        run_analytic(source, output, CameraHypothesis(), target_megapixels=0.001024)


def test_admission_and_failure_cleanup(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(source)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="memory budget"):
        run_analytic(source, output, CameraHypothesis(), max_develop_bytes=1)
    assert not output.exists()

    def fail(*args, **kwargs):
        raise RuntimeError("injected sensor failure")

    monkeypatch.setattr("prd_v5.pipeline.BayerSensor.process", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run_analytic(source, output, CameraHypothesis(), target_megapixels=0.001024, assume_unprofiled_srgb=True)
    assert list(output.iterdir()) == []


def test_mask_export_failure_has_no_completion_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    output = tmp_path / "run"

    def fail(*args, **kwargs):
        raise OSError("injected mask export failure")

    monkeypatch.setattr("prd_v5.pipeline.np.lib.format.open_memmap", fail)
    with pytest.raises(OSError, match="mask export"):
        run_analytic(source, output, CameraHypothesis(), target_megapixels=0.001024, assume_unprofiled_srgb=True)
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("working-*"))
