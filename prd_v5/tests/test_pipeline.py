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
        "source_clipped.npy",
        "manifest.json",
        "config.json",
        "report.html",
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
    page = (output / "report.html").read_text()
    assert "Scientific acceptance: not measured" in page
    assert report["artifacts"]["capture.dng"]["sha256"] in page
    assert "source_clipped.npy" in page
    assert "passed" in page
    assert report["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
    assert report["registry"]["gps"]["value"] is None
    registry = report["registry"]
    assert registry["source.sha256"]["value"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert registry["source.width"]["provenance"] == "OBSERVED"
    assert registry["source.colour_policy"]["provenance"] == "SYNTHETIC"
    assert registry["isp.camera_to_srgb"]["provenance"] == "SYNTHETIC"
    assert registry["geometry.width"]["value"] == report["geometry"]["width"]
    assert registry["geometry.width"]["provenance"] == "DERIVED"
    for record in registry.values():
        assert record["confidence"] is None
        assert record["posterior"] is None
        assert record["method"] and record["validation"]
    assert "source.path" not in registry
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


@pytest.mark.parametrize("strip_rows", [1, 7])
def test_clipping_mask_retains_endpoint_locations(tmp_path, strip_rows):
    pixels = np.full((32, 32, 3), 128, dtype=np.uint8)
    pixels[4:9, 6:11, 0] = 255
    pixels[20, 15] = 255
    source = tmp_path / "source.png"
    Image.fromarray(pixels).save(source)
    output = tmp_path / "run"
    report = run_analytic(
        source,
        output,
        CameraHypothesis(),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        strip_rows=strip_rows,
    )
    mask = np.load(output / "source_clipped.npy", mmap_mode="r", allow_pickle=False)
    try:
        assert mask.dtype == np.bool_
        np.testing.assert_array_equal(mask, np.any(pixels == 255, axis=-1))
        assert report["masks"]["source_clipped"]["clipped_pixels"] == 26
        assert report["stages"]["analytic"]["source_clipped_pixels"] == 26
    finally:
        mask._mmap.close()


def test_report_failure_has_no_completion_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    output = tmp_path / "run"

    def fail(*args, **kwargs):
        raise OSError("injected report failure")

    monkeypatch.setattr("prd_v5.pipeline.render_report", fail)
    with pytest.raises(OSError, match="report failure"):
        run_analytic(source, output, CameraHypothesis(), target_megapixels=0.001024, assume_unprofiled_srgb=True)
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("working-*"))


def test_saved_configuration_replays_exact_adc(tmp_path):
    from prd_v5.analytic import AnalyticISP
    from prd_v5.config import AnalyticRunConfig

    source = tmp_path / "source.png"
    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    Image.fromarray(pixels).save(source)
    config = AnalyticRunConfig(
        camera=CameraHypothesis(seed=938, iso=400),
        isp=AnalyticISP(exposure_gain=1.3),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        strip_rows=5,
        max_decode_bytes=100_000,
        max_develop_bytes=100_000,
    )
    first = config.run(source, tmp_path / "first")
    saved = AnalyticRunConfig.from_json((tmp_path / "first/config.json").read_text())
    assert saved == config
    second = saved.run(source, tmp_path / "second")
    for name in ("capture.dng", "developed.tiff", "source_support.npy", "source_clipped.npy", "config.json"):
        assert first["artifacts"][name] == second["artifacts"][name]
    assert "config.json" in (tmp_path / "first/report.html").read_text()
