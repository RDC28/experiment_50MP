"""Tiny paired execution through independent LibRaw development."""

import hashlib
import json

import pytest
from PIL import Image

from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.paired import run_analytic_pair


def test_pair_uses_shared_hypotheses_and_links_completed_bundles(tmp_path):
    low, high = tmp_path / "low.png", tmp_path / "high.png"
    Image.new("RGB", (16, 16), (128, 80, 100)).save(low)
    Image.new("RGB", (32, 32), (128, 80, 100)).save(high)
    config = AnalyticRunConfig(
        camera=CameraHypothesis.sample(951),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        strip_rows=3,
    )
    output = tmp_path / "pair"
    report = run_analytic_pair(low, high, output, config)
    assert json.loads((output / "pair_manifest.json").read_text()) == report
    assert report["comparison"]["status"] == "not measured"
    assert report["runs"]["2k"]["source_dimensions"] == [16, 16]
    assert report["runs"]["4k"]["source_dimensions"] == [32, 32]
    for label in ("2k", "4k"):
        run = report["runs"][label]
        manifest = output / run["manifest"]
        assert run["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        child = json.loads(manifest.read_text())
        assert child["camera"] == report["shared_camera"]
        assert child["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
        assert json.loads((output / label / "config.json").read_text()) == report["shared_config"]
        assert not list((output / label).glob("working-*"))
    assert (output / "2k/capture.dng").read_bytes() == (output / "4k/capture.dng").read_bytes()
    with pytest.raises(FileExistsError):
        run_analytic_pair(low, high, output, config)


def test_failed_second_run_retains_first_without_pair_completion(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32)).save(source)
    config = AnalyticRunConfig(target_megapixels=0.001024, assume_unprofiled_srgb=True)
    original = AnalyticRunConfig.run
    calls = []

    def fail_second(self, path, output):
        assert self is config
        calls.append(path)
        if len(calls) == 2:
            assert (output.parent / "2k/manifest.json").is_file()
            assert not list((output.parent / "2k").glob("working-*"))
            raise RuntimeError("second source failed")
        return original(self, path, output)

    monkeypatch.setattr(AnalyticRunConfig, "run", fail_second)
    output = tmp_path / "pair"
    with pytest.raises(RuntimeError, match="second source failed"):
        run_analytic_pair(source, source, output, config)
    assert len(calls) == 2
    assert not (output / "pair_manifest.json").exists()
    assert (output / "2k/manifest.json").is_file()
