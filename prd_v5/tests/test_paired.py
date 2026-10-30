"""Tiny paired execution through independent LibRaw development."""

import hashlib
import json

import pytest
from PIL import Image

from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.paired import run_analytic_pair
from prd_v5.panel import resolve_camera_panel


def test_pair_uses_shared_hypotheses_and_links_completed_bundles(tmp_path):
    low, high = tmp_path / "low.png", tmp_path / "high.png"
    Image.new("RGB", (16, 16), (128, 80, 100)).save(low)
    Image.new("RGB", (32, 32), (128, 80, 100)).save(high)
    camera, selections = resolve_camera_panel(
        CameraHypothesis.sample(951), [{"name": "ISO", "mode": "AUTO", "value": 400}], {"ISO": "USER"}
    )
    config = AnalyticRunConfig(
        camera=camera,
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        strip_rows=3,
    )
    output = tmp_path / "pair"
    report = run_analytic_pair(low, high, output, config, camera_selections=selections)
    assert json.loads((output / "pair_manifest.json").read_text()) == report
    assert report["comparison"]["status"] == "measured render disagreement"
    assert report["comparison"]["identical"] is True
    assert report["comparison"]["rmse"] == 0
    assert report["runs"]["2k"]["source_dimensions"] == [16, 16]
    assert report["runs"]["4k"]["source_dimensions"] == [32, 32]
    for label in ("2k", "4k"):
        run = report["runs"][label]
        manifest = output / run["manifest"]
        assert run["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        child = json.loads(manifest.read_text())
        assert child["camera"] == report["shared_camera"]
        assert child["registry"]["iso"]["mode"] == "USER"
        assert child["registry"]["iso"]["provenance"] == "USER_SET"
        assert child["camera"]["iso"] == 400
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

    def fail_second(self, path, output, **kwargs):
        assert self is config
        calls.append(path)
        if len(calls) == 2:
            assert (output.parent / "2k/manifest.json").is_file()
            assert not list((output.parent / "2k").glob("working-*"))
            raise RuntimeError("second source failed")
        return original(self, path, output, **kwargs)

    monkeypatch.setattr(AnalyticRunConfig, "run", fail_second)
    output = tmp_path / "pair"
    with pytest.raises(RuntimeError, match="second source failed"):
        run_analytic_pair(source, source, output, config)
    assert len(calls) == 2
    assert not (output / "pair_manifest.json").exists()
    assert (output / "2k/manifest.json").is_file()


def test_pair_rejects_conflicting_selection_before_creating_output(tmp_path):
    _, selections = resolve_camera_panel(
        CameraHypothesis(), [{"name": "ISO", "mode": "AUTO", "value": 400}], {"ISO": "USER"}
    )
    output = tmp_path / "pair"
    with pytest.raises(ValueError, match="disagree"):
        run_analytic_pair("missing2k", "missing4k", output, AnalyticRunConfig(), camera_selections=selections)
    assert not output.exists()


def test_pair_reports_unequal_rasters_without_metrics(tmp_path):
    low, high = tmp_path / "low.png", tmp_path / "high.png"
    Image.new("RGB", (32, 32), (90, 80, 100)).save(low)
    Image.new("RGB", (48, 32), (90, 80, 100)).save(high)
    config = AnalyticRunConfig(target_megapixels=0.001024, assume_unprofiled_srgb=True)
    report = run_analytic_pair(low, high, tmp_path / "pair", config)
    assert report["comparison"]["status"] == "not measured"
    assert "Unequal raster" in report["comparison"]["reason"]
    assert "rmse" not in report["comparison"]
    assert (tmp_path / "pair/pair_manifest.json").exists()


def test_comparison_failure_cannot_publish_pair_completion(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (90, 80, 100)).save(source)
    config = AnalyticRunConfig(target_megapixels=0.001024, assume_unprofiled_srgb=True)

    def fail(*args, **kwargs):
        raise OSError("comparison read failed")

    monkeypatch.setattr("prd_v5.paired.compare_renders", fail)
    output = tmp_path / "pair"
    with pytest.raises(OSError, match="comparison read failed"):
        run_analytic_pair(source, source, output, config)
    assert (output / "2k/manifest.json").exists()
    assert (output / "4k/manifest.json").exists()
    assert not (output / "pair_manifest.json").exists()


def test_pair_retains_whole_raster_and_native_crop_metrics(tmp_path):
    from prd_v5.comparison import compare_renders

    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (90, 80, 100)).save(source)
    config = AnalyticRunConfig(target_megapixels=0.001024, assume_unprofiled_srgb=True, strip_rows=3)
    output = tmp_path / "pair"
    region = (2, 3, 10, 12)
    report = run_analytic_pair(source, source, output, config, comparison_region=region)
    assert report["comparison"]["samples"] == 32 * 32 * 3
    crop = report["crop_comparison"]
    assert crop["region"] == list(region)
    assert crop["samples"] == 8 * 9 * 3
    independent = compare_renders(output / "2k/developed.tiff", output / "4k/developed.tiff", region=region)
    assert {key: value for key, value in crop.items() if key != "limitations"} == independent
    assert json.loads((output / "pair_manifest.json").read_text()) == report


@pytest.mark.parametrize("region", [(0, 0, 0, 2), [0, 0, 2, 2], (False, 0, 2, 2)])
def test_pair_rejects_invalid_crop_before_source_io(tmp_path, region):
    output = tmp_path / "pair"
    with pytest.raises(ValueError, match="region must"):
        run_analytic_pair("missing", "missing", output, AnalyticRunConfig(), comparison_region=region)
    assert not output.exists()


@pytest.mark.parametrize("region", [(0, 0, 33, 32), (0, 0, 32, 33)])
def test_out_of_bounds_crop_fails_before_any_run(tmp_path, region):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (90, 80, 100)).save(source)
    output = tmp_path / "pair"
    config = AnalyticRunConfig(target_megapixels=0.001024, assume_unprofiled_srgb=True)
    with pytest.raises(ValueError, match="exceeds"):
        run_analytic_pair(source, source, output, config, comparison_region=region)
    assert not output.exists()


def test_pair_executes_shared_vignetting_before_independent_roundtrip(tmp_path):
    import numpy as np
    import rawpy

    from prd_v5.optics import Vignetting

    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(source)
    base = dict(target_megapixels=0.001024, assume_unprofiled_srgb=True, strip_rows=3)
    neutral = run_analytic_pair(source, source, tmp_path / "neutral", AnalyticRunConfig(**base))
    config = AnalyticRunConfig(**base, vignetting=Vignetting(0.5), accept_experimental_optics=True)
    report = run_analytic_pair(source, source, tmp_path / "attenuated", config)
    assert report["shared_config"]["vignetting"] == {"corner_strength": 0.5}
    for label in ("2k", "4k"):
        stage = report["runs"][label]["optics"]
        assert stage["status"] == "experimental"
        assert stage["changed_pixels"] == 1024
        assert neutral["runs"][label]["optics"]["status"] == "zero_effect"
        with rawpy.imread(str(tmp_path / "neutral" / label / "capture.dng")) as raw:
            before = raw.raw_image_visible.copy().astype(float)
        with rawpy.imread(str(tmp_path / "attenuated" / label / "capture.dng")) as raw:
            after = raw.raw_image_visible.copy().astype(float)
        black = config.camera.black_level
        assert np.mean(after - black) < np.mean(before - black)
        assert after[0, 0] < after[16, 16]
    assert report["comparison"]["identical"] is True


def test_pair_executes_shared_distortion_before_independent_roundtrip(tmp_path):
    import numpy as np
    import rawpy

    from prd_v5.optics import Distortion

    source = tmp_path / "edge.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 17:] = 180
    Image.fromarray(image).save(source)
    base = dict(target_megapixels=0.001024, assume_unprofiled_srgb=True, strip_rows=3)
    neutral = run_analytic_pair(source, source, tmp_path / "neutral", AnalyticRunConfig(**base))
    config = AnalyticRunConfig(**base, distortion=Distortion(k1=0.2), accept_experimental_optics=True)
    report = run_analytic_pair(source, source, tmp_path / "warped", config)
    assert report["shared_config"]["distortion"] == {"k1": 0.2, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    for label in ("2k", "4k"):
        stage = report["runs"][label]["distortion"]
        assert stage["status"] == "experimental"
        assert stage["changed_pixels"] > 0
        assert neutral["runs"][label]["distortion"]["status"] == "zero_effect"
        with rawpy.imread(str(tmp_path / "neutral" / label / "capture.dng")) as raw:
            before = raw.raw_image_visible.copy()
        with rawpy.imread(str(tmp_path / "warped" / label / "capture.dng")) as raw:
            after = raw.raw_image_visible.copy()
        assert np.count_nonzero(before != after) > 0
        assert (tmp_path / "warped" / label / "developed.tiff").is_file()
    assert report["comparison"]["identical"] is True
