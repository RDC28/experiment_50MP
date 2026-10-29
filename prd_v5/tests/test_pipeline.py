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
        "sensor_defects.npy",
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
    assert report["runtime"]["libraries"]["numpy"] == np.__version__
    assert report["runtime"]["libraries"]["libraw"] in page
    assert "Runtime and replay limits" in page
    assert report["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
    assert report["stages"]["analytic"]["forward_consistency"]["status"] == "passed"
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


def test_selected_input_provenance_survives_bundle(tmp_path):
    from prd_v5.analytic import AnalyticISP
    from prd_v5.config import AnalyticRunConfig
    from prd_v5.fields import resolve_camera, resolve_isp
    from prd_v5.hypothesis import CameraField

    def evidence(name, value, units, mode, provenance):
        return CameraField(name, value, units, mode, provenance, None, None, "Supplied test evidence", "Uncalibrated")

    camera, camera_records = resolve_camera(
        CameraHypothesis(), {"iso": "USER"}, {"iso": [evidence("iso", 400, "ISO", "USER", "USER_SET")]}
    )
    isp, isp_records = resolve_isp(
        AnalyticISP(),
        {"exposure_gain": "ESTIMATE"},
        {"exposure_gain": [evidence("exposure_gain", 2, "ratio", "ESTIMATE", "ESTIMATED")]},
    )
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    config = AnalyticRunConfig(camera=camera, isp=isp, target_megapixels=0.001024, assume_unprofiled_srgb=True)
    output = tmp_path / "selected"
    report = config.run(source, output, camera_selections=camera_records, isp_selections=isp_records)
    saved = json.loads((output / "manifest.json").read_text())
    for registry in (report["registry"], saved["registry"]):
        assert registry["iso"]["mode"] == "USER"
        assert registry["iso"]["provenance"] == "USER_SET"
        assert registry["iso"]["value"] == report["camera"]["iso"] == 400
        assert registry["isp.exposure_gain"]["name"] == "isp.exposure_gain"
        assert registry["isp.exposure_gain"]["provenance"] == "ESTIMATED"
        assert registry["isp.exposure_gain"]["value"] == report["isp"]["exposure_gain"] == 2
        assert registry["analogue_gain"]["provenance"] == "DERIVED"
    page = (output / "report.html").read_text()
    assert "USER_SET" in page and "ESTIMATED" in page and "Supplied test evidence" in page
    assert AnalyticRunConfig.from_json((output / "config.json").read_text()) == config


@pytest.mark.parametrize(
    ("name", "value", "units", "mode", "provenance"),
    [
        ("iso", 400, "ISO", "USER", "USER_SET"),  # effective default is 100
        ("iso", 100, "ISO", "USER", "SYNTHETIC"),
        ("iso", 100, "EV", "AUTO", "SYNTHETIC"),
        ("gps", [1, 2], None, "USER", "USER_SET"),
        ("analogue_gain", 1, "ratio", "AUTO", "DERIVED"),
    ],
)
def test_selection_claims_rejected_before_io(tmp_path, name, value, units, mode, provenance):
    from prd_v5.hypothesis import CameraField

    record = CameraField(name, value, units, mode, provenance, None, None, "Claim", "Unchecked")
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        run_analytic(tmp_path / "missing.png", output, CameraHypothesis(), camera_selections=(record,))
    assert not output.exists()


def test_duplicate_and_mismatched_isp_selections_rejected_before_io(tmp_path):
    from prd_v5.hypothesis import CameraField

    record = CameraField("exposure_gain", 2, "ratio", "USER", "USER_SET", None, None, "Claim", "Unchecked")
    for records in ((record,), (record, record)):
        with pytest.raises(ValueError):
            run_analytic(tmp_path / "missing.png", tmp_path / "out", CameraHypothesis(), isp_selections=records)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("hot_fraction,dead_fraction", [(0, 0), (1, 1), (0.4, 0.3)])
def test_bundle_defect_mask_matches_seeded_sensor_and_replay(tmp_path, hot_fraction, dead_fraction):
    from dataclasses import replace

    from prd_v5.config import AnalyticRunConfig
    from prd_v5.sensor import SensorEffects

    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    config = AnalyticRunConfig(
        sensor_effects=SensorEffects(hot_pixel_fraction=hot_fraction, dead_pixel_fraction=dead_fraction),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        strip_rows=1,
    )
    first = config.run(source, tmp_path / "first")
    second = replace(config, strip_rows=7).run(source, tmp_path / "second")
    import rawpy
    import tifffile

    assert first["artifacts"]["sensor_defects.npy"] == second["artifacts"]["sensor_defects.npy"]
    with rawpy.imread(str(tmp_path / "first/capture.dng")) as a:
        with rawpy.imread(str(tmp_path / "second/capture.dng")) as b:
            np.testing.assert_array_equal(a.raw_image, b.raw_image)
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "first/developed.tiff"), tifffile.imread(tmp_path / "second/developed.tiff")
    )
    seed = config.camera.stream_seed("sensor_body")
    mask = np.load(tmp_path / "first/sensor_defects.npy", mmap_mode="r", allow_pickle=False)
    try:
        assert mask.dtype == np.uint8 and mask.shape == (32, 32)
        for y in range(32):
            hot = np.random.default_rng(np.random.SeedSequence([seed, 5, y])).random(32) < hot_fraction
            dead = np.random.default_rng(np.random.SeedSequence([seed, 6, y])).random(32) < dead_fraction
            np.testing.assert_array_equal(mask[y], hot.astype(np.uint8) | (dead.astype(np.uint8) << 1))
        evidence = first["masks"]["sensor_defects"]
        assert evidence["occupancy"]["hot_selected"] == np.count_nonzero(mask & 1)
        assert evidence["occupancy"]["dead_selected"] == np.count_nonzero(mask & 2)
        assert evidence["occupancy"]["hot_and_dead"] == np.count_nonzero(mask == 3)
    finally:
        mask._mmap.close()
    assert first["stages"]["sensor"]["body_effects"]["defect_mask"]["written"] is True
    saved = json.loads((tmp_path / "first/manifest.json").read_text())
    assert saved["masks"]["sensor_defects"] == first["masks"]["sensor_defects"]
    path = tmp_path / "first/sensor_defects.npy"
    assert first["artifacts"][path.name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "sensor_defects.npy" in (tmp_path / "first/report.html").read_text()
