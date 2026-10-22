import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import exporters, test_images
from synthetic_raw_lab import sensor_config as sc
from synthetic_raw_lab import sensor_research as sr
from synthetic_raw_lab import sensor_simulation as ss

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "sensor_research.yaml"


def master(width=192, height=108) -> np.ndarray:
    linear = cm.srgb_to_linear(test_images.make_test_image(width, height).astype(np.float32) / 255)
    linear[0, 0] = 1.4  # interpolation overshoot, as in a real Lanczos master
    return linear


def small_config(**overrides) -> sc.SensorResearchConfig:
    values = {"crop_size": 48, "crop_count": 2, "review_max_edge": 256, "metric_tile_size": 128}
    return sc.load_sensor_config(CONFIG_FILE, **{**values, **overrides})


def test_yaml_config_matches_documented_defaults():
    config = sc.load_sensor_config(CONFIG_FILE)
    defaults = sc.SensorResearchConfig()
    differing = {k for k, v in config.to_dict().items() if v != defaults.to_dict()[k]}
    assert differing == {"export_synthetic_dng"}  # PRD 6.1 keeps it off by default; the research config enables it
    model = config.sensor_model()
    assert (model.black, model.white) == (512, 16383)
    assert model.color_matrix == ss.DEFAULT_COLOR_MATRIX


def test_parameter_table_documents_every_parameter():
    rows = sc.parameter_table(sc.SensorResearchConfig(analog_gain=4.0))
    assert len(rows) == len(sc.SensorResearchConfig().to_dict())
    assert all(row["description"] and row["accepted"] and row["effect"] for row in rows)
    gain = next(row for row in rows if row["parameter"] == "analog_gain")
    assert (gain["default"], gain["current"]) == (1.0, 4.0) and gain["warning"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown_key": 1},
        {"cfa_pattern": "RGBW"},
        {"analog_gain": 100},
        {"black_level": "auto"},
        {"black_level": 16000, "white_level": 1000},
        {"color_matrix": [[1, 0], [0, 1]]},
        {"color_matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"white_balance_multipliers": [1, "a", 1]},
        {"demosaic_methods": ["ahd"]},
        {"synthetic_raw_label": "RAW"},
        {"shot_noise_enabled": "yes"},
    ],
)
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValueError):
        sc.load_sensor_config(CONFIG_FILE, **overrides)


def test_integer_values_and_empty_levels_load():
    config = sc.from_dict({"analog_gain": 2, "white_balance_multipliers": [2, 1, 2], "black_level": 256})
    assert config.analog_gain == 2.0 and config.white_balance_multipliers == (2.0, 1.0, 2.0)
    assert config.sensor_model().black == 256 and config.sensor_model().white == 16383


def test_run_from_an_exr_master_writes_labelled_validated_outputs(project):
    exr = exporters.write_exr(project / "master.exr", master(), {"syntheticRawLabRunId": "phase01-run"})
    codes = test_images.make_test_image(96, 64)
    config = small_config(shot_noise_enabled=True, read_noise_enabled=True, analog_gain=2.0)
    result = sr.run(exr.path, config, project, source_codes=codes, log=lambda _: None)
    manifest = result.manifest

    assert manifest["status"] == "complete"
    assert manifest["input"]["lineage"]["syntheticRawLabRunId"] == "phase01-run"
    assert manifest["labels"]["all_labelled"], manifest["labels"]
    assert set(manifest["roundtrip"]) == {"bilinear", "malvar", "libraw"}
    dng = manifest["exports"]["dng"]
    assert dng["valid"] and dng["checks"]["libraw_mosaic_identical"]
    assert manifest["exports"]["mosaic_tiff"]["valid"]
    for export in manifest["exports"].values():
        sidecar = json.loads(Path(export["path"] + ".json").read_text())
        assert sidecar["provenance"]["synthetic_raw_label"] == ss.SYNTHETIC_RAW_LABEL
        assert "not camera-native RAW" in sidecar["provenance"]["format_class"]
    with tifffile.TiffFile(dng["path"]) as tif:
        assert "SYNTHETIC MOSAIC" in tif.pages[0].description
    assert "SYNTHETIC" in Path(dng["path"]).name

    for method, summary in manifest["roundtrip"].items():
        assert 15 < summary["full_frame"]["psnr_encoded_db"] < 80, method
        assert summary["sample"]["description"] == "whole image"
        assert summary["information"]["estimated"].startswith("the two unsampled channels")
    assert manifest["bit_depth"]["expansion"]["information_added"] is False
    assert {r["name"] for r in manifest["bit_depth"]["representations"]} >= {"rgb8_srgb", "rgb32_float_linear"}
    assert manifest["mosaic_quantization"]["levels_in_unit_range"] == 16383 - 512 + 1
    assert [s["classification"] for s in manifest["stages"][1:3]] == ["Simulation", "Simulation"]
    assert len(list((result.outputs_dir / "bit_depth").iterdir())) == 4

    report = result.report_path.read_text()
    assert sr.NOTICE in report and ss.SYNTHETIC_RAW_LABEL in report
    assert "Phase 5 exit criterion met" in report
    assert "Camera-native RAW" in report and "Synthetic mosaic DNG" in report
    for panel in manifest["crop_panels"]:
        assert Path(panel["path"]).exists()
    assert (result.reports_dir / "figures" / "delta_e_libraw.png").exists()


def test_run_from_an_array_without_dng(project):
    config = small_config(export_synthetic_dng=False, bit_depth_write_files=False, demosaic_methods=["malvar"])
    result = sr.run(master(161, 91), config, project, log=lambda _: None)
    manifest = result.manifest
    assert manifest["status"] == "complete"
    assert (manifest["input"]["width"], manifest["input"]["height"]) == (160, 90)
    assert "cropped to even size" in manifest["input"]["notes"][0]
    assert set(manifest["exports"]) == {"mosaic_tiff"} and set(manifest["roundtrip"]) == {"malvar"}
    assert "expansion" not in manifest["bit_depth"]
    assert "no DNG was exported" in result.report_path.read_text()


def test_a_failing_stage_keeps_a_partial_report(project, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("demosaic exploded")

    monkeypatch.setattr(ss, "develop", broken)
    with pytest.raises(sr.pipeline.StageError) as caught:
        sr.run(master(), small_config(export_synthetic_dng=False), project, log=lambda _: None)
    assert caught.value.stage == "develop_bilinear"
    manifest = json.loads(caught.value.manifest_path.read_text())
    assert manifest["status"] == "failed in stage develop_bilinear"
    assert "did not complete" in (caught.value.manifest_path.parent / "report.md").read_text()


def test_roundtrip_metrics_sample_tiles_on_large_images():
    rng = np.random.default_rng(0)
    reference = rng.uniform(0, 1, (300, 400, 3)).astype(np.float32)
    candidate = np.clip(reference + rng.normal(0, 0.01, reference.shape).astype(np.float32), 0, 1)
    metrics, delta_e = sr.roundtrip_metrics(reference, candidate, tile=128, grid=2)
    assert metrics["sample"] == {"description": "2x2 grid of 128-px full-resolution tiles", "width": 256, "height": 256}
    assert delta_e.shape == (256, 256)
    assert 30 < metrics["full_frame"]["psnr_encoded_db"] < 60
    identical, _ = sr.roundtrip_metrics(reference, reference, tile=128, grid=2)
    assert identical["full_frame"]["psnr_encoded_db"] == float("inf")


def test_read_linear_master_converts_other_primaries(tmp_path):
    p3 = (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290)
    image = np.full((4, 6, 3), 0.25, dtype=np.float32)
    image[..., 0] = 0.5
    path = exporters.write_exr(tmp_path / "p3.exr", image, {}, precision="float", chromaticities=p3).path
    loaded = sr.read_linear_master(path)
    expected = cm.to_working_space(image, cm.linear_color_space_from_chromaticities(p3))
    np.testing.assert_allclose(loaded.linear, expected, atol=1e-6)
    assert "converted from" in loaded.notes[0]
