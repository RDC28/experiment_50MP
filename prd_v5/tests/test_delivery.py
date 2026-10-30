"""Real tiny output bundle contract and synthetic ADC/gain validation."""

import json
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest
import rawpy
from PIL import Image

from prd_v5.delivery import digest
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.physics import photon_transfer
from prd_v5.presets import full_quality_config


@pytest.mark.parametrize("adc_bits", [12, 14, 16])
def test_full_quality_fixture_outputs_registry_and_decodable_adc(tmp_path, adc_bits):
    source = tmp_path / "source.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = [90, 128, 160]
    image[0, 0] = 255
    Image.fromarray(image).save(source)
    config = replace(
        full_quality_config(), target_megapixels=0.001024, camera=CameraHypothesis(adc_bits=adc_bits, profile="GENERIC")
    )
    output = tmp_path / "run"
    report = config.run(source, output)
    expected = {
        "synthetic_50mp.dng",
        "synthetic_50mp_16bit.tif",
        "synthetic_50mp_preview.jpg",
        "clean_bayer.npy",
        "noisy_bayer.npy",
        "field_registry.json",
        "camera_hypothesis.json",
        "experiment_config.json",
        "validation.json",
        "validation_report.html",
        "generated_region_mask.png",
        "uncertainty_map.png",
        "source_hashes.json",
        "checkpoint_hashes.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with rawpy.imread(str(output / "synthetic_50mp.dng")) as raw:
        assert raw.white_level == (1 << adc_bits) - 1
        assert raw.raw_image.shape == (32, 32)
    catalogue = json.loads(files("prd_v5").joinpath("field_catalogue.json").read_text())
    registry = json.loads((output / "field_registry.json").read_text())
    assert {entry["name"] for entry in catalogue} <= registry.keys()
    for record in registry.values():
        assert record["method"] and record["validation"]
    for name in ("GPS", "Camera serial", "Lens serial", "Timestamp", "Battery"):
        assert registry[name]["value"] is None
    assert np.load(output / "clean_bayer.npy").shape == (32, 32)
    assert np.load(output / "noisy_bayer.npy").shape == (32, 32)
    for name, evidence in report["artifacts"].items():
        assert digest(output / name) == evidence["sha256"]
    assert report["stages"]["raster"]["method"].startswith("Lanczos-3")
    assert report["validation"]["acceptance"].startswith("conditional")
    assert np.asarray(Image.open(output / "uncertainty_map.png"))[0, 0] == 255


@pytest.mark.parametrize("seed", [0, 42, 100])
def test_photon_transfer_measures_consistent_temporal_model(seed):
    result = photon_transfer(CameraHypothesis(seed=seed))
    assert result["status"] == "passed"
    assert result["slope_relative_error"] < result["relative_tolerance"]
    assert result["dark_relative_error"] < result["relative_tolerance"]
    assert result["scope"] == "baseline simulator self-consistency only"


def test_gain_clipping_is_not_reported_as_noise_validation():
    assert photon_transfer(CameraHypothesis(iso=102400))["status"] == "not measured"


def test_extended_low_iso_well_and_adc_tags_remain_coherent(tmp_path):
    camera = CameraHypothesis(iso=50, shot_noise=False, read_noise=False)
    gain = camera.analogue_gain / camera.conversion_gain_e_per_dn
    assert camera.saturation_e == camera.full_well_e
    assert camera.white_level == round(camera.black_level + camera.full_well_e * gain)
    assert camera.white_level < (1 << camera.adc_bits) - 1
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "white").save(source)
    config = replace(full_quality_config(), target_megapixels=0.001024, camera=camera)
    output = tmp_path / "run"
    report = config.run(source, output)
    with rawpy.imread(str(output / "capture.dng")) as raw:
        assert raw.white_level == camera.white_level
        assert int(raw.raw_image.max()) <= camera.white_level
    assert report["registry"]["White level"]["value"] == camera.white_level
