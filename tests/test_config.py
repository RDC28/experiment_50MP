from pathlib import Path

import pytest

from synthetic_raw_lab.config import ExperimentConfig, load_config, parameter_table

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_code_defaults():
    assert load_config(ROOT / "configs" / "baseline_default.yaml") == ExperimentConfig()


def test_prd_defaults():
    config = ExperimentConfig()
    assert config.target_megapixels == 50.0
    assert config.preserve_aspect_ratio is True
    assert config.working_color_space == "linear_rec709"
    assert config.processing_precision == "float32"
    assert config.export_tiff16 and config.export_exr and config.save_intermediates
    assert config.random_seed == 42
    assert config.baselines == ("lanczos", "bicubic")


def test_overrides_and_quick_preset(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("runtime_preset: quick\nbaselines: [bicubic]\ntarget_megapixels: 40\n")
    config = load_config(path, quick_target_megapixels=1.5)
    assert config.baselines == ("bicubic",)
    assert config.target_megapixels == 40.0
    assert config.effective_target_megapixels == 1.5
    assert config.replace(runtime_preset="standard").effective_target_megapixels == 40.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown_key": 1},
        {"resolution_mode": "nearest"},
        {"baselines": ["lanczos", "nearest"]},
        {"baselines": []},
        {"baselines": ["lanczos", "lanczos"]},
        {"target_megapixels": 0},
        {"pixel_count_tolerance": 0.5},
        {"export_exr": "yes"},
        {"crop_count": 2.5},
        {"crops": [[0, 0, 10]]},
        {"crops": [[0, 0, 0, 10]]},
        {"profile": " "},
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        load_config(**overrides)


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_every_parameter_is_documented():
    rows = parameter_table(ExperimentConfig(runtime_preset="quick"))
    assert {row["parameter"] for row in rows} == set(ExperimentConfig().to_dict())
    for row in rows:
        assert row["description"] and row["accepted"] and row["effect"]
    preset = next(row for row in rows if row["parameter"] == "runtime_preset")
    assert (preset["default"], preset["current"]) == ("standard", "quick")


def test_super_resolution_profile():
    config = load_config(ROOT / "configs" / "super_resolution_default.yaml")
    assert config.enable_super_resolution and config.preserve_source_structure
    assert config.sr_backend == "swinir" and config.sr_modes == ("conservative", "balanced")
    assert (config.sr_tile_size, config.sr_tile_overlap, config.sr_strength) == (0, 64, 1.0)
    assert config.replace(enable_super_resolution=False, profile="baselines") == ExperimentConfig()
