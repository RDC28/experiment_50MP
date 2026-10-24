"""Human profile loading, documentation and protection of invalid settings."""

from pathlib import Path

import pytest
import yaml

from synthetic_raw_lab.human_config import HumanConfig, from_dict, load_config, parameter_table

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_defaults_and_round_trip():
    config = load_config(ROOT / "configs/human_default.yaml")
    assert config == HumanConfig()
    assert from_dict(yaml.safe_load(yaml.safe_dump(config.to_dict()))) == config


def test_overrides_preserve_human_type_and_nested_settings():
    config = load_config(
        runtime_preset="quick",
        quick_target_megapixels=0.02,
        manual_face_boxes=[[0, 1, 20, 30, 0.9]],
        primary_subject_index=0,
        regions={"hair": 0.1},
    )
    assert config.effective_target_megapixels == 0.02
    assert config.regions.hair == 0.1
    assert config.regions.primary_face == 0
    assert config.replace(random_seed=7).manual_face_boxes == config.manual_face_boxes
    assert isinstance(config.replace(random_seed=7), HumanConfig)


@pytest.mark.parametrize(
    "values",
    [
        {"unknown": 1},
        {"sr_backend": "learned"},
        {"face_detector": "unknown"},
        {"input_category": "night"},
        {"profile": "other"},
        {"regions": {"typo": 1}},
        {"regions": {"skin": 2}},
        {"regions": {"hair": float("nan")}},
        {"identity": {"retry_strength": 1}},
        {"face_tiers": {"tier2_min_pixels": 100}},
        {"manual_face_boxes": [[0, 0, 0, 1]]},
        {"manual_face_boxes": [[True, 0, 10, 10]]},
        {"manual_face_boxes": [3]},
        {"manual_face_boxes": [[0, 0, 10, 10], [0, 0, 10, 10]]},
        {"primary_subject_index": 0},
        {"primary_subject_index": False},
        {"export_tiff16": False},
        {"exr_precision": "float"},
        {"target_megapixels": 0},
        {"identity": []},
        {"manual_face_boxes": None},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        from_dict(values)


@pytest.mark.parametrize("text", ["", "- a", "regions: null"])
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_every_leaf_is_documented():
    rows = parameter_table()
    assert len({row["parameter"] for row in rows}) == len(rows)
    for row in rows:
        assert all(key in row for key in ("default", "current", "warning"))
        assert all(row[key] for key in ("description", "accepted", "effect"))
    for row in rows:
        if row["parameter"].startswith("face_tiers."):
            assert "calibrated experimentally" in row["warning"]
    assert {"regions.skin", "identity.min_ssim", "sr_backend", "primary_subject_index"} <= {
        row["parameter"] for row in rows
    }


@pytest.mark.parametrize("key", ["general_sr_strength", "edge_sharpening"])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), True, "1"])
def test_invalid_classical_control_strengths(key, value):
    with pytest.raises(ValueError):
        from_dict({key: value})


@pytest.mark.parametrize("value", [-0.51, 0.51, float("nan"), float("inf"), True, "0"])
def test_invalid_tonal_contrast(value):
    import numpy as np

    from synthetic_raw_lab.human_pipeline import adjust_tonal_contrast

    with pytest.raises(ValueError):
        from_dict({"tonal_contrast": value})
    with pytest.raises(ValueError):
        adjust_tonal_contrast(np.zeros((1, 1, 3), np.float32), value)


@pytest.mark.parametrize(
    "value", [[1, 1], [1, 1, 1, 1], [0.49, 1, 1], [1, 1.51, 1], [1, 1, float("nan")], [True, 1, 1], ["1", 1, 1], None]
)
def test_invalid_colour_balance(value):
    import numpy as np

    from synthetic_raw_lab.human_pipeline import balance_colour

    with pytest.raises(ValueError):
        from_dict({"colour_balance": value})
    with pytest.raises(ValueError):
        balance_colour(np.zeros((1, 1, 3), np.float32), value)


@pytest.mark.parametrize("backend", ["swinir", "realesrgan", "lanczos_stand_in"])
def test_sr_backend_round_trip_and_shared_controls(backend):
    config = from_dict({"sr_backend": backend, "sr_modes": ["conservative", "balanced"], "sr_device": "cpu"})
    assert config.sr_backend == backend
    assert config.sr_modes == ("conservative", "balanced")
    assert config.sr_device == "cpu"
    assert from_dict(config.to_dict()) == config
    assert HumanConfig().sr_backend == "swinir"


@pytest.mark.parametrize("backend", ["swinir", "realesrgan", "lanczos_stand_in"])
@pytest.mark.parametrize(
    "invalid", [{"sr_tile_size": 32}, {"sr_tile_size": 128, "sr_tile_overlap": 80}, {"sr_device": "invalid"}]
)
def test_all_backends_retain_phase2_validation(backend, invalid):
    with pytest.raises(ValueError):
        from_dict({"sr_backend": backend, **invalid})
