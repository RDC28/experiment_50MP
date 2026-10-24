"""The notebook 06 evaluation profile (PRD Phase 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_raw_lab import ablation, reference_test
from synthetic_raw_lab import evaluation_config as ec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "evaluation_default.yaml"


def test_default_file_matches_the_code_defaults_apart_from_the_source_labels():
    from_file, from_code = ec.load_evaluation_config(DEFAULT).to_dict(), ec.load_evaluation_config().to_dict()
    labels = from_file["aggregation"].pop("labels")
    from_code["aggregation"].pop("labels")
    assert from_file == from_code
    assert labels == {"40e304406633": "portrait (red saree, festival)", "5fe78cd34888": "night forest path"}


def test_defaults_cover_the_matrix_and_the_captain_scale():
    config = ec.load_evaluation_config(DEFAULT)
    assert config.ablation.configurations == ablation.DEFAULT_MATRIX
    assert config.reference.scales == (ec.CAPTAIN_SCALE,) and abs(ec.CAPTAIN_SCALE - 1.7134) < 1e-9
    assert set(config.reference.references) == set(reference_test.REFERENCES)
    assert "lanczos" in config.reference.configurations
    assert config.universal.sr_modes == ("conservative",)
    assert config.aggregation.label("40e304406633ec29dce11d9ca381d93b4ed24b4096c95a5cbf0b40973d0f6329").startswith(
        "portrait"
    )


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"ablation": {"configurations": ["nope"]}}, "unknown ablation configurations"),
        ({"ablation": {"configurations": ["lanczos", "lanczos"]}}, "twice"),
        ({"ablation": {"crop_size": 10}}, "at least 64"),
        ({"ablation": {"scene": "space"}}, "ablation.scene"),
        ({"ablation": {"grain_amount": 0.5}}, "grain_amount"),
        ({"ablation": {"sensor": {"adc_bit_depth": 9}}}, "adc_bit_depth"),
        ({"ablation": {"sensor": {"warp": 1}}}, "unknown ablation.sensor"),
        ({"ablation": {"performance_sweep": [{"speed": 2}]}}, "performance_sweep"),
        ({"reference": {"references": ["moon"]}}, "unknown reference"),
        ({"reference": {"degradations": ["blur"]}}, "degradations"),
        ({"reference": {"scales": [1.0]}}, "scales"),
        ({"reference": {"configurations": ["sr_conservative"]}}, "must include lanczos"),
        ({"aggregation": {"review_crops": [[0.9, 0, 0.5, 0.5]]}}, "review_crops"),
        ({"report": {"c_max_psnr_gain_db": 2.0}}, "overlap"),
        ({"ablation": {"surprise": 1}}, "unknown keys in ablation"),
        ({"extra": 1}, "unknown evaluation keys"),
    ],
)
def test_invalid_settings_are_rejected(sections, message):
    with pytest.raises(ValueError, match=message):
        ec.load_evaluation_config(DEFAULT, **sections)


def test_section_overrides_update_single_keys_and_build_the_sensor_model():
    config = ec.load_evaluation_config(
        DEFAULT,
        ablation={"crop_size": 0, "crop_center": [10, 20], "sensor": {"adc_bit_depth": 12, "demosaic": "bilinear"}},
    )
    assert config.ablation.crop_size == 0 and config.ablation.crop_center == (10, 20)
    assert config.ablation.grain_amount == 0.008  # untouched keys keep the file's values
    assert config.ablation.sensor_model().adc_bit_depth == 12 and config.ablation.demosaic == "bilinear"
    rows = {r["parameter"]: r for r in ec.parameter_rows(config)}
    assert rows["ablation.crop_size"]["current"] == 0 and rows["ablation.crop_size"]["default"] == 1536


def test_universal_overrides_reach_the_universal_profile():
    config = ec.load_evaluation_config(DEFAULT, universal_overrides={"super_resolution": {"enabled": False}})
    assert not config.universal.sr_enabled
