"""The notebook 03 universal profile and the specialised presets it derives (PRD 20, Phase 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_raw_lab import universal_config as uc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "universal_default.yaml"


def test_default_file_matches_the_code_defaults():
    assert uc.load_universal_config(DEFAULT).to_dict() == uc.load_universal_config().to_dict()


def test_default_profile_is_conservative():
    config = uc.load_universal_config(DEFAULT)
    assert config.experiment.profile == "universal" and config.experiment.input_category == "universal"
    assert config.sr_modes == ("conservative",)
    assert config.super_resolution.mode == "conservative"
    assert config.experiment.sr_modes == ("conservative",)
    assert config.experiment.enable_super_resolution
    processing = config.processing
    assert processing.shadows.lift_strength == "none"
    assert processing.color.saturation_adjustment == "none"
    assert processing.noise.denoise_source == "off" and processing.noise.grain_amount == 0
    assert processing.failure_rule.enabled
    assert "mode" not in config.to_dict()["super_resolution"]


def test_ladder_and_sections_are_validated(tmp_path):
    with pytest.raises(ValueError, match="ladder"):
        uc.load_universal_config(DEFAULT, super_resolution={"modes": ["balanced", "conservative"]})
    with pytest.raises(ValueError, match="once each"):
        uc.load_universal_config(DEFAULT, super_resolution={"modes": ["aggressive"]})
    with pytest.raises(ValueError, match="super_resolution.mode"):
        uc.load_universal_config(DEFAULT, super_resolution={"mode": "balanced"})
    with pytest.raises(ValueError, match="unknown keys in improvement"):
        uc.load_universal_config(DEFAULT, improvement={"min_gain": 1})
    with pytest.raises(ValueError, match="unknown keys in analysis"):
        uc.load_universal_config(DEFAULT, analysis={"detector": "yunet"})
    with pytest.raises(ValueError, match="manual_override"):
        uc.load_universal_config(DEFAULT, analysis={"manual_override": "portrait"})
    with pytest.raises(ValueError, match="processing.super_resolution"):
        uc.load_universal_config(DEFAULT, processing={"super_resolution": {"enabled": False}})
    with pytest.raises(ValueError, match="tiff16"):
        uc.load_universal_config(DEFAULT, export_tiff16=False)
    with pytest.raises(ValueError, match="profile and overrides"):
        uc.load_universal_config(DEFAULT, human={"file": "x.yaml"})


def test_overrides_change_single_keys():
    config = uc.load_universal_config(
        DEFAULT,
        super_resolution={"modes": ["conservative", "balanced"], "backend": "realesrgan"},
        processing={"texture": {"sharpening_strength": 0.1}},
        analysis={"manual_override": "night", "thresholds": {"human_min_face_px": 128}},
    )
    assert config.sr_modes == ("conservative", "balanced")
    assert config.experiment.sr_backend == "realesrgan"
    assert config.processing.texture.sharpening_strength == 0.1
    assert config.processing.texture.architecture_detail_strength == "low"  # other keys keep the universal value
    assert config.analysis.manual_override == "night"
    assert config.analysis.thresholds.human_min_face_px == 128
    rows = {r["parameter"]: r for r in uc.parameter_rows(config)}
    assert rows["super_resolution.modes"]["current"] == ["conservative", "balanced"]
    assert rows["super_resolution.modes"]["default"] == ["conservative"]


def test_human_preset_forwards_shared_settings_and_overrides(tmp_path):
    config = uc.load_universal_config(
        DEFAULT,
        runtime_preset="quick",
        output_root=str(tmp_path),
        analysis={"face_detection": False, "manual_face_boxes": [[1, 2, 30, 40]]},
        human={"overrides": {"edge_sharpening": 0.1}},
    )
    human = config.human_config(ROOT)
    assert human.profile == "human_crowd" and human.input_category == "human"
    assert human.runtime_preset == "quick" and human.output_root == str(tmp_path)
    assert human.sr_backend == "swinir" and human.face_detector == "none"
    assert human.manual_face_boxes == ((1, 2, 30, 40),)
    assert human.edge_sharpening == 0.1
    assert config.human_config(ROOT, sr_enabled=False).sr_backend == "lanczos_stand_in"


def test_night_preset_uses_the_first_rung_and_overrides(tmp_path):
    config = uc.load_universal_config(
        DEFAULT, reports_root=str(tmp_path), night={"overrides": {"texture": {"sharpening_strength": 0.2}}}
    )
    preset = config.night_config(ROOT)
    assert preset.experiment.profile == "nighttime_scenery" and preset.experiment.input_category == "night"
    assert preset.night.super_resolution.mode == "conservative"
    assert preset.night.super_resolution.enabled
    assert preset.experiment.reports_root == str(tmp_path)
    assert preset.night.texture.sharpening_strength == 0.2
    assert preset.night.shadows.lift_strength == "low"  # the night profile's own value
    assert not config.night_config(ROOT, sr_enabled=False).night.super_resolution.enabled
