"""Tiny, allocation-free checks of the PRD_v3 configuration contract."""

import ast
import json
import re
from pathlib import Path
from typing import Literal, get_args, get_type_hints

import pytest

from synthraw.config import ExperimentConfig


def test_every_prd_default_is_present():
    prd = (Path(__file__).parents[2] / "PRD_v3.md").read_text()
    block = prd.split("The required configuration surface is:")[1].split("```python")[1].split("```")[0]
    config = ExperimentConfig().to_dict()
    for name, value in re.findall(r"^(\w+) = (.+)$", block, re.M):
        assert config[name] == ast.literal_eval(value)


def test_json_round_trip():
    config = ExperimentConfig(SPECTRAL_MODE="CALIBRATED_CURVES", CALIBRATION_ASSETS=("curves.json",))
    assert ExperimentConfig.from_dict(json.loads(json.dumps(config.to_dict(), allow_nan=False))) == config


@pytest.mark.parametrize("name,hint", get_type_hints(ExperimentConfig).items())
def test_each_typed_control_rejects_wrong_type_or_choice(name, hint):
    invalid = "UNKNOWN" if getattr(hint, "__origin__", None) is Literal else None
    if name in ("INPUT_PATH", "REFERENCE_RAW_PATH"):
        invalid = 12
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig(**{name: invalid})


@pytest.mark.parametrize(
    "name,value",
    [
        ("ADC_BITS", 12),
        ("DNG_STORAGE_BITS", 14),
        ("N_SENSOR_REALISATIONS", 0),
        ("SCENE_SEED", -1),
        ("SENSOR_NOISE_SEED", True),
        ("SENSOR_BODY_SEED", 1.5),
        ("GENERATIVE_DETAIL_STRENGTH", -0.1),
        ("GENERATIVE_DETAIL_STRENGTH", 1.1),
        ("HDR_HIGHLIGHT_EV", -1),
        ("HDR_SHADOW_EV", float("nan")),
        ("HDR_HIGHLIGHT_EV", float("inf")),
        ("INPUT_PATH", ""),
        ("DRIVE_MOUNT_PATH", ""),
        ("CALIBRATION_ASSETS", ("",)),
    ],
)
def test_invalid_ranges(name, value):
    with pytest.raises(ValueError):
        ExperimentConfig(**{name: value})


@pytest.mark.parametrize("settings", [{"EXPERIMENTAL_16BIT_SENSOR": True}, {"ADC_BITS": 16}])
def test_16bit_is_always_virtual(settings):
    config = ExperimentConfig(**settings)
    assert config.TARGET_PROFILE == "VIRTUAL_16BIT_SENSOR"
    assert config.ADC_BITS == 16


def test_cinema_is_virtual_and_16bit_takes_precedence():
    assert ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL").TARGET_PROFILE == "CINEMA_50MP_VIRTUAL"
    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", ADC_BITS=16)
    assert config.TARGET_PROFILE == "VIRTUAL_16BIT_SENSOR"


def test_calibrated_spectra_require_assets():
    with pytest.raises(ValueError, match="requires CALIBRATION_ASSETS"):
        ExperimentConfig(SPECTRAL_MODE="CALIBRATED_CURVES")


@pytest.mark.parametrize("mode", ["PATH", "DRIVE"])
def test_input_paths_required(mode):
    with pytest.raises(ValueError, match="require INPUT_PATH"):
        ExperimentConfig(INPUT_MODE=mode)
    assert ExperimentConfig(INPUT_MODE=mode, INPUT_PATH="image.png").INPUT_PATH == "image.png"


def test_unknown_keys_rejected():
    with pytest.raises(TypeError):
        ExperimentConfig.from_dict({"TYPO": True})


def test_all_documented_choices_construct():
    for name, hint in get_type_hints(ExperimentConfig).items():
        if getattr(hint, "__origin__", None) is Literal:
            for value in get_args(hint):
                ExperimentConfig(
                    **{name: value},
                    **({"INPUT_PATH": "image.png"} if name == "INPUT_MODE" else {}),
                    **({"CALIBRATION_ASSETS": ("curves.json",)} if name == "SPECTRAL_MODE" else {}),
                )


def test_unimplemented_lossless_dng_compression_is_not_offered():
    assert get_args(get_type_hints(ExperimentConfig)["DNG_COMPRESSION"]) == ("UNCOMPRESSED",)
    with pytest.raises(ValueError, match="DNG_COMPRESSION"):
        ExperimentConfig(DNG_COMPRESSION="LOSSLESS")
