"""Portable structural schema and executable semantic validation stay distinct."""

import json
from dataclasses import fields
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from prd_v5.analytic import AnalyticISP
from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.optics import Distortion, Vignetting


@pytest.fixture
def schema():
    return json.loads((Path(__file__).parents[1] / "analytic-config.schema.json").read_text())


def test_schema_covers_every_implemented_setting(schema):
    Draft202012Validator.check_schema(schema)
    assert set(schema["properties"]) == {f.name for f in fields(AnalyticRunConfig)}
    for name, cls in (("camera", CameraHypothesis), ("isp", AnalyticISP)):
        assert set(schema["$defs"][name]["properties"]) == {f.name for f in fields(cls)}
    validator = Draft202012Validator(schema)
    validator.validate({})
    validator.validate(json.loads(AnalyticRunConfig().to_json()))
    config = AnalyticRunConfig(
        vignetting=Vignetting(0.7),
        distortion=Distortion(k1=0.1, k2=-0.1, p1=0.02, p2=-0.02),
        accept_experimental_optics=True,
        camera=CameraHypothesis(seed=2**64 - 1, black_level=0, iso=32000, read_noise_e=0),
        isp=AnalyticISP(white_balance=(2, 1, 3), inverse_smoothstep=True),
        geometry_mode="SONY_A1II_NATIVE",
        native_fit_policy="CROP_HORIZONTAL",
    )
    validator.validate(json.loads(config.to_json()))
    assert AnalyticRunConfig.from_json(config.to_json()) == config


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": 1},
        {"distortion": {"k1": 0.26}},
        {"distortion": {"p1": True}},
        {"distortion": {"unknown": 0}},
        {"vignetting": {"corner_strength": 1.1}},
        {"vignetting": {"unknown": 0}},
        {"accept_experimental_optics": 1},
        {"schema_version": 2},
        {"strip_rows": True},
        {"max_decode_bytes": 0},
        {"target_megapixels": 1001},
        {"geometry_mode": "UNKNOWN"},
        {"native_fit_policy": "UNKNOWN"},
        {"camera": {"seed": 2**64}},
        {"camera": {"adc_bits": 16}},
        {"camera": {"black_level": 16383}},
        {"camera": {"iso": 99}},
        {"camera": {"serial": "invented"}},
        {"isp": {"white_balance": [1, 0, 1]}},
        {"isp": {"camera_to_srgb": [[1, 0, 0]]}},
        {"isp": {"inverse_smoothstep": 1}},
    ],
)
def test_schema_and_parser_reject_invalid_settings(schema, value):
    assert list(Draft202012Validator(schema).iter_errors(value))
    with pytest.raises(ValueError):
        AnalyticRunConfig.from_json(json.dumps(value))


def test_structural_schema_does_not_claim_matrix_physics(schema):
    value = {"isp": {"camera_to_srgb": [[0, 0, 0]] * 3}}
    Draft202012Validator(schema).validate(value)
    with pytest.raises(ValueError, match="well-conditioned"):
        AnalyticRunConfig.from_json(json.dumps(value))
