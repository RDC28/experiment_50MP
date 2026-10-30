"""JSON contracts and actual tiny configured source-to-TIFF execution."""

import json

import numpy as np
import pytest
from PIL import Image

from prd_v5.analytic import AnalyticISP
from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis


def test_config_roundtrip_and_execution(tmp_path):
    config = AnalyticRunConfig(
        camera=CameraHypothesis(seed=19, iso=200, shot_noise=False, read_noise=False),
        isp=AnalyticISP(exposure_gain=1.7, white_balance=(1.2, 1, 1.4)),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        strip_rows=3,
    )
    restored = AnalyticRunConfig.from_json(config.to_json())
    assert restored == config
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    report = restored.run(source, tmp_path / "bundle")
    assert report["camera"] == json.loads(config.to_json())["camera"]
    assert json.loads(json.dumps(report["isp"])) == json.loads(config.to_json())["isp"]
    assert report["limits"]["strip_rows"] == 3
    assert report["geometry"]["width"] == 32
    assert report["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"schema_version":2}',
        '{"schema_version":true}',
        '{"strip_rows":true}',
        '{"max_decode_bytes":0}',
        '{"target_megapixels":NaN}',
        '{"target_megapixels":1e999}',
        '{"target_megapixels":true}',
        '{"assume_unprofiled_srgb":1}',
        '{"unknown":1}',
        '{"camera":{"unknown":1}}',
        '{"camera":[]}',
        '{"camera":{"adc_bits":17}}',
        '{"isp":{"white_balance":[0,1,1]}}',
        '{"strip_rows":1,"strip_rows":2}',
        '{"camera":{"iso":100,"iso":200}}',
    ],
)
def test_invalid_config(text):
    with pytest.raises(ValueError):
        AnalyticRunConfig.from_json(text)


def test_default_full_scale_still_requires_colab_budget(tmp_path):
    config = AnalyticRunConfig.from_json("{}")
    assert config.target_megapixels == 50
    source = tmp_path / "source.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(source)
    with pytest.raises(ValueError, match="memory budget"):
        config.run(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("policy", ["LETTERBOX_SENSOR_MARGIN", "CROP_HORIZONTAL", "OUTPAINT_VERTICAL"])
def test_native_config_uses_full_sensor_despite_tiny_target(tmp_path, policy):
    config = AnalyticRunConfig.from_json(
        json.dumps(
            {
                "geometry_mode": "SONY_A1II_NATIVE",
                "native_fit_policy": policy,
                "target_megapixels": 0.001024,
                "assume_unprofiled_srgb": True,
            }
        )
    )
    assert AnalyticRunConfig.from_json(config.to_json()) == config
    source = tmp_path / "source.png"
    # Exact 3:2 requires no outpainting, so every policy reaches memory admission.
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(source)
    with pytest.raises(ValueError, match="LibRaw development memory budget"):
        config.run(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_outpainting_rejected_before_output_or_frame_allocation(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(source)
    config = AnalyticRunConfig(
        geometry_mode="SONY_A1II_NATIVE",
        native_fit_policy="OUTPAINT_VERTICAL",
        assume_unprofiled_srgb=True,
        max_develop_bytes=4_000_000_000,
    )

    def forbidden_allocation(*args, **kwargs):
        pytest.fail("Outpainting must be rejected before any frame allocation")

    monkeypatch.setattr(np, "memmap", forbidden_allocation)
    with pytest.raises(ValueError, match="requires outpainting"):
        config.run(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "text",
    [
        '{"geometry_mode":"UNKNOWN"}',
        '{"geometry_mode":null}',
        '{"native_fit_policy":"UNKNOWN"}',
        '{"native_fit_policy":false}',
        '{"target_megapixels":1001}',
    ],
)
def test_config_rejects_unsupported_geometry_before_execution(text):
    with pytest.raises(ValueError):
        AnalyticRunConfig.from_json(text)
