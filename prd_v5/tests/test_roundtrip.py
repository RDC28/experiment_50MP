"""Real tiny LibRaw reads and TIFF exports, with no model or full-frame work."""

import json

import numpy as np
import pytest
import tifffile

from prd_v5.analytic import AnalyticISP
from prd_v5.dng import write_dng
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.roundtrip import develop_roundtrip
from synthraw.libraw_validator import HardValidationError


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_real_roundtrip(tmp_path, pattern):
    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    camera = CameraHypothesis(cfa_pattern=pattern)
    isp = AnalyticISP(white_balance=(2, 1, 1.5))
    mosaic = (np.arange(64 * 48).reshape(48, 64) + 800).astype(np.uint16)
    source, destination = tmp_path / "source.dng", tmp_path / "render.tiff"
    write_dng(source, mosaic, camera, plan, isp)
    report = develop_roundtrip(source, destination, mosaic, camera, plan, isp)
    json.dumps(report, allow_nan=False)
    rgb = tifffile.imread(destination)
    assert rgb.shape == (48, 64, 3)
    assert rgb.dtype == np.uint16
    assert np.ptp(rgb) > 0
    assert report["tiff_roundtrip"].startswith("passed")
    with pytest.raises(HardValidationError, match="pixels differ"):
        develop_roundtrip(source, tmp_path / "bad.tiff", mosaic + 1, camera, plan, isp)
    assert not (tmp_path / "bad.tiff").exists()


def test_budget_and_alias_rejected_before_decode(tmp_path):
    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    args = (tmp_path / "absent.dng", tmp_path / "out.tiff", None, CameraHypothesis(), plan, AnalyticISP())
    with pytest.raises(ValueError, match="memory budget"):
        develop_roundtrip(*args, max_working_bytes=1)
    with pytest.raises(ValueError, match="positive integer"):
        develop_roundtrip(*args, max_working_bytes=True)
    with pytest.raises(ValueError, match="differ"):
        develop_roundtrip(args[0], args[0], *args[2:])


def test_exposure_sweep_real_dng_linear_response(tmp_path):
    from prd_v5.roundtrip import develop_exposure_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    camera, isp = CameraHypothesis(), AnalyticISP()
    mosaic = np.full((48, 64), camera.black_level + 100, dtype=np.uint16)
    source = tmp_path / "source.dng"
    write_dng(source, mosaic, camera, plan, isp)
    original = source.read_bytes()
    report = develop_exposure_sweep(source, tmp_path / "edits")
    json.dumps(report, allow_nan=False)
    assert [item["ev"] for item in report["renders"]] == [-3, -2, 0, 2, 3]
    images = [tifffile.imread(item["path"]) for item in report["renders"]]
    baseline = images[2].astype(float)
    for item, rgb in zip(report["renders"], images, strict=True):
        assert rgb.dtype == np.uint16 and rgb.shape == (48, 64, 3)
        np.testing.assert_allclose(rgb, baseline * 2.0 ** item["ev"], atol=8)
        assert item["saturated_samples"] == 0
    assert source.read_bytes() == original
    assert report["highlight_recovery"] == report["physical_dynamic_range"] == "not measured"
    with pytest.raises(ValueError, match="memory budget"):
        develop_exposure_sweep(source, tmp_path / "budget", max_working_bytes=1)
    assert not (tmp_path / "budget").exists()
    with pytest.raises(FileExistsError):
        develop_exposure_sweep(source, tmp_path / "edits")
    assert len(list((tmp_path / "edits").iterdir())) == 5


def test_sweep_failure_removes_only_its_new_outputs(tmp_path, monkeypatch):
    from prd_v5.roundtrip import develop_exposure_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    write_dng(source, np.full((48, 64), 1000, np.uint16), CameraHypothesis(), plan, AnalyticISP())
    original_write = tifffile.imwrite
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("fixture disk failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imwrite", fail_second)
    with pytest.raises(OSError, match="fixture disk failure"):
        develop_exposure_sweep(source, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert source.is_file()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 17, "0"])
def test_sweep_invalid_baseline_fails_before_io(tmp_path, value):
    from prd_v5.roundtrip import develop_exposure_sweep

    with pytest.raises(ValueError, match="baseline_exposure_ev"):
        develop_exposure_sweep(tmp_path / "absent", tmp_path / "edits", baseline_exposure_ev=value)
    assert not (tmp_path / "edits").exists()


def test_wb_sweep_real_dng_relative_response(tmp_path):
    from prd_v5.roundtrip import develop_wb_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    write_dng(source, np.full((48, 64), 2500, np.uint16), CameraHypothesis(), plan, AnalyticISP())
    original = source.read_bytes()
    report = develop_wb_sweep(source, tmp_path / "wb")
    json.dumps(report, allow_nan=False)
    cool, neutral, warm = report["renders"]
    assert [entry["name"] for entry in report["renders"]] == ["cool", "neutral", "warm"]
    assert cool["user_wb"][0] == pytest.approx(neutral["user_wb"][0] / 1.25)
    assert warm["user_wb"][2] == pytest.approx(neutral["user_wb"][2] / 1.25)
    for entry, red, blue in ((cool, 0.8, 1.25), (neutral, 1, 1), (warm, 1.25, 0.8)):
        means = entry["mean_linear_rgb_code"]
        assert means[0] / means[1] == pytest.approx(red, abs=0.001)
        assert means[2] / means[1] == pytest.approx(blue, abs=0.001)
    for entry in report["renders"]:
        rgb = tifffile.imread(entry["path"])
        assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint16
    assert source.read_bytes() == original
    assert report["white_balance_accuracy"] == "not measured"
    with pytest.raises(FileExistsError):
        develop_wb_sweep(source, tmp_path / "wb")
    with pytest.raises(ValueError, match="memory budget"):
        develop_wb_sweep(source, tmp_path / "budget", max_working_bytes=1)
    assert not (tmp_path / "budget").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": 1},
        {"strength": True},
        {"strength": float("nan")},
        {"baseline_exposure_ev": 17},
        {"max_working_bytes": False},
    ],
)
def test_wb_sweep_invalid_settings_before_io(tmp_path, kwargs):
    from prd_v5.roundtrip import develop_wb_sweep

    with pytest.raises(ValueError):
        develop_wb_sweep(tmp_path / "absent", tmp_path / "wb", **kwargs)
    assert not (tmp_path / "wb").exists()


def test_wb_sweep_write_failure_cleanup(tmp_path, monkeypatch):
    from prd_v5.roundtrip import develop_wb_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    write_dng(source, np.full((48, 64), 2500, np.uint16), CameraHypothesis(), plan, AnalyticISP())

    def fail(*args, **kwargs):
        raise OSError("fixture write failure")

    monkeypatch.setattr(tifffile, "imwrite", fail)
    with pytest.raises(OSError, match="fixture write failure"):
        develop_wb_sweep(source, tmp_path / "wb")
    assert not (tmp_path / "wb").exists()
    assert source.is_file()


def test_demosaic_sweep_real_dng(tmp_path):
    import rawpy

    from prd_v5.roundtrip import develop_demosaic_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    y, x = np.indices((48, 64))
    mosaic = (800 + 100 * ((x + y) % 7) + x * 15).astype(np.uint16)
    write_dng(source, mosaic, CameraHypothesis(), plan, AnalyticISP())
    original = source.read_bytes()
    report = develop_demosaic_sweep(source, tmp_path / "edits", baseline_exposure_ev=-1)
    json.dumps(report, allow_nan=False)
    assert report["status"] == "measured"
    assert [entry["algorithm"] for entry in report["renders"]] == ["LINEAR", "VNG", "AHD"]
    images = []
    for entry in report["renders"]:
        rgb = tifffile.imread(entry["path"])
        assert rgb.dtype == np.uint16 and rgb.shape == (48, 64, 3)
        with rawpy.imread(str(source)) as raw:
            expected = raw.postprocess(
                output_bps=16,
                use_camera_wb=True,
                use_auto_wb=False,
                no_auto_bright=True,
                bright=0.5,
                user_flip=0,
                output_color=rawpy.ColorSpace.sRGB,
                gamma=(1, 1),
                highlight_mode=rawpy.HighlightMode.Clip,
                demosaic_algorithm=rawpy.DemosaicAlgorithm[entry["algorithm"]],
            )
        np.testing.assert_array_equal(rgb, expected)
        images.append(rgb)
    assert not np.array_equal(images[0], images[-1])
    assert source.read_bytes() == original
    assert report["detail_accuracy"] == "not measured"
    with pytest.raises(FileExistsError):
        develop_demosaic_sweep(source, tmp_path / "edits")
    with pytest.raises(ValueError, match="memory budget"):
        develop_demosaic_sweep(source, tmp_path / "budget", max_working_bytes=1)
    assert not (tmp_path / "budget").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_exposure_ev": True},
        {"baseline_exposure_ev": float("nan")},
        {"baseline_exposure_ev": 17},
        {"max_working_bytes": False},
    ],
)
def test_demosaic_invalid_settings_before_io(tmp_path, kwargs):
    from prd_v5.roundtrip import develop_demosaic_sweep

    with pytest.raises(ValueError):
        develop_demosaic_sweep(tmp_path / "absent", tmp_path / "edits", **kwargs)
    assert not (tmp_path / "edits").exists()


def test_demosaic_failure_cleanup(tmp_path, monkeypatch):
    from prd_v5.roundtrip import develop_demosaic_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    write_dng(source, np.full((48, 64), 2500, np.uint16), CameraHypothesis(), plan, AnalyticISP())
    original_write = tifffile.imwrite
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("fixture write failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imwrite", fail_second)
    with pytest.raises(OSError, match="fixture write failure"):
        develop_demosaic_sweep(source, tmp_path / "edits")
    assert not (tmp_path / "edits").exists()
    assert source.is_file()


def test_demosaic_unavailable_algorithms_visible(tmp_path, monkeypatch):
    import rawpy

    from prd_v5.roundtrip import develop_demosaic_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    write_dng(source, np.full((48, 64), 2500, np.uint16), CameraHypothesis(), plan, AnalyticISP())
    monkeypatch.setattr(rawpy.DemosaicAlgorithm, "isSupported", property(lambda self: False))
    report = develop_demosaic_sweep(source, tmp_path / "edits")
    assert report["status"] == "not measured"
    assert report["renders"] == []
    assert [entry["algorithm"] for entry in report["skipped"]] == ["LINEAR", "VNG", "AHD"]
    assert all("Unavailable" in entry["reason"] for entry in report["skipped"])
    assert list((tmp_path / "edits").iterdir()) == []


def test_tone_sweep_real_dng_response(tmp_path):
    import rawpy

    from prd_v5.roundtrip import develop_tone_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    camera = CameraHypothesis()
    source = tmp_path / "source.dng"
    mosaic = np.broadcast_to(np.linspace(camera.black_level, camera.white_level, 64), (48, 64)).astype(np.uint16).copy()
    write_dng(source, mosaic, camera, plan, AnalyticISP(white_balance=(2, 1, 1.5)))
    original = source.read_bytes()
    report = develop_tone_sweep(source, tmp_path / "tone", baseline_exposure_ev=-1)
    json.dumps(report, allow_nan=False)
    assert [entry["name"] for entry in report["renders"]] == [
        "highlights_low",
        "highlights_high",
        "shadows_low",
        "shadows_high",
    ]
    shadows = []
    for entry in report["renders"]:
        with rawpy.imread(str(source)) as raw:
            baseline = raw.postprocess(
                output_bps=16,
                use_camera_wb=True,
                use_auto_wb=False,
                no_auto_bright=True,
                bright=0.5,
                user_flip=0,
                output_color=rawpy.ColorSpace.sRGB,
                gamma=(1, 1),
                highlight_mode=entry["highlight_level"],
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            )
        actual = tifffile.imread(entry["path"])
        assert actual.dtype == np.uint16 and actual.shape == (48, 64, 3)
        if entry["shadow_lift_ev"]:
            # Independent equivalent reciprocal form of the rational lift.
            expected = baseline.astype(float)
            nonzero = expected > 0
            gain = 2 ** entry["shadow_lift_ev"]
            expected[nonzero] = 65535 / (1 + (65535 / expected[nonzero] - 1) / gain)
            np.testing.assert_array_equal(actual, np.rint(expected).astype(np.uint16))
            assert np.all(actual >= baseline)
            assert np.any(actual > baseline)
            np.testing.assert_array_equal(actual[baseline == 0], 0)
            np.testing.assert_array_equal(actual[baseline == 65535], 65535)
            shadows.append(actual)
        else:
            np.testing.assert_array_equal(actual, baseline)
    assert np.any(shadows[1] > shadows[0])
    assert np.all(shadows[1] >= shadows[0])
    assert source.read_bytes() == original
    assert report["highlight_recovery_accuracy"] == report["shadow_radiance_accuracy"] == "not measured"
    with pytest.raises(FileExistsError):
        develop_tone_sweep(source, tmp_path / "tone")
    assert len(list((tmp_path / "tone").iterdir())) == 4
    with pytest.raises(ValueError, match="memory budget"):
        develop_tone_sweep(source, tmp_path / "budget", max_working_bytes=1)
    assert not (tmp_path / "budget").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_exposure_ev": True},
        {"baseline_exposure_ev": float("nan")},
        {"baseline_exposure_ev": 17},
        {"max_working_bytes": False},
        {"shadow_lifts_ev": (1,)},
        {"shadow_lifts_ev": (2, 1)},
        {"shadow_lifts_ev": (0, 1)},
        {"shadow_lifts_ev": (1, 5)},
        {"shadow_lifts_ev": (True, 2)},
        {"shadow_lifts_ev": (1, float("nan"))},
    ],
)
def test_tone_invalid_settings_before_io(tmp_path, kwargs):
    from prd_v5.roundtrip import develop_tone_sweep

    with pytest.raises(ValueError):
        develop_tone_sweep(tmp_path / "absent", tmp_path / "tone", **kwargs)
    assert not (tmp_path / "tone").exists()


def test_tone_failure_cleanup(tmp_path, monkeypatch):
    from prd_v5.roundtrip import develop_tone_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    write_dng(source, np.full((48, 64), 2500, np.uint16), CameraHypothesis(), plan, AnalyticISP())
    original_write = tifffile.imwrite
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("fixture write failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imwrite", fail_second)
    with pytest.raises(OSError, match="fixture write failure"):
        develop_tone_sweep(source, tmp_path / "tone")
    assert not (tmp_path / "tone").exists()
    assert source.is_file()
