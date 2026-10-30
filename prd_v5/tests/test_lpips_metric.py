"""Fixture tests for LPIPS tiling, contract and loader wiring (stand-in distance, no model)."""

import json
import sys
import types

import numpy as np
import pytest
import tifffile

from prd_v5.lpips_metric import MAX_PATCHES, load_lpips, lpips_distance

CONTRACT = {
    "camera_profile": "fixture camera / profile",
    "trusted_development": "fixture development",
    "alignment": "identical native coordinates",
    "colour_space": "sRGB_D65_2deg",
}


def tiff(tmp_path, name, value, shape=(130, 200, 3)):
    path = tmp_path / name
    tifffile.imwrite(path, np.full(shape, value, np.uint16), photometric="rgb")
    return path


def mean_abs(a, b):
    return np.abs(a - b).mean(axis=(1, 2, 3))


def test_missing_truth_never_loads_network():
    def boom(*_):
        raise AssertionError

    assert lpips_distance("absent", None, distance=boom)["status"] == "not measured"
    assert lpips_distance("absent", "absent", distance=boom)["status"] == "not measured"


def test_patch_tiling_scale_and_json(tmp_path):
    black, white = tiff(tmp_path, "b.tif", 0), tiff(tmp_path, "w.tif", 65535)
    seen = []

    def distance(a, b):
        seen.append(a.shape)
        assert a.dtype == np.float32 and a.min() >= -1 and a.max() <= 1
        return mean_abs(a, b)

    report = lpips_distance(black, white, calibration=CONTRACT, patch=64, distance=distance)
    assert report["patches"] == 2 * 3 == len(seen) and set(seen) == {(1, 3, 64, 64)}
    assert report["mean_lpips"] == pytest.approx(2.0)
    assert report["uncovered_pixels"] == 130 * 200 - 6 * 64 * 64
    assert lpips_distance(white, white, calibration=CONTRACT, patch=64, distance=distance)["max_lpips"] == 0
    json.dumps(report, allow_nan=False)


def test_region_and_limits(tmp_path):
    a = tiff(tmp_path, "a.tif", 0)
    assert (
        lpips_distance(a, a, calibration=CONTRACT, region=(0, 0, 10, 10), distance=mean_abs)["status"] == "not measured"
    )
    with pytest.raises(ValueError, match="exceeds"):
        lpips_distance(a, a, calibration=CONTRACT, region=(0, 0, 999, 10), distance=mean_abs)
    with pytest.raises(ValueError, match="patch"):
        lpips_distance(a, a, calibration=CONTRACT, patch=8)
    with pytest.raises(ValueError, match="calibration"):
        lpips_distance(a, a, calibration={})
    with pytest.raises(ValueError, match="sRGB"):
        lpips_distance(a, a, calibration={**CONTRACT, "colour_space": "x"})
    b = tiff(tmp_path, "b.tif", 0, (64, 64, 3))
    assert lpips_distance(a, b, calibration=CONTRACT, distance=mean_abs)["reason"].startswith("Unequal")
    assert MAX_PATCHES == 4096


def test_loader_uses_official_package_shape(monkeypatch):
    calls = {}

    class Model:
        def __init__(self, net, version):
            calls["init"] = (net, version)

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            return self

        def __call__(self, a, b):
            return types.SimpleNamespace(
                reshape=lambda *_: types.SimpleNamespace(
                    cpu=lambda: types.SimpleNamespace(numpy=lambda: np.array([0.25], np.float32))
                )
            )

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda *a: None)
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.from_numpy = lambda x: types.SimpleNamespace(to=lambda d: x)
    mod = types.ModuleType("lpips")
    mod.LPIPS = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "lpips", mod)
    fn = load_lpips("vgg", "cuda")
    assert calls["init"] == ("vgg", "0.1") and calls["device"] == "cuda"
    assert fn(np.zeros((1, 3, 64, 64), np.float32), np.zeros((1, 3, 64, 64), np.float32))[0] == 0.25
    with pytest.raises(ValueError, match="net"):
        load_lpips("resnet")
