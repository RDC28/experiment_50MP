"""Loader contract fixtures with fake torch; no HDRUNet weights or model are executed."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from prd_v5 import hdrunet
from prd_v5.analytic import AnalyticISP
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.learned import apply_residual


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, device):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeModel:
    def __init__(self, delta=0.0):
        self.delta = delta
        self.loaded = None
        self.shapes = []

    def load_state_dict(self, state, strict):
        assert strict is True
        self.loaded = state

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pair):
        image, cond = pair
        assert image is cond
        self.shapes.append(image.array.shape)
        return FakeTensor(image.array + self.delta)


def fake_torch(state):
    calls = []

    def load(path, map_location, weights_only):
        calls.append((map_location, weights_only))
        return state

    return SimpleNamespace(
        load=load,
        from_numpy=FakeTensor,
        no_grad=lambda: __import__("contextlib").nullcontext(),
        calls=calls,
    )


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "fake.pth"
    path.write_bytes(b"not a real checkpoint")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_digest_mismatch_refuses_before_unpickling(checkpoint):
    path, _ = checkpoint
    torch = fake_torch({})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        hdrunet.load_hdrunet_adapter(path, "0" * 64, CameraHypothesis(), torch_module=torch)
    assert torch.calls == []


def test_argument_validation(checkpoint):
    path, digest = checkpoint
    camera = CameraHypothesis()
    with pytest.raises(ValueError, match="multiples"):
        hdrunet.load_hdrunet_adapter(path, digest, camera, halo=130, torch_module=fake_torch({}))
    with pytest.raises(ValueError, match="64 hex"):
        hdrunet.load_hdrunet_adapter(path, "abc", camera, torch_module=fake_torch({}))
    with pytest.raises(ValueError, match="max_delta"):
        hdrunet.load_hdrunet_adapter(path, digest, camera, max_delta=0, torch_module=fake_torch({}))


def test_strict_load_strips_module_prefix_and_runs_bounded_residual(checkpoint, tmp_path, monkeypatch):
    path, digest = checkpoint
    camera = CameraHypothesis()
    model = FakeModel(delta=0.1)
    monkeypatch.setattr(hdrunet, "build_hdrunet", lambda torch: model)
    torch = fake_torch({"module.conv_first.weight": 1})
    adapter = hdrunet.load_hdrunet_adapter(
        path,
        digest.upper(),
        camera,
        licence_accepted=True,
        experimental_accepted=True,
        device="cpu",
        tile=4,
        halo=4,
        torch_module=torch,
    )
    assert torch.calls == [("cpu", True)]
    assert model.loaded == {"conv_first.weight": 1}
    rgb = np.full((6, 7, 3), 0.4, dtype=np.float32)
    clean = np.zeros_like(rgb)
    support = np.ones((6, 7), dtype=bool)
    destination = np.memmap(tmp_path / "out", mode="w+", dtype="float32", shape=rgb.shape)
    report = apply_residual(rgb, clean, support, destination, camera, adapter, isp=AnalyticISP(), tile=4, halo=4)
    assert report["status"] == "executed experimental residual"
    # every inference tile is padded to a multiple of the network stride
    assert all(shape[2] % 4 == 0 and shape[3] % 4 == 0 for shape in model.shapes)
    # +0.1 at 0.4 in the gamma-2.24 HDR domain is a linear change of about 0.083, so confidence ~0.67 passes
    delta = 0.5**2.24 - 0.4**2.24
    np.testing.assert_allclose(destination, delta * (1 - delta / 0.25), rtol=1e-4)


def test_large_change_is_low_confidence(checkpoint):
    path, digest = checkpoint
    camera = CameraHypothesis()
    predict = hdrunet.make_predictor(FakeModel(delta=0.5), fake_torch({}), device="cpu", max_delta=0.25)
    residual, confidence = predict(np.full((4, 4, 3), 0.3, dtype=np.float32), None, (0, 0), camera)
    assert residual.shape == (4, 4, 3) and (confidence == 0).all()


def test_nonfinite_output_rejected():
    predict = hdrunet.make_predictor(FakeModel(delta=np.nan), fake_torch({}), device="cpu", max_delta=0.25)
    with pytest.raises(ValueError, match="nonfinite"):
        predict(np.zeros((4, 4, 3), dtype=np.float32), None, (0, 0), CameraHypothesis())


def test_residual_is_normalized_not_electrons():
    camera = CameraHypothesis()
    model = FakeModel(delta=0.1)
    predict = hdrunet.make_predictor(model, fake_torch({}), device="cpu", max_delta=0.25)
    encoded = np.full((4, 4, 3), 0.4, dtype=np.float32)
    residual, _ = predict(encoded, None, (0, 0), camera)
    delta_linear = 0.5**2.24 - 0.4**2.24
    # same order as the encoded delta, not scaled by saturation_e
    assert camera.saturation_e > 100
    np.testing.assert_allclose(residual, delta_linear, rtol=1e-4)
    assert float(residual.max()) < 1.0


def test_output_above_clip_level_is_kept_as_hdr_radiance():
    predict = hdrunet.make_predictor(FakeModel(delta=0.05), fake_torch({}), device="cpu", max_delta=0.25)
    encoded = np.ones((4, 4, 3), dtype=np.float32)
    residual, confidence = predict(encoded, None, (0, 0), CameraHypothesis())
    # 1.05 in the gamma-2.24 output domain is linear radiance above the clip level, not clipped to it
    np.testing.assert_allclose(residual, 1.05**2.24 - 1.0, rtol=1e-4)
    np.testing.assert_allclose(confidence, 1 - (1.05**2.24 - 1.0) / 0.25, rtol=1e-4)
