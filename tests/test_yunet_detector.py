"""Offline source-scale YuNet adapter contracts."""

import numpy as np
import pytest

from synthetic_raw_lab import face_preservation as faces
from synthetic_raw_lab import human_models


@pytest.mark.parametrize("rows", [None, np.array([[-2, 3, 12, 40] + [0] * 10 + [0.95]])])
def test_yunet_source_size_bgr_and_clipped_boxes(monkeypatch, tmp_path, rows):
    import cv2

    model = tmp_path / "verified.onnx"
    model.write_bytes(b"fake verified artifact")
    monkeypatch.setattr(human_models, "ensure_yunet_model", lambda: model)
    calls = []

    class FakeDetector:
        def detect(self, image):
            assert image.shape == (20, 30, 3)
            assert image.dtype == np.uint8
            assert image[0, 0].tolist() == [0, 0, 255]
            return 1, rows

    def create(*args):
        calls.append(args)
        return FakeDetector()

    monkeypatch.setattr(cv2, "FaceDetectorYN", type("Factory", (), {"create": staticmethod(create)}))
    image = np.zeros((20, 30, 3), np.float32)
    image[..., 0] = 1
    result = faces.YuNetFaceDetector()(image)
    assert calls == [(str(model), "", (30, 20), 0.9, 0.3, 5000)]
    assert result.method == "yunet"
    assert result.classification == "Inference"
    assert result.faces == (() if rows is None else (faces.FaceBox(0, 3, 10, 17, 0.95),))


@pytest.mark.parametrize("reason", ["checksum/size mismatch", "runtime weights unavailable: offline"])
def test_yunet_acquisition_failure_preserves_manual_boxes(monkeypatch, reason):
    def unavailable():
        raise RuntimeError(reason)

    monkeypatch.setattr(human_models, "ensure_yunet_model", unavailable)
    manual = faces.ManualFaceDetector((faces.FaceBox(1, 1, 5, 5),))
    result = faces.observe_faces(np.zeros((20, 30, 3), np.float32), manual, faces.YuNetFaceDetector())
    assert result.faces == manual.faces
    assert reason in result.description
    assert "manual boxes only" in result.description


def test_yunet_inference_failure_is_explicit(monkeypatch, tmp_path):
    import cv2

    monkeypatch.setattr(human_models, "ensure_yunet_model", lambda: tmp_path / "local.onnx")

    def broken(*args):
        raise cv2.error("unsupported model on this build")

    monkeypatch.setattr(cv2, "FaceDetectorYN", type("Factory", (), {"create": staticmethod(broken)}))
    result = faces.YuNetFaceDetector()(np.zeros((20, 30, 3), np.float32))
    assert not result.faces
    assert "unsupported model on this build" in result.description
    assert "manual boxes only" in result.description
