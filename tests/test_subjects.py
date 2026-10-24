import hashlib
from pathlib import Path

import numpy as np
import pytest

from synthetic_raw_lab import subjects
from synthetic_raw_lab.test_images import make_test_image


class FakeCascade:
    """Stands in for cv2.CascadeClassifier; returns boxes in the reduced detection image."""

    boxes = []
    seen = []

    def __init__(self, path):
        self.path = path

    def empty(self):
        return False

    def detectMultiScale(self, image, **kwargs):  # noqa: N802 - OpenCV's name
        FakeCascade.seen.append((image.shape, image.dtype, kwargs))
        return np.array(FakeCascade.boxes, dtype=np.int32).reshape(-1, 4)


@pytest.fixture
def fake_cascade(monkeypatch, tmp_path):
    cascade = tmp_path / subjects.HAAR_CASCADE
    cascade.write_text("<opencv_storage/>")
    monkeypatch.setattr(subjects, "haar_cascade_path", lambda: cascade)
    monkeypatch.setattr(subjects.cv2, "CascadeClassifier", FakeCascade, raising=False)  # absent in OpenCV 5
    FakeCascade.seen = []
    return FakeCascade


def blob_image(width, height, cx, cy, radius=12):
    yy, xx = np.mgrid[:height, :width]
    image = np.full((height, width, 3), 0.1, dtype=np.float32)
    image[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = 0.9
    return image


def test_faces_are_mapped_back_to_source_pixels_largest_first(fake_cascade):
    # A 4096 x 1024 source is reduced by 0.5 for detection (FACE_DETECTION_MAX_EDGE = 2048).
    fake_cascade.boxes = [(100, 50, 30, 30), (1000, 200, 120, 140)]
    analysis = subjects.analyse(np.full((1024, 4096, 3), 0.5, dtype=np.float32))
    shape, dtype, kwargs = fake_cascade.seen[0]
    assert shape == (512, 2048) and dtype == np.uint8 and kwargs == subjects.HAAR_PARAMETERS
    assert analysis.faces == (subjects.Face(2000, 400, 240, 280), subjects.Face(200, 100, 60, 60))
    assert analysis.detector == "opencv_haar_haarcascade_frontalface_default"
    assert "2 found" in analysis.note and "faces from about 48 source px" in analysis.note
    assert analysis.high_information_face and analysis.center_of_interest is None
    record = analysis.to_dict()
    assert record["primary_face"] == {"x": 2000, "y": 400, "width": 240, "height": 280}
    assert record["primary_face_area_fraction"] == round(240 * 280 / (4096 * 1024), 6)


def test_a_small_face_is_not_high_information(fake_cascade):
    fake_cascade.boxes = [(10, 10, subjects.TIER1_MIN_FACE_PX - 1, 200)]
    analysis = subjects.analyse(np.full((400, 600, 3), 0.5, dtype=np.float32))
    assert analysis.primary_face.width == subjects.TIER1_MIN_FACE_PX - 1
    assert not analysis.high_information_face


def test_boxes_are_clipped_to_the_image(fake_cascade):
    fake_cascade.boxes = [(550, 350, 80, 80)]
    analysis = subjects.analyse(np.full((400, 600, 3), 0.5, dtype=np.float32))
    assert analysis.faces == (subjects.Face(550, 350, 50, 50),)


def test_without_a_cascade_the_centre_of_interest_is_used(monkeypatch):
    monkeypatch.setattr(subjects, "haar_cascade_path", lambda: None)
    analysis = subjects.analyse(blob_image(640, 360, 420, 150))
    assert analysis.faces == () and analysis.detector == "none" and "does not ship" in analysis.note
    x, y = analysis.center_of_interest
    assert abs(x - 420) <= 30 and abs(y - 150) <= 30
    assert "saliency" in analysis.interest_basis


def test_a_flat_image_has_no_centre_of_interest(monkeypatch):
    monkeypatch.setattr(subjects, "haar_cascade_path", lambda: None)
    analysis = subjects.analyse(np.full((200, 300, 3), 0.4, dtype=np.float32))
    assert analysis.center_of_interest is None and "flat" in analysis.interest_basis


@pytest.mark.skipif(subjects.haar_cascade_path() is None, reason="this OpenCV build does not ship the Haar cascade")
def test_the_bundled_cascade_runs_on_a_face_free_image():
    analysis = subjects.analyse(make_test_image(640, 360).astype(np.float32) / 255)
    assert analysis.detector.startswith("opencv_haar_")
    assert all(f.x + f.width <= 640 and f.y + f.height <= 360 for f in analysis.faces)


class FakeYuNet:
    """Stands in for cv2.FaceDetectorYN: rows of x, y, w, h, ten landmark values and a score."""

    rows = []
    created = []

    @classmethod
    def create(cls, model, config, size, score, nms, top_k):
        cls.created.append((Path(model).name, size, score))
        return cls()

    def detect(self, bgr):
        FakeYuNet.image = bgr
        return 1, (np.array(FakeYuNet.rows, dtype=np.float32).reshape(-1, 15) if FakeYuNet.rows else None)


@pytest.fixture
def fake_yunet(monkeypatch, tmp_path):
    model = tmp_path / subjects.YUNET["file"]
    model.write_bytes(b"onnx")
    monkeypatch.setitem(subjects.YUNET, "sha256", hashlib.sha256(b"onnx").hexdigest())
    monkeypatch.setattr(subjects.cv2, "FaceDetectorYN", FakeYuNet, raising=False)
    FakeYuNet.rows, FakeYuNet.created = [], []
    return tmp_path


def yunet_row(x, y, w, h, score):
    return [x, y, w, h, *([0.0] * 10), score]


def test_yunet_is_preferred_and_low_scores_are_dropped(fake_yunet):
    FakeYuNet.rows = [yunet_row(500, 100, 200, 220, 0.97), yunet_row(10, 10, 300, 300, 0.5)]
    image = np.zeros((1024, 4096, 3), dtype=np.float32)
    image[..., 0] = 1.0  # red, to check the BGR order handed to OpenCV
    analysis = subjects.analyse(image, fake_yunet)
    assert FakeYuNet.created == [(subjects.YUNET["file"], (2048, 512), subjects.YUNET_SCORE_THRESHOLD)]
    assert FakeYuNet.image[0, 0].tolist() == [0, 0, 255]
    assert analysis.faces == (subjects.Face(1000, 200, 400, 440),)
    assert analysis.detector == "opencv_yunet_2023mar" and "SHA-256 verified" in analysis.note
    assert analysis.to_dict()["model"]["url"] == subjects.YUNET["url"]


def test_yunet_without_faces_falls_back_to_the_centre_of_interest(fake_yunet):
    analysis = subjects.analyse(blob_image(640, 360, 420, 150), fake_yunet)
    assert analysis.faces == () and analysis.detector == "opencv_yunet_2023mar"
    assert analysis.center_of_interest is not None


def test_the_yunet_model_is_downloaded_once_and_verified(monkeypatch, tmp_path):
    fetched = []

    def fetch(url, sha256, path, timeout):
        fetched.append(url)
        path.write_bytes(b"model")
        return path

    monkeypatch.setattr(subjects, "fetch_verified", fetch)
    monkeypatch.setitem(subjects.YUNET, "sha256", hashlib.sha256(b"model").hexdigest())
    (tmp_path / subjects.YUNET["file"]).write_bytes(b"corrupt")
    assert subjects.yunet_model(tmp_path).read_bytes() == b"model"
    assert subjects.yunet_model(tmp_path) == tmp_path / subjects.YUNET["file"]
    assert fetched == [subjects.YUNET["url"]]
    with pytest.raises(subjects.WeightsError, match="downloading is disabled"):
        subjects.yunet_model(tmp_path / "empty", download=False)


def test_an_unavailable_yunet_model_falls_back_and_says_why(monkeypatch, tmp_path):
    monkeypatch.setattr(subjects.cv2, "FaceDetectorYN", FakeYuNet, raising=False)
    monkeypatch.setattr(subjects, "haar_cascade_path", lambda: None)
    analysis = subjects.analyse(blob_image(640, 360, 300, 200), tmp_path)
    assert analysis.detector == "none" and "YuNet unavailable (downloads are disabled in tests" in analysis.note
