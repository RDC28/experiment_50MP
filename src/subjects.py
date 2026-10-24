"""Main-subject analysis for evaluation crops and the final-output rule (PRD sections 10.2, 11.2 and 26.2).

Faces are found on a copy reduced to at most ``FACE_DETECTION_MAX_EDGE`` pixels, by the first detector this
OpenCV offers:

1. YuNet (``cv2.FaceDetectorYN``, in OpenCV 4.8 and later, including the 5.x that Colab ships). Its
   232,589-byte ONNX model (MIT licence) is not part of opencv-python, so it is downloaded once from a pinned
   opencv_zoo revision into the weights folder and its SHA-256 is checked before every use.
2. The frontal-face Haar cascade bundled with opencv-python 4.x (OpenCV 5 dropped it).

Without either, or when no face is found, a centre of interest stands in for the subject: spectral-residual
saliency (Hou and Zhang, 2007) weighted by a centre prior.

Detections only frame review crops and choose the final master. They are uncalibrated inference that produces
no image data: never identity evidence, and a missed face or a false detection is possible, so every report
records what was found and how.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .quality_metrics import luma
from .sr_models import WeightsError, fetch_verified, sha256_file

FACE_DETECTION_MAX_EDGE = 2048
_YUNET_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
YUNET = {
    "name": "OpenCV YuNet face detector, March 2023",
    "file": "face_detection_yunet_2023mar.onnx",
    "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{_YUNET_REVISION}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "size_bytes": 232_589,
    "license": "MIT (Copyright (c) 2020 Shiqi Yu), opencv_zoo revision " + _YUNET_REVISION[:12],
}
YUNET_SCORE_THRESHOLD = 0.9  # the opencv_zoo default; uncalibrated, never an identity probability
HAAR_CASCADE = "haarcascade_frontalface_default.xml"
HAAR_PARAMETERS = {"scaleFactor": 1.1, "minNeighbors": 6, "minSize": (24, 24)}
# PRD 11.2 tier 1 (high-information face), provisional until calibrated: the shortest side of the face box
# in source pixels. The same starting value as the Phase 3 face tiers.
TIER1_MIN_FACE_PX = 96
SALIENCY_SIZE = 96


@dataclass(frozen=True)
class Face:
    """A detected face box in source pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class SubjectAnalysis:
    """What the source shows as its main subject: faces (largest first) or a centre of interest."""

    width: int
    height: int
    faces: tuple[Face, ...]
    detector: str
    note: str
    center_of_interest: tuple[int, int] | None = None
    interest_basis: str = ""

    @property
    def primary_face(self) -> Face | None:
        return self.faces[0] if self.faces else None

    @property
    def high_information_face(self) -> bool:
        """True when the primary face reaches the provisional PRD 11.2 tier 1 size."""
        face = self.primary_face
        return face is not None and min(face.width, face.height) >= TIER1_MIN_FACE_PX

    def to_dict(self) -> dict:
        face = self.primary_face
        return {
            "classification": "Measurement (face detection; produces no image data, never identity evidence)",
            "model": dict(YUNET) if self.detector.startswith("opencv_yunet") else None,
            "detector": self.detector,
            "note": self.note,
            "faces": [f.to_dict() for f in self.faces],
            "primary_face": face.to_dict() if face else None,
            "primary_face_area_fraction": round(face.area / (self.width * self.height), 6) if face else None,
            "high_information_face": self.high_information_face,
            "tier1_min_face_px": TIER1_MIN_FACE_PX,
            "center_of_interest": list(self.center_of_interest) if self.center_of_interest else None,
            "interest_basis": self.interest_basis,
        }


def analyse(encoded: np.ndarray, weights_dir: str | Path | None = None, *, download: bool = True) -> SubjectAnalysis:
    """Find the main subject of an sRGB-encoded float image (H, W, 3) in [0, 1].

    ``weights_dir`` caches the YuNet model (default ``$SYNTHETIC_RAW_LAB_WEIGHTS`` or
    ``~/.cache/synthetic_raw_lab``); ``download`` allows fetching it when it is not cached.
    """
    height, width = encoded.shape[:2]
    scale = min(1.0, FACE_DETECTION_MAX_EDGE / max(width, height))
    small = encoded
    if scale < 1:
        small = cv2.resize(
            encoded.astype(np.float32), (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    small8 = np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)
    faces, detector, note = detect_faces(small8, scale, (width, height), weights_dir, download)
    point, basis = (None, "")
    if not faces:
        point, basis = center_of_interest(luma(small.astype(np.float32)), (width, height))
    return SubjectAnalysis(width, height, tuple(faces), detector, note, point, basis)


def default_model_dir() -> Path:
    return Path(os.environ.get("SYNTHETIC_RAW_LAB_WEIGHTS") or Path.home() / ".cache" / "synthetic_raw_lab")


def yunet_model(weights_dir: str | Path | None = None, *, download: bool = True) -> Path:
    """The verified YuNet model file, downloaded when missing; raises ``WeightsError`` when unavailable."""
    path = Path(weights_dir or default_model_dir()) / YUNET["file"]
    if path.is_file() and sha256_file(path) == YUNET["sha256"]:
        return path
    if not download:
        raise WeightsError(f"{path} is missing or corrupt and downloading is disabled")
    path.unlink(missing_ok=True)
    return fetch_verified(YUNET["url"], YUNET["sha256"], path, timeout=30)


def haar_cascade_path() -> Path | None:
    """The bundled frontal-face cascade, or None when this OpenCV build does not ship it."""
    directory = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not directory or not hasattr(cv2, "CascadeClassifier"):
        return None
    path = Path(directory) / HAAR_CASCADE
    return path if path.is_file() else None


def detect_faces(
    rgb8: np.ndarray,
    scale: float,
    size: tuple[int, int],
    weights_dir: str | Path | None = None,
    download: bool = True,
) -> tuple[list[Face], str, str]:
    """Faces in a reduced uint8 RGB copy (``scale`` times the source ``size``), as source boxes, largest first."""
    reasons = []
    if hasattr(cv2, "FaceDetectorYN"):
        try:
            model = yunet_model(weights_dir, download=download)
            height, width = rgb8.shape[:2]
            detector = cv2.FaceDetectorYN.create(str(model), "", (width, height), YUNET_SCORE_THRESHOLD, 0.3, 5000)
            _, found = detector.detect(cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
            rows = np.zeros((0, 15)) if found is None else np.asarray(found, dtype=np.float64)
            faces = _source_boxes(rows[rows[:, 14] >= YUNET_SCORE_THRESHOLD, :4], scale, size)
            note = (
                f"YuNet ({YUNET['file']}, SHA-256 verified) in OpenCV {cv2.__version__}, score at least "
                f"{YUNET_SCORE_THRESHOLD}, on a copy reduced by {scale:.3f}: {len(faces)} found"
            )
            return faces, "opencv_yunet_2023mar", note
        except (WeightsError, cv2.error) as error:
            reasons.append(f"YuNet unavailable ({error})")
    else:
        reasons.append(f"OpenCV {cv2.__version__} has no FaceDetectorYN")
    path = haar_cascade_path()
    if path is not None:
        cascade = cv2.CascadeClassifier(str(path))
        if not cascade.empty():
            gray = cv2.equalizeHist(cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY))
            boxes = np.asarray(cascade.detectMultiScale(gray, **HAAR_PARAMETERS), dtype=np.float64).reshape(-1, 4)
            faces = _source_boxes(boxes, scale, size)
            minimum = round(HAAR_PARAMETERS["minSize"][0] / scale)
            note = "; ".join(
                reasons
                + [
                    f"OpenCV {cv2.__version__} frontal-face Haar cascade on a copy reduced by {scale:.3f} "
                    f"(faces from about {minimum} source px): {len(faces)} found"
                ]
            )
            return faces, f"opencv_haar_{Path(HAAR_CASCADE).stem}", note
        reasons.append(f"{path} could not be loaded")
    else:
        reasons.append(f"this OpenCV build does not ship {HAAR_CASCADE}")
    return [], "none", "no face detector: " + "; ".join(reasons)


def _source_boxes(boxes: np.ndarray, scale: float, size: tuple[int, int]) -> list[Face]:
    width, height = size
    faces = []
    for x, y, w, h in np.asarray(boxes, dtype=np.float64).reshape(-1, 4) / scale:
        x0, y0 = max(0, round(x)), max(0, round(y))
        x1, y1 = min(width, round(x + w)), min(height, round(y + h))
        if x1 > x0 and y1 > y0:
            faces.append(Face(x0, y0, x1 - x0, y1 - y0))
    faces.sort(key=lambda f: (-f.area, f.y, f.x))
    return faces


def center_of_interest(y_luma: np.ndarray, size: tuple[int, int] | None = None) -> tuple[tuple[int, int] | None, str]:
    """Peak of spectral-residual saliency times a centre prior, in pixels of ``size`` (default: the luma image);
    None on flat images."""
    luma_height, luma_width = y_luma.shape
    width, height = size or (luma_width, luma_height)
    scale = SALIENCY_SIZE / max(luma_width, luma_height)
    grid = (max(8, round(luma_width * scale)), max(8, round(luma_height * scale)))
    small = cv2.resize(y_luma.astype(np.float32), grid, interpolation=cv2.INTER_AREA)
    if float(small.std()) < 1e-4:
        return None, "flat image: no centre of interest"
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-9)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3), borderType=cv2.BORDER_REPLICATE)
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))) ** 2
    saliency = cv2.GaussianBlur(saliency, (0, 0), 0.04 * max(grid))
    ys, xs = np.mgrid[0 : grid[1], 0 : grid[0]]
    prior = np.exp(-(((xs + 0.5) / grid[0] - 0.5) ** 2 + ((ys + 0.5) / grid[1] - 0.5) ** 2) / (2 * 0.3**2))
    y, x = np.unravel_index(np.argmax(saliency * prior), saliency.shape)
    point = (min(width - 1, round((x + 0.5) / grid[0] * width)), min(height - 1, round((y + 0.5) / grid[1] * height)))
    return point, "spectral-residual saliency peak weighted by a centre prior (no face detected)"
