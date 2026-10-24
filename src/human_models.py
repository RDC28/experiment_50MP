"""Verified runtime inference artifacts for Notebook 01; never training assets."""

import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

YUNET_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{YUNET_REVISION}/models/face_detection_yunet/{YUNET_FILENAME}"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_SIZE = 232589


def yunet_model_metadata() -> dict:
    """Expected artifact provenance, including when acquisition is unavailable."""
    return {
        "name": "OpenCV YuNet March 2023",
        "classification": "Inference",
        "source_url": YUNET_URL,
        "revision": YUNET_REVISION,
        "file_name": YUNET_FILENAME,
        "size_bytes": YUNET_SIZE,
        "sha256": YUNET_SHA256,
        "license": "MIT",
        "license_text_location": (
            f"https://raw.githubusercontent.com/opencv/opencv_zoo/{YUNET_REVISION}/models/face_detection_yunet/LICENSE"
        ),
        "commercial_use": "Permitted, including sale; retain copyright and permission notice; no warranty.",
    }


def ensure_yunet_model(cache_dir: Path | None = None) -> Path:
    """Return verified weights or raise with the precise acquisition failure.

    Callers must record failure and continue with manual observations. Existing
    corrupt caches are rejected, never silently trusted or replaced. Downloads
    are verified before atomic publication; concurrent readers see whole files.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / "weights"
    path = cache_dir / YUNET_FILENAME

    def verify(data: bytes) -> None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != YUNET_SHA256 or len(data) != YUNET_SIZE:
            raise RuntimeError(f"YuNet checksum/size mismatch: {len(data)} bytes, SHA-256 {digest}")

    try:
        if path.exists():
            verify(path.read_bytes())
            return path
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urlopen(YUNET_URL, timeout=30) as response:
            data = response.read(YUNET_SIZE + 1)
        verify(data)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path
    except (OSError, ValueError) as error:
        raise RuntimeError(f"YuNet runtime weights unavailable: {error}") from error
