from __future__ import annotations

import pytest

from synthetic_raw_lab import subjects
from synthetic_raw_lab.sr_models import WeightsError


@pytest.fixture
def project(tmp_path):
    """An empty project root with the standard folders."""
    for name in ("inputs", "outputs", "reports"):
        (tmp_path / name).mkdir()
    return tmp_path


@pytest.fixture(autouse=True)
def no_face_model_download(monkeypatch, tmp_path_factory):
    """Tests never fetch the YuNet model; face detection falls back unless a test provides a detector."""

    def refuse(url, *args, **kwargs):
        raise WeightsError(f"downloads are disabled in tests ({url})")

    monkeypatch.setattr(subjects, "fetch_verified", refuse)
    monkeypatch.setattr(subjects, "default_model_dir", lambda: tmp_path_factory.mktemp("face_model"))
