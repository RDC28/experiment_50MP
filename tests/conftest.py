from __future__ import annotations

import pytest


@pytest.fixture
def project(tmp_path):
    """An empty project root with the standard folders."""
    for name in ("inputs", "outputs", "reports"):
        (tmp_path / name).mkdir()
    return tmp_path
