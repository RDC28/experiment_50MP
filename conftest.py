"""Adds video_v1/tests to a plain `pytest` run, next to the testpaths in pyproject.toml.

They cannot be listed in testpaths: prd_v6/tests/test_notebook.py requires pyproject.toml to be unchanged since
the v6 notebook's REPO_REF, and re-pinning that notebook is out of scope for video_v1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

EXTRA_TESTPATHS = ("video_v1/tests",)


def pytest_configure(config: pytest.Config) -> None:
    if config.args_source == pytest.Config.ArgsSource.TESTPATHS:
        config.args.extend(str(Path(config.rootpath, path)) for path in EXTRA_TESTPATHS)
