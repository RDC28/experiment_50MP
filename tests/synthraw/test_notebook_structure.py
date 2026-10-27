"""Notebook structure plus bounded cell executions with Colab and pip mocked; never runs reconstruction."""

import re
import sys
import types
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "synthetic_camera_raw_50mp_colab.ipynb"
SECTIONS = [
    "Environment + reproducibility",
    "Input audit",
    "Profile selection",
    "Target geometry",
    "Source diagnostics",
    "SR reconstruction",
    "HDR / scene-linear reconstruction",
    "Camera geometry + exposure inference",
    "Inverse ISP / camera colour",
    "Optical forward model",
    "Sensor/electron simulation",
    "CFA construction",
    "ADC quantisation",
    "Metadata construction",
    "DNG export",
    "RAW round-trip development",
    "16-bit TIFF export",
    "Validation suite",
    "2K/4K comparison",
    "Provenance + report packaging",
]


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_schema_and_order(notebook):
    nbformat.validate(notebook)
    headers = [
        line.removeprefix("## ")
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    assert headers == SECTIONS


def _run_install_cell(source, ref, failing_refs, monkeypatch, *, colab=False, apt_exit=0, apt_calls=None):
    """Execute the setup cell with REPO_REF replaced and pip, apt and Colab mocked; returns (namespace, refs).

    ExifTool starts missing. ``colab`` makes ``import google.colab`` succeed; apt commands are appended
    to ``apt_calls`` and exit with ``apt_exit``.
    """
    calls = []
    apt_calls = [] if apt_calls is None else apt_calls

    def fake_call(command):
        if command[0] == "apt-get":
            apt_calls.append(command)
            return apt_exit
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        url, target = command[4].rsplit("@", 1)
        assert url == "git+https://github.com/RDC28/experiment_50MP.git" and len(command) == 5
        calls.append(target)
        return 1 if target in failing_refs else 0

    google = types.ModuleType("google")
    google.colab = types.ModuleType("google.colab")
    monkeypatch.setitem(sys.modules, "google", google if colab else None)
    monkeypatch.setitem(sys.modules, "google.colab", google.colab if colab else None)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("subprocess.call", fake_call)
    monkeypatch.setattr("subprocess.check_call", lambda *args, **kwargs: pytest.fail("use subprocess.call"))
    patched = re.sub(r'^REPO_REF = ".*"$', f"REPO_REF = {ref!r}", source, count=1, flags=re.MULTILINE)
    assert patched != source or ref == "fm/exp50mp-v3-synthetic-raw"
    namespace = {}
    exec(compile(patched, "setup_cell", "exec"), namespace)
    return namespace, calls


@pytest.fixture
def install_source(notebook):
    return next(cell.source for cell in notebook.cells if cell.cell_type == "code")


def test_install_cell_tries_repo_ref_first(install_source, monkeypatch):
    namespace, calls = _run_install_cell(install_source, "fm/exp50mp-v3-synthetic-raw", set(), monkeypatch)
    assert calls == ["fm/exp50mp-v3-synthetic-raw"]
    assert namespace["INSTALLED_REF"] == "fm/exp50mp-v3-synthetic-raw"


def test_install_cell_falls_back_to_main_when_branch_is_merged(install_source, monkeypatch):
    with pytest.warns(UserWarning, match="merged commit SHA"):
        namespace, calls = _run_install_cell(
            install_source, "fm/exp50mp-v3-synthetic-raw", {"fm/exp50mp-v3-synthetic-raw"}, monkeypatch
        )
    assert calls == ["fm/exp50mp-v3-synthetic-raw", "main"]
    assert namespace["INSTALLED_REF"] == "main"


def test_install_cell_pinned_sha_never_falls_back(install_source, monkeypatch):
    sha = "0123456789abcdef0123456789abcdef01234567"
    namespace, calls = _run_install_cell(install_source, sha, set(), monkeypatch)
    assert calls == [sha] and namespace["INSTALLED_REF"] == sha
    with pytest.raises(RuntimeError, match="check REPO_REF"):
        _run_install_cell(install_source, sha, {sha}, monkeypatch)


def test_install_cell_skips_exiftool_outside_colab(install_source, monkeypatch, capsys):
    apt_calls = []
    _run_install_cell(install_source, "fm/exp50mp-v3-synthetic-raw", set(), monkeypatch, apt_calls=apt_calls)
    assert apt_calls == []
    assert "ExifTool: not installed" in capsys.readouterr().out


def test_install_cell_installs_exiftool_on_colab(install_source, monkeypatch):
    apt_calls = []
    _run_install_cell(
        install_source, "fm/exp50mp-v3-synthetic-raw", set(), monkeypatch, colab=True, apt_calls=apt_calls
    )
    assert apt_calls == [["apt-get", "-qq", "update"], ["apt-get", "-qq", "install", "-y", "libimage-exiftool-perl"]]


def test_install_cell_continues_with_the_seed_when_exiftool_cannot_be_installed(install_source, monkeypatch):
    apt_calls = []
    with pytest.warns(UserWarning, match="bundled seed"):
        namespace, calls = _run_install_cell(
            install_source,
            "fm/exp50mp-v3-synthetic-raw",
            set(),
            monkeypatch,
            colab=True,
            apt_exit=100,
            apt_calls=apt_calls,
        )
    assert apt_calls == [["apt-get", "-qq", "update"]]
    assert namespace["INSTALLED_REF"] == "fm/exp50mp-v3-synthetic-raw"


@pytest.mark.parametrize("ref", ["v1.0", "some-branch", "0123456", "0123456789abcdef0123456789abcdef0123456g"])
def test_install_cell_rejects_unpinned_refs(install_source, monkeypatch, ref):
    with pytest.raises(ValueError, match="40-character commit SHA"):
        _run_install_cell(install_source, ref, set(), monkeypatch)


def test_all_cells_are_unexecuted_valid_python(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
            compile(cell.source, str(NOTEBOOK), "exec")


def cell_by_id(notebook, cell_id):
    return next(cell for cell in notebook.cells if cell.get("id") == cell_id)


def raw_gate_cell(notebook):
    return cell_by_id(notebook, "89ff0cbb")


def test_input_audit_refuses_to_run_before_the_raw_gate(notebook):
    audit = cell_by_id(notebook, "c1b3f8df")
    for namespace in ({}, {"RAW_GATE_PASSED": False}):
        with pytest.raises(RuntimeError, match="mandatory tiny RAW gate"):
            exec(compile(audit.source, str(NOTEBOOK), "exec"), namespace)
        assert "pipeline" not in namespace


def test_tiny_raw_gate_cell_with_real_libraw(notebook):
    # Execute only the bounded fixture cell, never the notebook or Colab setup.
    from synthraw.config import ExperimentConfig

    namespace = {"ExperimentConfig": ExperimentConfig}
    exec(compile(raw_gate_cell(notebook).source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["RAW_GATE_PASSED"] is True


def test_tiny_raw_gate_failure_blocks_progress_and_cleans_up(notebook, monkeypatch):
    from synthraw.config import ExperimentConfig
    from synthraw.libraw_validator import HardValidationError, LibRawValidator

    paths = []

    def fail(self, path, *args, **kwargs):
        paths.append(path)
        raise HardValidationError("decoder rejected fixture")

    monkeypatch.setattr(LibRawValidator, "validate", fail)
    namespace = {"ExperimentConfig": ExperimentConfig, "RAW_GATE_PASSED": True}
    with pytest.raises(HardValidationError, match="decoder rejected fixture"):
        exec(compile(raw_gate_cell(notebook).source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["RAW_GATE_PASSED"] is False
    assert len(paths) == 1
    assert not paths[0].parent.exists()
