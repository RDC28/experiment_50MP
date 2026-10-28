"""Notebook structure plus bounded cell executions with Colab and pip mocked; never runs reconstruction."""

import os
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
    "Final image",
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


def _run_install_cell(source, monkeypatch, *, ref=None, failing_refs=(), colab=False, sr_modules=("torch", "spandrel")):
    """Execute the setup cell with pip and Colab mocked; returns (namespace, refs).

    With ``ref`` the cell's REPO_REF is replaced first; without it the cell runs as written.

    ExifTool starts missing. ``colab`` makes ``import google.colab`` succeed; ``sr_modules`` are then the
    learned SR packages that import, the others are missing.
    """
    calls = []

    def fake_call(command):
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        url, target = command[4].rsplit("@", 1)
        # The sr extra is what installs spandrel, without which SwinIR and Real-ESRGAN cannot load.
        assert url == "synthetic-raw-50mp-lab[sr] @ git+https://github.com/RDC28/experiment_50MP.git"
        assert len(command) == 5
        calls.append(target)
        return 1 if target in failing_refs else 0

    google = types.ModuleType("google")
    google.colab = types.ModuleType("google.colab")
    monkeypatch.setitem(sys.modules, "google", google if colab else None)
    monkeypatch.setitem(sys.modules, "google.colab", google.colab if colab else None)
    for module in ("torch", "spandrel"):
        monkeypatch.setitem(sys.modules, module, types.ModuleType(module) if module in sr_modules else None)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("subprocess.call", fake_call)
    monkeypatch.setattr("subprocess.check_call", lambda *args, **kwargs: pytest.fail("use subprocess.call"))
    if ref is not None:
        patched = re.sub(r'^REPO_REF = ".*"$', f"REPO_REF = {ref!r}", source, count=1, flags=re.MULTILINE)
        assert patched != source
        source = patched
    namespace = {}
    exec(compile(source, "setup_cell", "exec"), namespace)
    return namespace, calls


@pytest.fixture
def install_source(notebook):
    return next(cell.source for cell in notebook.cells if cell.cell_type == "code")


def test_install_cell_defaults_to_main(install_source, monkeypatch):
    namespace, calls = _run_install_cell(install_source, monkeypatch)
    assert calls == ["main"] and namespace["INSTALLED_REF"] == "main"
    with pytest.raises(RuntimeError, match="check REPO_REF"):
        _run_install_cell(install_source, monkeypatch, failing_refs={"main"})


def test_install_cell_pinned_sha_never_falls_back(install_source, monkeypatch):
    sha = "0123456789abcdef0123456789abcdef01234567"
    namespace, calls = _run_install_cell(install_source, monkeypatch, ref=sha)
    assert calls == [sha] and namespace["INSTALLED_REF"] == sha
    with pytest.raises(RuntimeError, match="check REPO_REF"):
        _run_install_cell(install_source, monkeypatch, ref=sha, failing_refs={sha})


@pytest.fixture
def exiftool_installs(monkeypatch):
    installs = []
    monkeypatch.setattr("synthraw.sony_registry.install_exiftool", installs.append)
    return installs


def test_install_cell_skips_exiftool_outside_colab(install_source, monkeypatch, capsys, exiftool_installs):
    _run_install_cell(install_source, monkeypatch, sr_modules=())
    assert exiftool_installs == []
    assert "ExifTool: not installed" in capsys.readouterr().out


def test_install_cell_installs_the_pinned_exiftool_on_colab(install_source, monkeypatch, exiftool_installs):
    namespace, calls = _run_install_cell(install_source, monkeypatch, colab=True)
    assert exiftool_installs == ["/content/exiftool"]
    assert calls == ["main"] and namespace["INSTALLED_REF"] == "main"


@pytest.mark.parametrize("missing", ["torch", "spandrel"])
def test_install_cell_fails_on_colab_without_a_learned_sr_package(install_source, monkeypatch, missing):
    present = tuple(module for module in ("torch", "spandrel") if module != missing)
    with pytest.raises(ImportError, match=missing):
        _run_install_cell(install_source, monkeypatch, colab=True, sr_modules=present)


def test_install_cell_continues_with_the_seed_when_exiftool_cannot_be_downloaded(install_source, monkeypatch, capsys):
    from synthraw.sony_registry import EXIFTOOL_UNAVAILABLE_ENV

    def offline(url, timeout):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    monkeypatch.setenv(EXIFTOOL_UNAVAILABLE_ENV, "")
    monkeypatch.delenv(EXIFTOOL_UNAVAILABLE_ENV)
    with pytest.warns(UserWarning, match="bundled seed"):
        namespace, _ = _run_install_cell(install_source, monkeypatch, colab=True)
    assert namespace["INSTALLED_REF"] == "main"
    assert "OSError: offline" in os.environ[EXIFTOOL_UNAVAILABLE_ENV]
    assert "ExifTool: not installed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "ref",
    ["v1.0", "some-branch", "fm/exp50mp-v3-synthetic-raw", "0123456", "0123456789abcdef0123456789abcdef0123456g"],
)
def test_install_cell_rejects_unpinned_refs(install_source, monkeypatch, ref):
    with pytest.raises(ValueError, match="40-character commit SHA"):
        _run_install_cell(install_source, monkeypatch, ref=ref)


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
