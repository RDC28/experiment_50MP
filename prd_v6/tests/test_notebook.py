"""Notebook contract (section 9.2): two code cells, literal settings, one try/finally, pinned code."""

from __future__ import annotations

import ast
import builtins
import json
import re
import shutil
import subprocess
import sys
import traceback
import types
import zipfile
from pathlib import Path

import pytest

from prd_v6 import delivery

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "prd_v6" / "notebooks" / "synthetic_50mp_v6.ipynb"
SETTINGS = {
    "PRESET": "max",
    "SOURCE_4K_URL": None,
    "SOURCE_2K_URL": None,
    "ILLUMINANT_CCT_K": 4300,
    "SOURCE_ILLUMINANT_CCT_K": None,
    "REFERENCE_ARW_URL": None,
    "DOWNLOAD_ALL": False,
    "DRIVE_MIRROR": False,
}
# Changes under these paths change what the notebook runs, so REPO_REF must include them.
CODE_PATHS = ["prd_v6", "prd_v5", "src", "pyproject.toml", ":!prd_v6/notebooks", ":!prd_v6/tests", ":!*.md"]


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text())


def code_cells(notebook):
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def settings_of(notebook) -> dict:
    tree = ast.parse(code_cells(notebook)[0])
    out = {}
    for node in tree.body:
        assert isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant), ast.dump(node)
        out[node.targets[0].id] = node.value.value
    return out


def test_two_code_cells_with_no_outputs(notebook):
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 2
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in cells)


def test_settings_are_literal_max_defaults(notebook):
    settings = settings_of(notebook)
    assert {k: settings[k] for k in SETTINGS} == SETTINGS
    assert re.fullmatch(r"[0-9a-f]{40}", settings["REPO_REF"])


class _ContentRoot(ast.NodeTransformer):
    """Moves the driver's /content paths under a test folder; nothing else in the cell changes."""

    def __init__(self, root: Path):
        self.root = root

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("/content/"):
            return ast.copy_location(ast.Constant(str(self.root) + node.value[len("/content") :]), node)
        return node


def run_driver_cell(notebook, root: Path, monkeypatch, stand_ins: dict):
    """Run the settings cell, then the driver cell as Colab would, with `stand_ins` for the named functions."""
    calls = {"download": [], "upload": [], "input": []}

    def forbidden(name):
        def call(*args, **kwargs):
            calls[name].append(args)
            raise AssertionError(f"the driver cell called {name}")

        return call

    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(
        download=lambda path: calls["download"].append(Path(path)), upload=forbidden("upload")
    )
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(builtins, "input", forbidden("input"))
    tree = _ContentRoot(root).visit(ast.parse(code_cells(notebook)[1]))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in stand_ins)]
    namespace: dict = {"__name__": "__main__"}
    exec(code_cells(notebook)[0], namespace)  # noqa: S102
    namespace.update(stand_ins)
    exec(compile(ast.fix_missing_locations(tree), "driver", "exec"), namespace)  # noqa: S102
    return namespace, calls


def _setup_fails():
    raise RuntimeError("clone failed")


def _benchmark_fails(run_dir, settings):
    delivery.mark_step_started(run_dir, 4)
    try:
        raise RuntimeError("benchmark exploded")
    except RuntimeError as error:
        delivery.mark_step_finished(run_dir, 4, error)
        raise


@pytest.mark.parametrize(
    ("setup", "run", "reason"),
    [
        (_setup_fails, None, "step 2 (Setup) raised RuntimeError: clone failed"),
        (lambda: None, _benchmark_fails, "step 4 (Benchmark) raised RuntimeError: benchmark exploded"),
    ],
    ids=["setup", "benchmark"],
)
def test_driver_cell_finalizes_and_downloads_an_invalid_run(
    notebook, tmp_path, monkeypatch, capsys, setup, run, reason
):
    from prd_v6 import driver

    monkeypatch.setattr(driver, "run", run or (lambda *_: pytest.fail("driver.run ran after a failed setup")))
    ns, calls = run_driver_cell(notebook, tmp_path, monkeypatch, {"_setup": setup})
    run_dir = ns["RUN_DIR"]
    assert run_dir.parent == tmp_path / "exp50mp_v6"
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["status"] == "INVALID" and verdict["final"]
    assert verdict["text"] == f"INVALID: {reason}"
    summary = delivery.summary_path(run_dir)
    assert calls["download"] == [summary] and ns["ARCHIVES"] == [summary]
    with zipfile.ZipFile(summary) as archive:
        assert json.loads(archive.read("verdict.json"))["text"] == verdict["text"]
    assert calls["input"] == calls["upload"] == []
    assert f"PRD v6 RUN {ns['RUN_ID']}: INVALID" in capsys.readouterr().out


def test_driver_cell_falls_back_when_packaging_fails(notebook, tmp_path, monkeypatch, capsys):
    def packaging_fails(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(delivery, "finalize_run", packaging_fails)
    ns, calls = run_driver_cell(notebook, tmp_path, monkeypatch, {"_setup": _setup_fails})
    fallback = tmp_path / "exp50mp_v6" / f"{ns['RUN_ID']}_fallback.zip"
    assert calls["download"] == [fallback]
    with zipfile.ZipFile(fallback) as archive:
        state = json.loads(archive.read(f"{ns['RUN_ID']}/state.json"))
        assert f"{ns['RUN_ID']}/config.json" in archive.namelist()
    assert [(r["step"], r["status"], r["error"]) for r in state["steps"]] == [
        (2, "failed", "RuntimeError: clone failed")
    ]
    assert calls["input"] == calls["upload"] == []
    out = capsys.readouterr().out
    assert f"PRD v6 RUN {ns['RUN_ID']}: INVALID" in out
    assert "INVALID: RuntimeError: clone failed; packaging: OSError: disk full" in out


def test_colab_runtime_and_tier_are_stated(notebook):
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    intro = "".join(notebook["cells"][0]["source"])
    for phrase in ("A100", "High-RAM", "Run all", "95%", "16-bit", "Photoshop", "T4"):
        assert phrase in intro


def test_printed_recovery_line_matches_the_package(notebook):
    tree = ast.parse(code_cells(notebook)[1])
    printed = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "print"
        and node.args
        and "download_results" in ast.unparse(node.args[0])
    ]
    assert len(printed) == 1, "exactly one recovery print expected"
    expression = compile(ast.Expression(printed[0]), "driver", "eval")
    line = eval(expression, {"CLONE": "/content/clone", "RUN_DIR": "/content/run"})  # noqa: S307
    assert line == delivery.recovery_line("/content/run", "/content/clone")


def _helpers(notebook, names, namespace):
    tree = ast.parse(code_cells(notebook)[1])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "driver", "exec"), namespace)  # noqa: S102
    return namespace


def test_inline_step_marks_use_the_delivery_state_format(notebook, tmp_path):
    import datetime

    ns = _helpers(
        notebook, {"_now", "_mark"}, {"datetime": datetime, "json": json, "traceback": traceback, "RUN_DIR": tmp_path}
    )
    ns["_mark"](2, "Setup", started=True)
    try:
        raise RuntimeError("clone failed")
    except RuntimeError as error:
        ns["_mark"](2, "Setup", error=error)
    verdict = delivery.finalize_verdict(tmp_path)
    assert verdict["status"] == "INVALID"
    assert verdict["text"] == "INVALID: step 2 (Setup) raised RuntimeError: clone failed"


def test_fallback_zips_the_run_without_large_inputs(notebook, tmp_path, capsys):
    import hashlib
    import zipfile

    run_dir = tmp_path / "run1"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "state.json").write_text("{}")
    (run_dir / "inputs" / "ref.arw").write_bytes(b"\0" * 10)
    ns = _helpers(
        notebook, {"_fallback"}, {"hashlib": hashlib, "zipfile": zipfile, "RUN_DIR": run_dir, "RUN_ID": "run1"}
    )
    ns["_fallback"]("setup failed")
    with zipfile.ZipFile(tmp_path / "run1_fallback.zip") as archive:
        assert archive.namelist() == ["run1/state.json"]
    assert "PRD v6 RUN run1: INVALID" in capsys.readouterr().out


def test_repo_ref_pins_the_current_code(notebook):
    """REPO_REF must contain every code change on this branch, or the notebook would run stale code."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    ref = settings_of(notebook)["REPO_REF"]
    git = ["git", "-C", str(ROOT)]
    if subprocess.run(git + ["cat-file", "-e", f"{ref}^{{commit}}"], capture_output=True).returncode != 0:
        pytest.fail(f"REPO_REF {ref} is not a commit in this repository")
    ancestor = subprocess.run(git + ["merge-base", "--is-ancestor", ref, "HEAD"], capture_output=True)
    assert ancestor.returncode == 0, f"REPO_REF {ref} is not an ancestor of HEAD"
    diff = subprocess.run(git + ["diff", "--stat", ref, "--", *CODE_PATHS], capture_output=True, text=True)
    assert diff.stdout == "", f"code changed after REPO_REF {ref}; update it:\n{diff.stdout}"
