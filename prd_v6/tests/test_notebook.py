"""Notebook contract (section 9.2): two code cells, literal settings, one try/finally, pinned code."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import traceback
from pathlib import Path

import pytest

from prd_v6 import delivery

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "prd_v6" / "notebooks" / "synthetic_50mp_v6.ipynb"
SETTINGS = {
    "PRESET": "max",
    "SOURCE_4K_URL": None,
    "SOURCE_2K_URL": None,
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


def test_driver_runs_everything_inside_one_try_finally(notebook):
    tree = ast.parse(code_cells(notebook)[1])
    last = tree.body[-1]
    assert isinstance(last, ast.Try) and last.finalbody, "the driver must end with the try/finally"
    finally_source = ast.unparse(last.finalbody)
    assert "finalize_run" in finally_source and "_fallback" in finally_source
    body = ast.unparse(last.body)
    for step in ("_setup()", "driver.run(RUN_DIR, SETTINGS)"):
        assert step in body


def test_driver_never_waits_for_the_user(notebook):
    source = code_cells(notebook)[1]
    for forbidden in ("files.upload", "input(", "ipywidgets", "lanczos_model"):
        assert forbidden not in source


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
