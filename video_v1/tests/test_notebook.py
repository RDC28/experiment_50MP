"""Notebook contract: two code cells, literal settings, the upload step, one try/finally, pinned code."""

from __future__ import annotations

import ast
import builtins
import json
import re
import shutil
import stat
import subprocess
import sys
import traceback
import types
import zipfile
from pathlib import Path

import pytest

from prd_v6.install import DNG_VALIDATE_BUILD
from video_v1 import delivery, preflight
from video_v1.tests.conftest import make_clip, needs_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "video_v1" / "notebooks" / "video_to_raw_v1.ipynb"
SETTINGS = {"VIDEO": None, "DRIVE_MIRROR": False}
# Changes under these paths change what the notebook runs, so REPO_REF must include them.
CODE_PATHS = [
    "video_v1",
    "prd_v6",
    "prd_v5",
    "src",
    "pyproject.toml",
    ":!video_v1/notebooks",
    ":!video_v1/tests",
    ":!*.md",
]


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


def test_settings_are_literal_defaults(notebook):
    settings = settings_of(notebook)
    assert settings.keys() == {"REPO_REF", *SETTINGS}
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


def run_driver_cell(notebook, root: Path, monkeypatch, stand_ins: dict, *, uploads=None, settings=None):
    """Run the settings cell, then the driver cell as Colab would.

    `uploads` is what the upload button returns ({file name: bytes}); None makes an upload a test failure.
    `stand_ins` replace the driver cell's functions of those names; `settings` override the settings cell.
    """
    calls = {"download": [], "upload": [], "input": [], "mount": []}

    def forbidden(name):
        def call(*args, **kwargs):
            calls[name].append(args)
            raise AssertionError(f"the driver cell called {name}")

        return call

    def upload():
        calls["upload"].append(())
        if uploads is None:
            raise AssertionError("the driver cell called upload")
        return dict(uploads)

    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=lambda path: calls["download"].append(Path(path)), upload=upload)
    colab.drive = types.SimpleNamespace(mount=lambda path: calls["mount"].append(path))
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(builtins, "input", forbidden("input"))
    tree = _ContentRoot(root).visit(ast.parse(code_cells(notebook)[1]))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in stand_ins)]
    namespace: dict = {"__name__": "__main__"}
    exec(code_cells(notebook)[0], namespace)  # noqa: S102
    namespace.update(settings or {})
    namespace.update(stand_ins)
    exec(compile(ast.fix_missing_locations(tree), "driver", "exec"), namespace)  # noqa: S102
    return namespace, calls


def _setup_fails():
    raise RuntimeError("clone failed")


def _convert_fails(run_dir, settings):
    delivery.mark_step_started(run_dir, 4)
    try:
        raise RuntimeError("convert exploded")
    except RuntimeError as error:
        delivery.mark_step_finished(run_dir, 4, error)
        raise


NO_UPLOAD = "step 1 (Input) raised RuntimeError: upload exactly one video file (got 0); or set VIDEO and run all again"


@pytest.mark.parametrize(
    ("setup", "run", "uploads", "reason"),
    [
        (lambda: None, None, {}, NO_UPLOAD),
        (_setup_fails, None, {"clip.mp4": b"x"}, "step 2 (Setup) raised RuntimeError: clone failed"),
        (lambda: None, _convert_fails, {"clip.mp4": b"x"}, "step 4 (Convert) raised RuntimeError: convert exploded"),
    ],
    ids=["no-upload", "setup", "convert"],
)
def test_driver_cell_finalizes_and_downloads_an_invalid_run(
    notebook, tmp_path, monkeypatch, capsys, setup, run, uploads, reason
):
    from video_v1 import driver

    monkeypatch.setattr(driver, "run", run or (lambda *_: pytest.fail("driver.run ran after a failed step")))
    ns, calls = run_driver_cell(notebook, tmp_path, monkeypatch, {"_setup": setup}, uploads=uploads)
    run_dir = ns["RUN_DIR"]
    assert run_dir.parent == tmp_path / "exp50mp_video_v1"
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["status"] == "INVALID" and verdict["final"]
    assert verdict["text"] == f"INVALID: {reason} {delivery.NO_ACCURACY_CLAIM}"
    summary = delivery.summary_path(run_dir)
    assert calls["download"] == [summary] and ns["ARCHIVES"] == [summary]
    with zipfile.ZipFile(summary) as archive:
        assert json.loads(archive.read("verdict.json"))["text"] == verdict["text"]
        assert not any(name.endswith("clip.mp4") for name in archive.namelist())
    assert len(calls["upload"]) == 1 and calls["input"] == calls["mount"] == []
    assert f"VIDEO v1 RUN {ns['RUN_ID']}: INVALID" in capsys.readouterr().out


@pytest.fixture
def setup_stand_in(tmp_path, monkeypatch):
    """Step 2 as Colab leaves it, without a clone or a build: a dng_validate record and a stand-in validator."""
    from prd_v5.dng_validate import ENV_VAR

    monkeypatch.setenv(ENV_VAR, "")
    monkeypatch.setattr(preflight, "gpu_name", lambda: "NVIDIA L4")
    tool = tmp_path / "bin" / "dng_validate"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\necho 'Validating \"$2\"...'\nexit 0\n")  # what dng_validate -v prints on a pass
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)

    def _setup():
        run_dir = next((tmp_path / "exp50mp_video_v1").iterdir())
        (run_dir / DNG_VALIDATE_BUILD).parent.mkdir(exist_ok=True)
        (run_dir / DNG_VALIDATE_BUILD).write_text(json.dumps({"status": "built", "binary": str(tool)}))

    return _setup


@needs_ffmpeg
def test_an_uploaded_clip_runs_to_a_complete_dng_sequence(notebook, tmp_path, monkeypatch, capsys, setup_stand_in):
    pytest.importorskip("rawpy")
    (tmp_path / "clips").mkdir()
    clip = make_clip(tmp_path / "clips" / "beach day.mkv", 3)
    ns, calls = run_driver_cell(
        notebook, tmp_path, monkeypatch, {"_setup": setup_stand_in}, uploads={clip.name: clip.read_bytes()}
    )
    run_dir = ns["RUN_DIR"]
    assert ns["VERDICT"]["status"] == "COMPLETE", ns["VERDICT"]["text"]
    stored = run_dir / "inputs" / "beach_day.mkv"
    assert stored.read_bytes() == clip.read_bytes()
    origin = json.loads((run_dir / "source.json").read_text())["origin"]
    assert origin == {"how": "uploaded", "path": str(stored), "name": "beach day.mkv", "bytes": clip.stat().st_size}
    rid = ns["RUN_ID"]
    assert [p.name for p in calls["download"]] == [f"{rid}_01_summary.zip", f"{rid}_02_dng_part1of1.zip"]
    with zipfile.ZipFile(calls["download"][1]) as archive:
        dngs = [n for n in archive.namelist() if n.endswith(".dng")]
    assert dngs == [f"dng/beach_day/beach_day_{i:06d}.dng" for i in range(3)]
    assert calls["mount"] == [] and f"VIDEO v1 RUN {rid}: COMPLETE" in capsys.readouterr().out


def test_a_drive_path_mounts_drive_and_skips_the_upload(notebook, tmp_path, monkeypatch):
    settings = {"VIDEO": str(tmp_path / "drive" / "MyDrive" / "clip.mp4")}  # as the cell sees /content/drive/
    ns, calls = run_driver_cell(notebook, tmp_path, monkeypatch, {"_setup": _setup_fails}, settings=settings)
    assert calls["mount"] == [str(tmp_path / "drive")] and calls["upload"] == []
    assert ns["VERDICT"]["reasons"] == ["step 2 (Setup) raised RuntimeError: clone failed"]


def test_driver_cell_falls_back_when_packaging_fails(notebook, tmp_path, monkeypatch, capsys):
    def packaging_fails(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(delivery, "finalize_run", packaging_fails)
    ns, calls = run_driver_cell(
        notebook, tmp_path, monkeypatch, {"_setup": _setup_fails}, uploads={"clip.mp4": b"\0" * 10}
    )
    fallback = tmp_path / "exp50mp_video_v1" / f"{ns['RUN_ID']}_fallback.zip"
    assert calls["download"] == [fallback]
    rid = ns["RUN_ID"]
    with zipfile.ZipFile(fallback) as archive:
        state = json.loads(archive.read(f"{rid}/state.json"))
        names = archive.namelist()
    assert f"{rid}/config.json" in names and f"{rid}/inputs/upload.json" in names
    assert f"{rid}/inputs/clip.mp4" not in names
    assert [(r["step"], r["status"], r.get("error")) for r in state["steps"]] == [
        (1, "finished", None),
        (2, "failed", "RuntimeError: clone failed"),
    ]
    out = capsys.readouterr().out
    assert f"VIDEO v1 RUN {rid}: INVALID" in out
    assert "INVALID: RuntimeError: clone failed; packaging: OSError: disk full" in out


def test_colab_runtime_and_tier_are_stated(notebook):
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    assert notebook["metadata"]["colab"]["machine_shape"] == "hm"


def test_printed_recovery_line_matches_the_package_and_downloads_again(notebook, tmp_path, monkeypatch, capsys):
    ns, calls = run_driver_cell(notebook, tmp_path, monkeypatch, {"_setup": _setup_fails}, uploads={"clip.mp4": b"x"})
    recovery = delivery.recovery_line(ns["RUN_DIR"], ns["CLONE"])
    assert recovery in capsys.readouterr().out.splitlines()
    summary = delivery.summary_path(ns["RUN_DIR"])
    assert calls["download"] == [summary]
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(recovery, {"__name__": "__main__"})  # noqa: S102 - the line pasted into a new cell
    assert calls["download"] == [summary, summary]


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
    ns["_mark"](1, "Input", started=True)
    ns["_mark"](1, "Input")
    ns["_mark"](2, "Setup", started=True)
    try:
        raise RuntimeError("clone failed")
    except RuntimeError as error:
        ns["_mark"](2, "Setup", error=error)
    state = delivery.load_state(tmp_path)
    assert delivery.finished_steps(state) == {1} and state["step_started"] == 2
    verdict = delivery.finalize_verdict(tmp_path)
    assert verdict["status"] == "INVALID"
    assert verdict["text"].startswith("INVALID: step 2 (Setup) raised RuntimeError: clone failed")


def test_fallback_zips_only_the_small_records(notebook, tmp_path, capsys):
    import hashlib

    run_dir = tmp_path / "run1"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "dng" / "c").mkdir(parents=True)
    (run_dir / "state.json").write_text("{}")
    (run_dir / "inputs" / "clip.MOV").write_bytes(b"\0" * 10)
    (run_dir / "dng" / "c" / "c_000000.dng").write_bytes(b"\0" * 10)
    ns = _helpers(
        notebook, {"_fallback"}, {"hashlib": hashlib, "zipfile": zipfile, "RUN_DIR": run_dir, "RUN_ID": "run1"}
    )
    ns["_fallback"]("setup failed")
    with zipfile.ZipFile(tmp_path / "run1_fallback.zip") as archive:
        assert archive.namelist() == ["run1/state.json"]
    assert "VIDEO v1 RUN run1: INVALID" in capsys.readouterr().out


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
