"""The Phase 4 universal notebook: its PRD 22.1 section sequence, and end-to-end runs (outside Colab).

Executing the notebook is opt-in exactly like the other notebook tests: it runs only with
SYNTHETIC_RAW_LAB_HEAVY_TESTS=1. Colab is the place to run it. The static checks always run.
"""

import json
import os
from pathlib import Path

import pytest

from synthetic_raw_lab import sr_models

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

heavy = pytest.mark.skipif(
    os.environ.get("SYNTHETIC_RAW_LAB_HEAVY_TESTS") != "1",
    reason="notebook execution is opt-in: set SYNTHETIC_RAW_LAB_HEAVY_TESTS=1",
)
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03_Universal_50MP.ipynb"
WEIGHTS_DIR = Path(os.environ.get("SYNTHETIC_RAW_LAB_WEIGHTS") or sr_models.default_weights_dir(ROOT))
SECTIONS = (
    "Introduction and scientific limitations",
    "Runtime setup",
    "Dependency installation",
    "GPU and environment inspection",
    "Input upload",
    "Input validation",
    "Experiment configuration",
    "Scene analysis",
    "Baseline reconstruction",
    "Pretrained SR",
    "Specialized reconstruction",
    "Tonal and optical processing",
    "Optional sensor simulation",
    "Professional exports",
    "Quality metrics",
    "Visual comparison",
    "Report generation",
    "Download and archive",
)


def run_notebook(parameters_source: str) -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameters = next(c for c in notebook.cells if "parameters" in c.metadata.get("tags", []))
    parameters.source += parameters_source
    nbclient.NotebookClient(notebook, timeout=900, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    return "\n".join(
        str(output.get("text", "")) + str(output.get("data", {}).get("text/markdown", ""))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_notebook_follows_the_prd_22_1_sequence():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    headings = [
        line.lstrip("# ").strip()
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    assert headings == [f"{n}. {title}" for n, title in enumerate(SECTIONS, start=1)]
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "\u2014" not in text
    assert "GITHUB_TOKEN" not in text  # the repository is public
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")
    assert sum("parameters" in cell.metadata.get("tags", []) for cell in notebook.cells) == 1


def test_notebook_code_compiles():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook cell>", "exec")


def parameters(tmp_path: Path, mode: str, sr: bool = False, size: str = "384x256", megapixels: float = 0.393216) -> str:
    return f"""
INPUT_MODE = "{mode}"
SYNTHETIC_SIZE = "{size}"
TARGET_MEGAPIXELS = {megapixels}
CROP_SIZE = 48
ENABLE_SUPER_RESOLUTION = {sr}
WEIGHTS_DIR = "{WEIGHTS_DIR}"
FACE_DETECTION = False
DOWNLOAD_FINAL = False
ADVANCED = {{"review_max_edge": 384}}
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
"""


@heavy
@pytest.mark.parametrize(
    ("mode", "category", "pipeline"),
    [
        ("synthetic_crowd", "human", "notebook 01"),
        ("synthetic_night", "night", "notebook 02"),
        ("synthetic_general", "general", "universal common pipeline"),
    ],
)
def test_notebook_routes_each_scene_and_ends_with_a_final_image(tmp_path, mode, category, pipeline):
    outputs = run_notebook(parameters(tmp_path, mode))
    assert f"Category: **{category}**" in outputs and ": complete in" in outputs
    assert "Measurable improvement over Lanczos" in outputs and "report archive:" in outputs
    manifest = json.loads(next((tmp_path / "reports").glob("*_universal/manifest.json")).read_text())
    assert manifest["status"] == "complete" and manifest["notebook_version"] == "03_Universal_50MP/1.0"
    assert manifest["route"]["pipeline"].startswith(pipeline) or pipeline in manifest["route"]["pipeline"]
    files = manifest["final_output"]["files"]
    assert set(files) == {"tiff16", "exr"} and all(Path(f["path"]).exists() for f in files.values())
    assert files["tiff16"]["width"] * files["tiff16"]["height"] == 393216


@heavy
def test_notebook_runs_the_sr_ladder_with_cached_weights(tmp_path):
    """The common pipeline with the real conservative network; never downloads weights."""
    if not sr_models.available():
        pytest.skip("PyTorch and spandrel are not installed (pip install .[sr])")
    if not sr_models.cached(sr_models.BACKENDS["swinir"]["fidelity"], WEIGHTS_DIR):
        pytest.skip(f"SwinIR weights are not cached in {WEIGHTS_DIR}")
    outputs = run_notebook(parameters(tmp_path, "synthetic_general", sr=True, size="320x180", megapixels=0.2304))
    assert ": complete in" in outputs
    manifest = json.loads(next((tmp_path / "reports").glob("*_universal/manifest.json")).read_text())
    candidates = manifest["route"]["selection"]["candidates"]
    assert "sr_conservative" in candidates
    seams = manifest["super_resolution"]["modes"]["conservative"]["seams"]
    if manifest["route"]["selection"]["input"] == "sr_conservative":
        assert seams["passed"]
