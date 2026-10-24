"""The Phase 3 night notebook: its PRD 22.1 section sequence, and an end-to-end run with SR (outside Colab).

Executing the notebook is opt-in exactly like ``test_notebook.py``: it runs only with
SYNTHETIC_RAW_LAB_HEAVY_TESTS=1, because the night run with the real SR networks is too heavy for a small
laptop. Colab is the place to run it. The static section-sequence check always runs.
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
NOTEBOOK = ROOT / "notebooks" / "02_Night_Scenery_50MP.ipynb"
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
    nbclient.NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    return "\n".join(
        str(output.get("text", "")) + str(output.get("data", {}).get("text/markdown", ""))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_notebook_follows_the_prd_22_1_sequence():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
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
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")


@heavy
def test_notebook_runs_on_synthetic_night_scene_with_sr(tmp_path):
    """Sections 9 to 18 with the real conservative and balanced networks; never downloads weights."""
    if not sr_models.available():
        pytest.skip("PyTorch and spandrel are not installed (pip install .[sr])")
    needed = sr_models.BACKENDS["swinir"].values()
    if not all(sr_models.cached(name, WEIGHTS_DIR) for name in needed):
        pytest.skip(f"SwinIR weights are not cached in {WEIGHTS_DIR}")
    outputs = run_notebook(f"""
INPUT_MODE = "synthetic"
SYNTHETIC_SIZE = "320x180"
RUNTIME_PRESET = "quick"
CROP_SIZE = 48
SHADOW_LIFT = "low"
HALO_SUPPRESSION = "true"
SHARPENING_STRENGTH = "0.3"
ADVANCED = {{
    "quick_target_megapixels": 0.25,
    "review_max_edge": 320,
    "failure_rule": {{"block_size": 32}},
    "super_resolution": {{"enabled": True, "weights_dir": "{WEIGHTS_DIR}"}},
}}
REGION_OVERRIDES = [{{"region": "sky_atmosphere", "boxes": [[0, 0, 32, 8]], "note": "notebook test"}}]
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
""")

    assert "run " in outputs and ": complete in" in outputs
    assert "Mode A, conservative SR, swinir_m_x4_classical (inferred, synthesised detail)" in outputs
    assert "Mode B, balanced SR, swinir_m_x4_realsr_gan (inferred, synthesised detail)" in outputs
    for metric in ("Highlight clipping, any channel", "Gradient banding score", "Shadow chroma noise (8-bit codes)"):
        assert metric in outputs
    assert "night input (before)" in outputs and "final night (after)" in outputs
    assert "override: sky_atmosphere" in outputs

    masters = list((tmp_path / "outputs" / "night").glob("*"))
    assert len(masters) == 1
    assert list(masters[0].glob("*_night_*_srgb16.tif"))
    assert list(masters[0].glob("*_night_*_lin_rec709_half.exr"))
    assert list((tmp_path / "inputs" / "night").glob("*_synthetic_night_320x180.png"))
    assert list((tmp_path / "reports").glob("*.zip"))

    manifest = json.loads(next((tmp_path / "reports").glob("*/manifest.json")).read_text())
    assert manifest["status"] == "complete"
    assert manifest["notebook_version"] == "02_Night_Scenery_50MP/1.2"
    assert manifest["config"]["shadows"]["lift_strength"] == "low"
    assert manifest["config"]["texture"]["sharpening_strength"] == 0.3
    assert manifest["regions"]["overrides"][0]["note"] == "notebook test"
    assert {p["crop"] for p in manifest["crop_panels"]} == {"highlight", "shadow", "reflection", "gradient"}
    assert {"regions", "fallbacks", "banding", "halo"} <= set(manifest["figures"])
    report = next((tmp_path / "reports").glob("*/report.md")).read_text()
    assert "figures/regions.png" in report and "night input (before)" in report
    assert "Pretrained super-resolution (PRD 10, experiment E02)" in report
    modes = manifest["super_resolution"]["modes"]
    assert [modes[m]["weights"] for m in modes] == ["swinir_m_x4_classical", "swinir_m_x4_realsr_gan"]
    assert manifest["night"]["input"]["slot"] == "balanced_sr"
    assert [m["weights"]["name"] for m in manifest["models"]] == list(sr_models.BACKENDS["swinir"].values())
    stages = [s["name"] for s in manifest["stages"]]
    assert stages.count("super_resolution_balanced") == 1 and stages.count("super_resolution_conservative") == 1
