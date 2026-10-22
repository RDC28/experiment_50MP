"""Execute the Phase 3 night notebook end to end on a small synthetic night scene (outside Colab)."""

import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "02_Night_Scenery_50MP.ipynb"
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


def test_notebook_runs_on_synthetic_night_scene(tmp_path):
    outputs = run_notebook(f"""
INPUT_MODE = "synthetic"
SYNTHETIC_SIZE = "320x180"
RUNTIME_PRESET = "quick"
CROP_SIZE = 48
SHADOW_LIFT = "low"
HALO_SUPPRESSION = "true"
SHARPENING_STRENGTH = "0.3"
ADVANCED = {{"quick_target_megapixels": 0.25, "review_max_edge": 320, "failure_rule": {{"block_size": 32}}}}
REGION_OVERRIDES = [{{"region": "sky_atmosphere", "boxes": [[0, 0, 32, 8]], "note": "notebook test"}}]
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
""")

    assert "run " in outputs and ": complete in" in outputs
    assert "Lanczos stand-in for Conservative SR (Phase 2 SR backend not merged)" in outputs
    assert "Lanczos stand-in for Balanced SR (Phase 2 SR backend not merged)" in outputs
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
    assert manifest["notebook_version"] == "02_Night_Scenery_50MP/1.0"
    assert manifest["config"]["shadows"]["lift_strength"] == "low"
    assert manifest["config"]["texture"]["sharpening_strength"] == 0.3
    assert manifest["regions"]["overrides"][0]["note"] == "notebook test"
    assert {p["crop"] for p in manifest["crop_panels"]} == {"highlight", "shadow", "reflection", "gradient"}
    assert {"regions", "fallbacks", "banding", "halo"} <= set(manifest["figures"])
    report = next((tmp_path / "reports").glob("*/report.md")).read_text()
    assert "figures/regions.png" in report and "night input (before)" in report
