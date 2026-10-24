"""The Phase 6 evaluation notebook: its section sequence and an end-to-end run (outside Colab).

Executing the notebook is opt-in like the other notebook tests: it runs only with SYNTHETIC_RAW_LAB_HEAVY_TESTS=1,
without super-resolution and without the network (the reference test is off). Colab is the place to run it.
"""

import json
import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

heavy = pytest.mark.skipif(
    os.environ.get("SYNTHETIC_RAW_LAB_HEAVY_TESTS") != "1",
    reason="notebook execution is opt-in: set SYNTHETIC_RAW_LAB_HEAVY_TESTS=1",
)
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "06_Evaluation.ipynb"
SECTIONS = (
    "Introduction and scientific limitations",
    "Runtime setup",
    "Dependency installation",
    "GPU and environment inspection",
    "Evaluation configuration",
    "Run collection",
    "Run comparison",
    "Visual review sheet",
    "Full-pipeline ablation",
    "Controlled reference test",
    "Performance measurements",
    "Blind review",
    "Final experiment report",
    "Download and archive",
)


def test_notebook_follows_its_section_sequence():
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
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")
    assert sum("parameters" in cell.metadata.get("tags", []) for cell in notebook.cells) == 1


def test_notebook_code_compiles():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook cell>", "exec")


@heavy
def test_notebook_collects_runs_ablates_and_writes_the_final_report(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameters = next(c for c in notebook.cells if "parameters" in c.metadata.get("tags", []))
    parameters.source += f"""
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
ABLATION_INPUT = "synthetic_night"
SYNTHETIC_SIZE = "320x180"
ABLATION_CROP_SIZE = 128
RUN_REFERENCE_TEST = False
ENABLE_SUPER_RESOLUTION = False
ADVANCED = {{"ablation": {{"target_megapixels": 0.2304, "performance_sweep": []}}}}
"""
    nbclient.NotebookClient(notebook, timeout=900, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    report = next((tmp_path / "reports").glob("final_report_*/final_report.json"))
    record = json.loads(report.read_text())
    assert len(record["answers"]) == 18
    assert {r["notebook"] for r in record["runs"]} >= {"00", "06 ablation"}
    manifests = [json.loads(p.read_text()) for p in (tmp_path / "reports").glob("*/manifest.json")]
    ablation = next(m for m in manifests if m.get("schema", "").startswith("synthetic_raw_lab.phase6_ablation"))
    assert ablation["notebook_version"] == "06_Evaluation/1.0"
