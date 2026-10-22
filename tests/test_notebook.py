"""Execute the Phase 0/1 notebook end to end on a small synthetic input (outside Colab)."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "00_Feasibility_Baselines.ipynb"


def test_notebook_runs_with_synthetic_input(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameters = next(c for c in notebook.cells if "parameters" in c.metadata.get("tags", []))
    parameters.source += f"""
INPUT_MODE = "synthetic"
SYNTHETIC_SIZE = "320x180"
RUNTIME_PRESET = "quick"
RUN_MEMORY_BENCHMARK = False
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
"""
    nbclient.NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()

    outputs = "\n".join(
        output.get("text", "") for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    )
    assert "run " in outputs and ": complete in" in outputs
    assert list((tmp_path / "outputs" / "universal").glob("*/*_srgb16.tif"))
    assert list((tmp_path / "reports").glob("*/report.md"))
    assert list((tmp_path / "reports").glob("*.zip"))
    assert list((tmp_path / "inputs" / "universal").glob("*_synthetic_320x180.png"))
