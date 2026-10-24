"""Execute the Phase 5 notebook end to end on a small synthetic input (outside Colab).

Opt-in, like the notebook 00 test: it runs only with SYNTHETIC_RAW_LAB_HEAVY_TESTS=1, because executing a
notebook is too heavy for a small laptop. Colab is the place to run it.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SYNTHETIC_RAW_LAB_HEAVY_TESTS") != "1",
    reason="notebook execution is opt-in: set SYNTHETIC_RAW_LAB_HEAVY_TESTS=1",
)
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "05_Synthetic_Sensor_Research.ipynb"


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


def test_notebook_builds_a_master_and_runs_phase_5(tmp_path):
    outputs = run_notebook(f"""
INPUT_MODE = "synthetic"
SYNTHETIC_SIZE = "320x180"
MASTER_PRESET = "quick"
QUICK_MEGAPIXELS = 0.3
SHOT_NOISE = True
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
""")

    assert ": complete in" in outputs
    assert "Phase 5 exit criterion met" in outputs
    assert "more bits per sample add container precision, not information" in outputs
    assert list((tmp_path / "outputs" / "universal").glob("*_sensor_research/*SYNTHETIC*.dng"))
    assert list((tmp_path / "outputs" / "universal").glob("*_baselines/*_lanczos_*.exr"))
    assert list((tmp_path / "reports").glob("*_sensor_research/report.md"))
    assert list((tmp_path / "reports").glob("*_sensor_research.zip"))
