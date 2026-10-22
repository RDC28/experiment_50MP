"""Execute the Phase 0/1 notebook end to end on a small synthetic input (outside Colab)."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "00_Feasibility_Baselines.ipynb"


def run_notebook(parameters_source: str):
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


def test_notebook_runs_with_synthetic_input(tmp_path):
    outputs = run_notebook(f"""
INPUT_MODE = "synthetic"
SYNTHETIC_SIZE = "320x180"
RUNTIME_PRESET = "quick"
RUN_MEMORY_BENCHMARK = False
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
""")

    assert "run " in outputs and ": complete in" in outputs
    assert list((tmp_path / "outputs" / "universal").glob("*/*_srgb16.tif"))
    assert list((tmp_path / "reports").glob("*/report.md"))
    assert list((tmp_path / "reports").glob("*.zip"))
    assert list((tmp_path / "inputs" / "universal").glob("*_synthetic_320x180.png"))


def test_notebook_uses_config_file_paths_and_color_space(tmp_path):
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(
        "baselines: [bicubic]\n"
        "source_color_space: display_p3\n"
        f"output_root: {tmp_path / 'masters'}\n"
        f"reports_root: {tmp_path / 'run_reports'}\n"
        f"inputs_root: {tmp_path / 'sources'}\n"
    )
    outputs = run_notebook(f"""
CONFIG_FILE = "{config_file}"
BASELINES = "bicubic"
INPUT_MODE = "synthetic"
SYNTHETIC_SIZE = "320x180"
RUNTIME_PRESET = "quick"
RUN_MEMORY_BENCHMARK = False
""")

    assert "Display P3 (P3 primaries, D65, sRGB transfer) (user_override)" in outputs
    assert list((tmp_path / "sources" / "universal").glob("*_synthetic_320x180.png"))
    assert list((tmp_path / "masters" / "universal").glob("*/*_srgb16.tif"))
    assert list((tmp_path / "run_reports").glob("*/report.md"))
