"""Execute notebook 00 end to end on a small synthetic input (outside Colab).

Opt-in: these run only with SYNTHETIC_RAW_LAB_HEAVY_TESTS=1, because executing the notebook (and, with cached
weights, the real networks) is too heavy for a small laptop. Colab is the place to run them.
"""

import os
from pathlib import Path

import pytest

from synthetic_raw_lab import sr_models

pytestmark = pytest.mark.skipif(
    os.environ.get("SYNTHETIC_RAW_LAB_HEAVY_TESTS") != "1",
    reason="notebook execution is opt-in: set SYNTHETIC_RAW_LAB_HEAVY_TESTS=1",
)
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "00_Feasibility_Baselines.ipynb"
WEIGHTS_DIR = Path(os.environ.get("SYNTHETIC_RAW_LAB_WEIGHTS") or sr_models.default_weights_dir(ROOT))


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
ENABLE_SUPER_RESOLUTION = False
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
""")

    assert "run " in outputs and ": complete in" in outputs
    assert list((tmp_path / "outputs" / "universal").glob("*/*_srgb16.tif"))
    assert "Final 50 MP image: lanczos" in outputs and "not in Colab: the final image is" in outputs
    assert list((tmp_path / "outputs" / "universal").glob("*/final/final_*_lanczos_*_srgb16.tif"))
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
ENABLE_SUPER_RESOLUTION = False
""")

    assert "Display P3 (P3 primaries, D65, sRGB transfer) (user_override)" in outputs
    assert list((tmp_path / "sources" / "universal").glob("*_synthetic_320x180.png"))
    assert list((tmp_path / "masters" / "universal").glob("*/*_srgb16.tif"))
    assert list((tmp_path / "run_reports").glob("*/report.md"))


def test_notebook_runs_super_resolution_with_cached_weights(tmp_path):
    """Section 10 with the real networks; skips unless the SwinIR weights are cached (it never downloads)."""
    if not sr_models.available():
        pytest.skip("PyTorch and spandrel are not installed (pip install .[sr])")
    needed = sr_models.BACKENDS["swinir"].values()
    if not all(sr_models.cached(name, WEIGHTS_DIR) for name in needed):
        pytest.skip(f"SwinIR weights are not cached in {WEIGHTS_DIR}")
    outputs = run_notebook(f"""
INPUT_MODE = "synthetic"
SYNTHETIC_SIZE = "320x180"
RUNTIME_PRESET = "quick"
RUN_MEMORY_BENCHMARK = False
BASELINES = "lanczos"
ENABLE_SUPER_RESOLUTION = True
WEIGHTS_DIR = "{WEIGHTS_DIR}"
OUTPUT_ROOT = "{tmp_path / "outputs"}"
REPORTS_ROOT = "{tmp_path / "reports"}"
INPUTS_ROOT = "{tmp_path / "inputs"}"
""")

    assert ": complete in" in outputs
    assert "Mode A, conservative" in outputs and "Mode B, balanced" in outputs
    assert "Final 50 MP image: sr_balanced" in outputs  # the synthetic chart has no face
    masters = list((tmp_path / "outputs" / "universal").glob("*/*_sr_*_srgb16.tif"))
    assert {p.name.split("_sr_")[1].split("_")[0] for p in masters} == {"conservative", "balanced"}
    report = next((tmp_path / "reports").glob("*/report.md")).read_text()
    assert "Pretrained super-resolution (PRD 10, experiment E02)" in report
