"""Static checks of the notebooks that run everywhere (executing them is opt-in, see test_notebook.py)."""

import ast
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "00_Feasibility_Baselines.ipynb"


def code_cells():
    return [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"]


def test_every_code_cell_is_valid_python():
    for source in code_cells():
        ast.parse(source)


def test_the_run_ends_with_the_final_image_delivered_before_the_report_archive():
    cells = code_cells()
    parameters = next(c for c in cells if "@title Run parameters" in c)
    for name in ("FINAL_OUTPUT", "DOWNLOAD_FINAL", "FINAL_DRIVE_DIR"):
        assert f"\n{name} = " in parameters
    delivery = next(i for i, c in enumerate(cells) if "final_output.deliver(" in c)
    archive = next(i for i, c in enumerate(cells) if "make_archive" in c)
    assert delivery == archive - 1 == len(cells) - 2
    assert '"final_output": FINAL_OUTPUT' in next(c for c in cells if "overrides = {" in c)
