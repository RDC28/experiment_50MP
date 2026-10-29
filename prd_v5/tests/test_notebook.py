"""Static notebook validation only; never execute the Colab notebook locally."""

import ast
import json
from pathlib import Path

import nbformat

from prd_v5.analytic import AnalyticISP
from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "synthetic_50mp_v5.ipynb"


def assignments():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    values = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and not cell.outputs
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    values[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return notebook, values


def test_notebook_maximum_requests_and_privacy():
    notebook, values = assignments()
    assert values["TARGET_MEGAPIXELS"] == 50
    assert values["SR_BACKEND"] == "HAT"
    assert values["INVERSE_ISP_MODE"] == "HYBRID"
    assert values["ADC_BITS"] == 14
    assert not values["FABRICATE_UNOBSERVABLE_METADATA"]
    assert not values["ALLOW_ANALYTIC_BASELINE"]
    assert all(v for k, v in values.items() if k.startswith(("ENABLE_", "WRITE_")))
    assert (values["SR_TILE"], values["SR_OVERLAP"], values["RAW_TILE"], values["RAW_OVERLAP"]) == (512, 64, 1024, 128)
    fields = {f["name"]: f for f in values["FIELD_REQUESTS"]}
    for name in ("GPS", "Camera serial", "Lens serial", "Timestamp"):
        assert fields[name]["value"] is None
        assert fields[name]["mode"] == "USER"
    assert all(f["mode"] in {"AUTO", "ESTIMATE", "FABRICATE", "USER"} for f in fields.values())
    source = "\n".join(c.source for c in notebook.cells)
    assert "run_analytic_pair" in source and "Colab-only" in source
    assert "skipped because integration unavailable" in source
    assert "\u2014" not in source


def test_notebook_hypotheses_are_valid_without_execution():
    _, values = assignments()
    # Extracting literal config is not executing notebook cells or full-scale work.
    camera_values = dict(values["CAMERA_VALUES"])
    # Shutter expression is intentionally not evaluated by literal_eval.
    camera_values.setdefault("shutter_s", 1 / 125)
    camera = CameraHypothesis(**camera_values)
    isp = AnalyticISP(**values["ISP_VALUES"])
    config = AnalyticRunConfig(camera=camera, isp=isp, max_develop_bytes=values["MAX_DEVELOP_BYTES"])
    assert json.loads(config.to_json())["camera"]["adc_bits"] == 14


def test_notebook_covers_prd_parameter_tables():
    _, values = assignments()
    prd = (NOTEBOOK.parents[1] / "PRD_v5.md").read_text()
    section = prd.split("### Sensor and ADC parameters")[1].split("## Colab implementation architecture")[0]
    expected = {
        line.split("|")[1].strip()
        for line in section.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip() not in {"Parameter", "Field"}
    }
    assert expected == {field["name"] for field in values["FIELD_REQUESTS"]}


def test_notebook_routes_supported_panel_values_and_evidence():
    notebook, _ = assignments()
    execution = notebook.cells[-1].source
    assert "resolve_analytic_panel(" in execution
    assert "isp_selections=isp_selections" in execution
    assert "camera_selections=camera_selections" in execution


def test_notebook_dispatches_verified_sr_and_records_only_completed_execution():
    notebook, values = assignments()
    assert values["SR_CHECKPOINT"] is None
    preflight = notebook.cells[9].source
    execution = notebook.cells[-1].source
    assert "resolve_sr_panel(" in preflight
    assert 'states["SR_BACKEND"] = sr_state' in preflight
    tree = ast.parse(execution)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    paired = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "run_analytic_pair")
    sr = next(keyword.value for keyword in paired.keywords if keyword.arg == "sr_checkpoint")
    assert isinstance(sr, ast.Name) and sr.id == "sr_checkpoint"
    assert execution.index("result = run_analytic_pair(") < execution.index('"status": "executed;')
    assert '"[sr]"' in execution
    assert "(SR_TILE, SR_OVERLAP) != (512, 64)" in execution


def test_notebook_exposes_and_dispatches_explicit_joint_exposure_prior():
    notebook, values = assignments()
    assert values["EXPOSURE_PRIOR"] is None
    execution = ast.parse(notebook.cells[-1].source)
    call = next(
        node
        for node in ast.walk(execution)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "resolve_analytic_panel"
    )
    prior = next(keyword.value for keyword in call.keywords if keyword.arg == "exposure_prior")
    assert isinstance(prior, ast.Name) and prior.id == "EXPOSURE_PRIOR"


def test_notebook_shared_colour_proxy_dispatch():
    notebook, values = assignments()
    assert values["COLOUR_SOURCE_PATH"] is None
    tree = ast.parse(notebook.cells[-1].source)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "resolve_analytic_panel"
    )
    kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
    for key, name in {
        "colour_source_path": "COLOUR_SOURCE_PATH",
        "assume_unprofiled_srgb": "ASSUME_UNPROFILED_SRGB",
        "max_decode_bytes": "MAX_DECODE_BYTES",
        "strip_rows": "STRIP_ROWS",
    }.items():
        assert isinstance(kwargs[key], ast.Name) and kwargs[key].id == name
    assert "resolve_analytic_panel_from_source as resolve_analytic_panel" in notebook.cells[-1].source


def test_notebook_dispatches_optional_native_crop_without_changing_default():
    notebook, values = assignments()
    assert values["COMPARISON_REGION"] is None
    tree = ast.parse(notebook.cells[-1].source)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_analytic_pair"
    )
    region = next(keyword.value for keyword in call.keywords if keyword.arg == "comparison_region")
    assert isinstance(region, ast.Name) and region.id == "COMPARISON_REGION"
    # Evaluate only the argument expression, never notebook setup or processing.
    expression = compile(ast.Expression(region), "<notebook crop argument>", "eval")
    for selected in (None, (0, 1, 4, 5)):
        assert eval(expression, {"COMPARISON_REGION": selected}) == selected
