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
    assert values["ALLOW_ANALYTIC_BASELINE"]
    assert all(v for k, v in values.items() if k.startswith(("ENABLE_", "WRITE_")))
    assert (values["SR_TILE"], values["SR_OVERLAP"], values["RAW_TILE"], values["RAW_OVERLAP"]) == (512, 64, 1024, 128)
    fields = {f["name"]: f for f in values["FIELD_REQUESTS"]}
    for name in ("GPS", "Camera serial", "Lens serial", "Timestamp"):
        assert fields[name]["value"] is None
        assert fields[name]["mode"] == "USER"
    assert all(f["mode"] in {"AUTO", "ESTIMATE", "FABRICATE", "USER"} for f in fields.values())


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


def test_full_preset_values_are_executable_and_match_notebook():
    from dataclasses import asdict

    from prd_v5.optical_effects import OpticalEffects
    from prd_v5.presets import full_quality_config
    from prd_v5.sensor import SensorEffects

    _, values = assignments()
    preset = full_quality_config()
    assert asdict(preset.sensor_effects) == values["SENSOR_VALUES"]
    assert asdict(preset.optical_effects) == values["OPTICAL_VALUES"]
    assert all(x["effective"] for x in SensorEffects(**values["SENSOR_VALUES"]).states().values())
    assert all(OpticalEffects(**values["OPTICAL_VALUES"]).states().values())
    assert values["ACCEPT_SENSOR_EXPERIMENT"] and values["ACCEPT_OPTICS_EXPERIMENT"]
    assert values["ACCEPT_ANALYTIC_HDR_EXPERIMENT"]
    assert AnalyticRunConfig.from_json(preset.to_json()) == preset
