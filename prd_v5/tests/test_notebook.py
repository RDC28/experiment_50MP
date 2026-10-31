"""Notebook validation: literal settings, plus selected cells executed against stubs.

The Colab run itself never executes locally; stubbed cells exercise only the wiring.
"""

import ast
import json
from pathlib import Path

import nbformat
import pytest

from prd_v5.analytic import AnalyticISP
from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.learned import ResidualAdapter

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
    assert preset.exposure_sweep == values["ENABLE_EXPOSURE_SWEEP"] is True
    assert preset.tone_sweep == values["ENABLE_TONE_SWEEP"] is True
    assert preset.shadow_lifts_ev == values["SHADOW_LIFTS_EV"] == (0.5, 2.0)
    assert preset.demosaic_sweep == values["ENABLE_DEMOSAIC_SWEEP"] is True
    assert preset.wb_sweep == values["ENABLE_WB_SWEEP"] is True
    assert preset.wb_sweep_strength == values["WB_SWEEP_STRENGTH"] == 1.25
    assert asdict(preset.sensor_effects) == values["SENSOR_VALUES"]
    assert asdict(preset.optical_effects) == values["OPTICAL_VALUES"]
    assert all(x["effective"] for x in SensorEffects(**values["SENSOR_VALUES"]).states().values())
    assert all(OpticalEffects(**values["OPTICAL_VALUES"]).states().values())
    assert values["ACCEPT_SENSOR_EXPERIMENT"] and values["ACCEPT_OPTICS_EXPERIMENT"]
    assert values["ACCEPT_ANALYTIC_HDR_EXPERIMENT"]
    assert AnalyticRunConfig.from_json(preset.to_json()) == preset


def code_sources():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def execute(sources, tmp_path):
    """Run sources in a fresh kernel, stopping at the first error, and return printed JSON."""
    nbclient = pytest.importorskip("nbclient")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source) for source in sources])
    client = nbclient.NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(tmp_path)}})
    error = None
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as raised:
        error = raised
    printed = [
        json.loads(line.removeprefix("RECORD "))
        for cell in notebook.cells
        for output in cell.outputs
        if output.get("name") == "stdout"
        for line in output["text"].splitlines()
        if line.startswith("RECORD ")
    ]
    return printed, error


PAIR_STUB = """
import json
from pathlib import Path

import prd_v5.paired


class PairStub(Exception):
    pass


def record_pair(*args, **kwargs):
    print("RECORD " + json.dumps({"args": [str(a) for a in args], "kwargs": kwargs}, default=repr))
    raise PairStub


prd_v5.paired.run_analytic_pair = record_pair
"""

PAIR_OVERRIDES = """
COMPARISON_REGION = (0, 0, 64, 64)
COLOUR_REFERENCES = {"4k": {"path": "/content/ref_4k.tiff", "calibration": {"colour_space": "sRGB_D65_2deg"}}}
LPIPS_OPTIONS = {"patch": 128, "net": "vgg", "device": "cpu"}
MTF_EDGE = {"region": (1, 2, 65, 66), "normal_axis": "x", "edge_evidence": "test edge"}
"""


@pytest.mark.parametrize(("learned", "mode"), [(True, "HYBRID"), (False, "ANALYTIC")])
def test_notebook_run_cell_passes_metric_controls_to_the_pair(tmp_path, learned, mode):
    _, params, fields, hypotheses, *_, run = code_sources()
    context = f"""
from prd_v5.learned import resolve_mode

output = Path({str(tmp_path / "out")!r})
runtime = {{"ram_available": 1 << 50, "disk_free": 1 << 50}}
states = {{}}
sr_checkpoint, sr_state = None, {{"planned": RESAMPLING}}
"""
    sources = [PAIR_STUB, params + PAIR_OVERRIDES + f"ENABLE_LEARNED_RGB2RAW = {learned}\n", fields, hypotheses]
    (call,), error = execute([*sources, context, run], tmp_path)
    assert error is not None and error.ename == "PairStub"
    assert call["args"][:3] == ["/content/source_2k.png", "/content/source_4k.png", str(tmp_path / "out" / "pair")]
    kwargs = call["kwargs"]
    assert kwargs["comparison_region"] == [0, 0, 64, 64]
    assert kwargs["colour_references"] == {
        "4k": {"path": "/content/ref_4k.tiff", "calibration": {"colour_space": "sRGB_D65_2deg"}}
    }
    assert kwargs["lpips_options"] == {"patch": 128, "net": "vgg", "device": "cpu"}
    assert kwargs["mtf_edge"] == {"region": [1, 2, 65, 66], "normal_axis": "x", "edge_evidence": "test edge"}
    assert kwargs["residual_adapter"] is None and kwargs["inverse_mode"] == mode


SETUP_STUB = """
import importlib.machinery
import os
import subprocess
import sys
import types

import prd_v5.dng_validate

for name in ("google", "google.colab"):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    module.__path__ = []
    sys.modules[name] = module
COMMANDS, BUILDS = [], []
subprocess.run = lambda command, **kwargs: COMMANDS.append(command) or subprocess.CompletedProcess(command, 0)
subprocess.check_output = lambda command, **kwargs: "0123abcd\\n"
os.environ.pop("SYNTHRAW_DNG_VALIDATE", None)
os.environ.pop("SYNTHRAW_DNG_VALIDATE_SKIP", None)
prd_v5.dng_validate.shutil.which = lambda name: None
BUILD = {"status": "built", "binary": "/content/dng_validate_build/dng_validate"}


def record_build(workdir, **kwargs):
    BUILDS.append(str(workdir))
    return BUILD


prd_v5.dng_validate.build_dng_validate = record_build
"""

BUILD_FAILURE = """
BUILD = {"status": "not measured", "reason": "dng_validate build failed (exit 1): undefined reference"}
"""

SETUP_REPORT = """
import json
gate = prd_v5.dng_validate.run_dng_validate("capture.dng")
tool = os.environ.get("SYNTHRAW_DNG_VALIDATE")
print("RECORD " + json.dumps({"builds": BUILDS, "tool": tool, "gate": gate, "commands": COMMANDS}))
"""


@pytest.mark.parametrize(
    ("prebuilt", "builds", "tool"),
    [
        (None, ["/content/dng_validate_build"], "/content/dng_validate_build/dng_validate"),
        ("/opt/dng_validate", [], "/opt/dng_validate"),
    ],
)
def test_notebook_setup_builds_dng_validate_only_without_a_prebuilt_path(tmp_path, prebuilt, builds, tool):
    setup = code_sources()[0]
    if prebuilt is not None:
        assert "DNG_VALIDATE_PATH = None" in setup
        setup = setup.replace("DNG_VALIDATE_PATH = None", f"DNG_VALIDATE_PATH = {prebuilt!r}", 1)
    (record,), error = execute([SETUP_STUB, setup, SETUP_REPORT], tmp_path)
    assert error is None
    assert record["builds"] == builds and record["tool"] == tool
    assert any("pip" in command for command in record["commands"])


def test_notebook_setup_build_failure_names_the_error_in_the_conformance_gate(tmp_path):
    (record,), error = execute([SETUP_STUB, BUILD_FAILURE, code_sources()[0], SETUP_REPORT], tmp_path)
    assert error is None
    assert record["builds"] == ["/content/dng_validate_build"] and record["tool"] is None
    assert record["gate"] == {
        "status": "not measured",
        "reason": "dng_validate build failed (exit 1): undefined reference",
    }


def test_notebook_hdrunet_defaults_are_inert():
    _, values = assignments()
    assert values["COLOUR_REFERENCES"] == {} and values["MTF_EDGE"] is None
    assert values["LPIPS_OPTIONS"] == {"patch": 256, "net": "alex", "device": "cuda"}
    assert values["HDRUNET_CHECKPOINT"] is None and values["HDRUNET_SHA256"] is None
    assert values["HDRUNET_LICENCE_ACCEPTED"] is False
    assert values["DNG_VALIDATE_PATH"] is None


def run_resolver(**overrides):
    from prd_v5.hdrunet import resolve_residual_adapter

    _, values = assignments()
    loads = []

    def loader(checkpoint, sha256, camera, **kwargs):
        loads.append((checkpoint, sha256, kwargs))
        return ResidualAdapter(
            predict=lambda *a: None,
            checkpoint_path=str(checkpoint),
            sha256=sha256,
            camera_sha256="c",
            source="fake",
            licence="fake",
            tile=kwargs["tile"],
            halo=kwargs["halo"],
        )

    settings = dict(
        enabled=values["ENABLE_LEARNED_RGB2RAW"],
        inverse_mode=values["INVERSE_ISP_MODE"],
        checkpoint=None,
        sha256=None,
        licence_accepted=False,
        experimental_accepted=False,
        tile=values["RAW_TILE"],
        halo=values["RAW_OVERLAP"],
        max_delta=values["HDRUNET_MAX_DELTA"],
        loader=loader,
    )
    settings.update(overrides)
    camera = CameraHypothesis(profile="SONY_A1II_LIKE", seed=1)
    adapter = settings.pop("adapter", None)
    return resolve_residual_adapter(adapter, camera, **settings), loads


def test_notebook_defaults_load_nothing():
    (adapter, loaded), loads = run_resolver()
    assert adapter is None and loaded is False and loads == []


def test_notebook_checkpoint_loads_with_notebook_tiling_through_loader():
    (adapter, loaded), loads = run_resolver(checkpoint="/content/h.pth", sha256="0" * 64)
    assert loaded and adapter.tile == 1024 and adapter.halo == 128
    assert loads[0][2]["tile"] == 1024 and loads[0][2]["halo"] == 128


def test_disabled_learned_branch_rejects_strict_mode_and_does_not_load():
    with pytest.raises(ValueError, match="disabled"):
        run_resolver(enabled=False, inverse_mode="LEARNED_RGB2RAW")
    (adapter, loaded), loads = run_resolver(enabled=False, checkpoint="/x", sha256="0" * 64)
    assert adapter is None and not loaded and loads == []
