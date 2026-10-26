"""Static notebook checks for PRD_v3 architecture; never execute Colab code."""

import ast
import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "synthetic_camera_raw_50mp_colab.ipynb"
SECTIONS = [
    "Environment + reproducibility",
    "Input audit",
    "Profile selection",
    "Target geometry",
    "Source diagnostics",
    "SR reconstruction",
    "HDR / scene-linear reconstruction",
    "Camera geometry + exposure inference",
    "Inverse ISP / camera colour",
    "Optical forward model",
    "Sensor/electron simulation",
    "CFA construction",
    "ADC quantisation",
    "Metadata construction",
    "DNG export",
    "RAW round-trip development",
    "16-bit TIFF export",
    "Validation suite",
    "2K/4K comparison",
    "Provenance + report packaging",
]


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_schema_and_order(notebook):
    nbformat.validate(notebook)
    headers = [
        line.removeprefix("## ")
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    assert headers == SECTIONS


def test_install_cell_is_first_code_and_ref_is_explicit(notebook):
    source = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    tree = ast.parse(source)
    assignments = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    ref = assignments["REPO_REF"]
    assert ref == "fm/exp50mp-v3-synthetic-raw" or re.fullmatch(r"[0-9a-fA-F]{40}", ref)
    assert "git+https://github.com/RDC28/experiment_50MP.git@{REPO_REF}" in source
    assert any(
        isinstance(node, ast.List)
        and [value.value for value in node.elts if isinstance(value, ast.Constant)][:3] == ["-m", "pip", "install"]
        for node in ast.walk(tree)
    )
    assert "synthraw" in source
    assert "commit SHA" in source


def test_all_cells_are_unexecuted_valid_python(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
            compile(cell.source, str(NOTEBOOK), "exec")


def test_foundation_uses_public_package_interfaces(notebook):
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("synthraw.")
        for alias in node.names
    }
    assert imports == {
        "CFA_PATTERNS",
        "DNGMetadataBuilder",
        "DNGWriter",
        "LibRawValidator",
        "ExperimentConfig",
        "EnvironmentInspector",
        "InputInspector",
        "ProvenanceWriter",
        "SonyARWEmulationBuilder",
        "SonyMetadataRegistry",
        "RandomStreams",
        "RunLayout",
        "TargetGeometryPlanner",
    }
    assert "fields(ExperimentConfig)" in source
    assert "get_type_hints(ExperimentConfig)" in source
    assert "ExperimentConfig.from_dict(values)" in source
    assert "ExperimentConfig.from_dict(CONFIG_VALUES)" in source
    assert "except ImportError:" in source
    for fragment in ("files.upload()", "drive.mount(", "InputInspector().inspect(", "TargetGeometryPlanner().plan("):
        assert fragment in source


def test_later_sections_identify_milestone_placeholders(notebook):
    section = None
    placeholders = set()
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            section = cell.source.removeprefix("## ")
        elif cell.cell_type == "code" and "milestone placeholder:" in cell.source:
            placeholders.add(section)
    assert placeholders == set(SECTIONS[5:]) - {"Metadata construction", "DNG export"}


def test_scientific_boundary_is_visible(notebook):
    introduction = notebook.cells[0].source
    for required in ("synthetic sensor reconstruction", "hypothetical", "Synthetic", "D/I/F/M", "GPS"):
        assert required in introduction


def raw_gate_cell(notebook):
    return next(cell for cell in notebook.cells if cell.cell_type == "code" and "def run_raw_gate():" in cell.source)


def test_raw_gate_precedes_input_audit(notebook):
    gate = raw_gate_cell(notebook)
    audit = next(
        cell for cell in notebook.cells if cell.cell_type == "code" and "config = read_configuration()" in cell.source
    )
    assert notebook.cells.index(gate) < notebook.cells.index(audit)
    assert 'globals().get("RAW_GATE_PASSED", False)' in audit.source


def test_tiny_raw_gate_cell_with_real_libraw(notebook):
    # Execute only the bounded fixture cell, never the notebook or Colab setup.
    from synthraw.config import ExperimentConfig

    namespace = {"ExperimentConfig": ExperimentConfig}
    exec(compile(raw_gate_cell(notebook).source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["RAW_GATE_PASSED"] is True


def test_tiny_raw_gate_failure_blocks_progress_and_cleans_up(notebook, monkeypatch):
    from synthraw.config import ExperimentConfig
    from synthraw.libraw_validator import HardValidationError, LibRawValidator

    paths = []

    def fail(self, path, *args, **kwargs):
        paths.append(path)
        raise HardValidationError("decoder rejected fixture")

    monkeypatch.setattr(LibRawValidator, "validate", fail)
    namespace = {"ExperimentConfig": ExperimentConfig, "RAW_GATE_PASSED": True}
    with pytest.raises(HardValidationError, match="decoder rejected fixture"):
        exec(compile(raw_gate_cell(notebook).source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["RAW_GATE_PASSED"] is False
    assert len(paths) == 1
    assert not paths[0].parent.exists()


def test_metadata_cell_writes_provenance_and_sidecars(notebook, tmp_path, monkeypatch):
    import json
    from dataclasses import replace

    from synthraw.config import ExperimentConfig
    from synthraw.geometry import TargetGeometryPlanner
    from synthraw.run_layout import RunLayout

    monkeypatch.setattr("synthraw.sony_registry.shutil.which", lambda _: None)
    cell = next(cell for cell in notebook.cells if cell.id == "8176dfe4")
    namespace = {}
    exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["dng_metadata"] is None
    geometry = replace(
        TargetGeometryPlanner().plan(64, 48),
        width=64,
        height=48,
        active_area=(0, 0, 48, 64),
        default_crop_origin=(0, 0),
        default_crop_size=(64, 48),
    )
    run = RunLayout.create(tmp_path)
    sensor = dict(
        cfa_phase="GBRG",
        black_level=512,
        white_level=16383,
        color_matrix1=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        color_matrix2=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        as_shot_neutral=(0.5, 1, 0.75),
        noise_profile=(0.002, 0.000003),
    )
    build = namespace["construct_metadata"]
    with pytest.raises(ValueError, match="Missing sensor metadata"):
        build(geometry, ExperimentConfig(), run, sensor_metadata={})
    assert not list(run.directory("metadata").iterdir())
    records = build(geometry, ExperimentConfig(), run, sensor_metadata=sensor)
    fields = json.loads((run.directory("metadata") / "dng_metadata.json").read_text())
    noise = next(field for field in fields if field["field"] == "NoiseProfile")
    assert noise["value"] == list(sensor["noise_profile"])
    assert noise["provenance"] == "D"
    assert records["Make"].value == "Synthetic"
    registry = json.loads((run.directory("metadata") / "sony_tag_registry.json").read_text())
    sidecar = json.loads((run.directory("metadata") / "sony_arw_emulation.json").read_text())
    assert registry["seed_only"] is True
    assert len(sidecar["fields"]) == len(registry["fields"])
    assert all(field["generated_value"] is not None or field["unsupported_reason"] for field in sidecar["fields"])


@pytest.mark.parametrize("failure", [None, "writer", "libraw"])
def test_notebook_export_publishes_only_validated_raw(notebook, tmp_path, monkeypatch, failure):
    from dataclasses import replace

    import numpy as np

    from synthraw.config import ExperimentConfig
    from synthraw.dng_metadata import DNGMetadataBuilder
    from synthraw.dng_writer import DNGWriter
    from synthraw.geometry import TargetGeometryPlanner
    from synthraw.libraw_validator import HardValidationError, LibRawValidator
    from synthraw.run_layout import RunLayout

    config = ExperimentConfig()
    run = RunLayout.create(tmp_path)
    cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "def export_dng(" in cell.source)
    namespace = {"config": config, "dng_metadata": None}
    exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    assert namespace["dng_path"] is None
    export = namespace["export_dng"]
    assert export(None, None, replace(config, WRITE_DNG=False), run) is None
    with pytest.raises(ValueError, match="Construct sensor metadata"):
        export(None, None, config, run)
    geometry = replace(
        TargetGeometryPlanner().plan(64, 48),
        width=64,
        height=48,
        active_area=(0, 0, 48, 64),
        default_crop_origin=(0, 0),
        default_crop_size=(64, 48),
    )
    records = DNGMetadataBuilder().build(geometry, config)
    mosaic = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + 512
    destination = run.directory("raw") / "synthetic_sensor.dng"
    if failure:
        destination.write_bytes(b"previous validated output")

        def fail(*args, **kwargs):
            if failure == "writer":
                args[1].write_bytes(b"partial")
            raise HardValidationError("export rejected")

        monkeypatch.setattr(
            DNGWriter if failure == "writer" else LibRawValidator, "write" if failure == "writer" else "validate", fail
        )
        with pytest.raises(HardValidationError, match="export rejected"):
            export(mosaic, records, config, run)
        assert destination.read_bytes() == b"previous validated output"
    else:
        assert export(mosaic, records, config, run) == destination
        LibRawValidator().validate(destination, mosaic, records, develop=True)
    assert list(run.directory("raw").iterdir()) == [destination]
