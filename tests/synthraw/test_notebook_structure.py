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
        "SuperResolutionEngine",
        "HDRReconstructionEngine",
        "resolve_colour_space",
        "CameraProfileManager",
        "CameraGeometryEstimator",
        "ExposurePosteriorEstimator",
        "WhiteBalanceEstimator",
        "CameraColourModel",
        "InverseISPEngine",
        "LensSimulator",
        "OpticalBlurParameters",
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
    assert placeholders == set(SECTIONS[10:]) - {"Metadata construction", "DNG export"}


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


@pytest.mark.parametrize("mode", ["OUTPAINT_VERTICAL", "CROP_HORIZONTAL", "FIT_WITH_SENSOR_MARGINS"])
def test_sr_cell_writes_disk_products(notebook, tmp_path, mode):
    # Execute only the SR cell with a tiny image and bounded FAST_DEV geometry.
    import json
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    from synthraw.config import ExperimentConfig
    from synthraw.environment import EnvironmentReport, TileParameters
    from synthraw.geometry import TargetGeometryPlanner
    from synthraw.run_layout import RunLayout

    source = np.full((20, 48, 3), 128, dtype=np.uint8)
    source[:, :12, 0] = 64
    source_path = tmp_path / "source.png"
    Image.fromarray(source).save(source_path)
    config = ExperimentConfig(SR_BACKEND="LANCZOS", SENSOR_FIT_MODE=mode, FAST_DEV_RUN=True)
    run = RunLayout.create(tmp_path / "runs")
    geometry = TargetGeometryPlanner().plan(48, 20, config)
    namespace = dict(
        np=np,  # Imported by the prerequisite RAW gate cell.
        source_path=source_path,
        audit=SimpleNamespace(width=48, height=20),
        config=config,
        run=run,
        geometry=geometry,
        environment=EnvironmentReport(None, None, None, None, 1024**3, 1024**3, 1, TileParameters(16, 4, True, False)),
    )
    cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "sr_result = " in cell.source)
    exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    np.testing.assert_allclose(np.load(run.directory("intermediates") / "01_input_rgb.npy"), source / 255)
    assert isinstance(namespace["sr_rgb"], np.memmap)
    assert namespace["sr_rgb"].shape == (geometry.height, geometry.width, 3)
    for name in ("generated_detail_mask", "source_supported_mask"):
        assert np.load(run.directory("confidence") / f"{name}.npy").shape == (geometry.height, geometry.width)
    report = json.loads((run.directory("metadata") / "sr_reconstruction.json").read_text())
    assert report["tiled"]["backend"] == "LANCZOS"
    assert report["tiled"]["benchmark_tile_count"] == 3
    assert report["seams"]["residual_map_path"]
    provenance = json.loads((run.directory("metadata") / "sr_provenance.json").read_text())
    assert provenance[0]["provenance"] == "F"


@pytest.mark.parametrize("hdr_enabled", [True, False])
@pytest.mark.parametrize("has_seams", [True, False])
def test_hdr_cell_writes_tiny_reconstruction_products(notebook, tmp_path, capsys, hdr_enabled, has_seams):
    # Execute this cell alone with 8 x 12 pixels, never a notebook or model.
    import json
    from types import SimpleNamespace

    import matplotlib
    import numpy as np
    from PIL import Image

    from synthraw.config import ExperimentConfig
    from synthraw.hdr import HDRReconstructionEngine
    from synthraw.input_audit import InputInspector
    from synthraw.provenance import ProvenanceWriter
    from synthraw.run_layout import RunLayout

    matplotlib.use("Agg")
    rgb = np.full((8, 12, 3), 0.5, dtype=np.float32)
    rgb[0] = 1
    rgb[-1] = 0
    source = tmp_path / "input.png"
    Image.fromarray((rgb * 255).astype(np.uint8)).save(source)
    run = RunLayout.create(tmp_path / "runs")
    sr_support = np.ones((8, 12), dtype=bool)
    np.save(run.directory("confidence") / "source_supported_mask.npy", sr_support)
    seams = None
    if has_seams:
        seam_path = run.directory("validation") / "tile_seam_map.npy"
        np.save(seam_path, np.zeros((8, 12), dtype=np.float32))
        seams = SimpleNamespace(residual_map_path=seam_path)
    namespace = dict(
        Path=Path,
        ProvenanceWriter=ProvenanceWriter,
        audit=InputInspector().inspect(source),
        sr_rgb=rgb,
        sr_result=SimpleNamespace(seams=seams, seam_unavailable_reason="Single tile: no internal tile boundaries"),
        run=run,
        config=ExperimentConfig(HDR_ENABLED=hdr_enabled),
    )
    cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "hdr_engine = " in cell.source)
    exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    linear = np.load(run.directory("intermediates") / "03_scene_linear.npy")
    reference = HDRReconstructionEngine().linearize(rgb, tmp_path / "reference.npy")
    np.testing.assert_allclose(linear, reference)
    radiance = np.load(run.directory("intermediates") / "04_hdr_radiance.npy")
    assert radiance.shape == rgb.shape
    assert np.isfinite(radiance).all() and (radiance >= 0).all()
    np.testing.assert_allclose(radiance[0], linear[0] * (8 if hdr_enabled else 1))
    np.testing.assert_array_equal(np.load(run.directory("confidence") / "sr_source_supported_mask.npy"), sr_support)
    for name in ("source_supported_mask", "highlight_fabricated_mask", "shadow_fabricated_mask", "hdr_confidence"):
        assert np.load(run.directory("confidence") / f"{name}.npy").shape == (8, 12)
    assert not np.load(run.directory("confidence") / "source_supported_mask.npy")[0].any()
    for name in ("input_histograms", "clipping_masks", "tile_seam_map"):
        with Image.open(run.directory("validation") / f"{name}.png") as image:
            assert max(image.size) <= 700
            image.verify()
    report = json.loads((run.directory("metadata") / "hdr_reconstruction.json").read_text())
    assert report["generated_hdr_area_percentage"] == 25
    assert report["gamut_mapping"]["policy"] == "CLIP"
    assert (run.directory("metadata") / "colour_space.json").is_file()
    assert "Generated HDR area: 25.00%" in capsys.readouterr().out
    assert not list(run.directory("intermediates").glob("hdr_*"))


def test_no_camera_placeholders(notebook):
    assert all("Camera milestone placeholder" not in cell.source for cell in notebook.cells)


@pytest.mark.parametrize("optical_enabled", [False, True])
@pytest.mark.parametrize("manual_wb", [False, True])
def test_camera_cells_write_tiny_disk_products(notebook, tmp_path, capsys, optical_enabled, manual_wb):
    import json
    from types import SimpleNamespace

    import numpy as np

    from synthraw.camera_colour import CameraColourModel, InverseISPEngine
    from synthraw.config import ExperimentConfig
    from synthraw.provenance import ProvenanceWriter
    from synthraw.run_layout import RunLayout

    run = RunLayout.create(tmp_path)
    radiance = np.full((12, 16, 3), (0.4, 0.6, 0.8), dtype=np.float32)
    radiance[0] *= 8  # HDR headroom must reach inverse ISP unchanged.
    namespace = dict(
        np=np,
        run=run,
        config=ExperimentConfig(OPTICAL_SIMULATION=optical_enabled),
        hdr_radiance=radiance,
        geometry=SimpleNamespace(
            width=16,
            height=12,
            active_area=(0, 0, 12, 16),
            default_crop_origin=(0, 0),
            default_crop_size=(16, 12),
        ),
        ProvenanceWriter=ProvenanceWriter,
    )
    for index in (21, 23, 25):
        source = notebook.cells[index].source
        if manual_wb and index == 21:
            source = source.replace("MANUAL_WB = None", 'MANUAL_WB = {"manual_cct": 5500.0, "manual_tint": 0.1}')
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    expected = (
        InverseISPEngine()
        .reconstruct(radiance, CameraColourModel(), white_balance_gains=namespace["white_balance"].gains)
        .camera_rgb
    )
    camera = np.load(run.directory("intermediates") / "05_camera_native_rgb.npy", mmap_mode="r")
    optical = np.load(run.directory("intermediates") / "06_optical_sensor_rgb.npy", mmap_mode="r")
    np.testing.assert_allclose(camera, expected, rtol=1e-6)
    assert isinstance(namespace["camera_native_rgb"], np.memmap)
    assert isinstance(namespace["optical_sensor_rgb"], np.memmap)
    assert np.isfinite(optical).all()
    if not optical_enabled:
        np.testing.assert_array_equal(optical, camera)
    else:
        assert not np.array_equal(optical, camera)
    for name in (
        "camera_profile",
        "white_balance",
        "exposure_posterior",
        "camera_geometry",
        "inverse_isp",
        "optical_parameters",
    ):
        records = json.loads((run.directory("metadata") / f"{name}.json").read_text())
        assert records and all(record["provenance"] in "DIFM" for record in records)
    assert namespace["white_balance"].records["CCT"].user_override == manual_wb
    assert "Seconds       F-number   ISO       Weight" in capsys.readouterr().out
    assert not list(run.directory("intermediates").glob("synthraw-optics-*"))
