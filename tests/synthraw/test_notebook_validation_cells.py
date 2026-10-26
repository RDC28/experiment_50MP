"""Execute only the notebook's RAW round-trip, Validation suite, 2K/4K and packaging cells on a tiny run.

The notebook objects these cells consume are taken from a real 48 x 20 run_pipeline state, so the
cells run against genuine records and arrays without executing the notebook or any Colab code.
"""

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.libraw_validator import LibRawValidator
from synthraw.pipeline import _PipelineRun, run_pipeline
from synthraw.run_layout import RunLayout

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "synthetic_camera_raw_50mp_colab.ipynb"
CELL_IDS = {"roundtrip": "da824d07", "validation": "d2942ebd", "comparison": "8764ae54", "packaging": "1c797b38"}


@pytest.fixture(scope="module")
def cells():
    import nbformat

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.id: cell.source for cell in notebook.cells}
    return {name: by_id[cell_id] for name, cell_id in CELL_IDS.items()}


@pytest.fixture(scope="module")
def pipeline_state(tmp_path_factory):
    """One tiny run plus the in-memory pipeline state the notebook cells would hold."""
    base = tmp_path_factory.mktemp("nbcells")
    rows, columns = np.mgrid[0:20, 0:48]
    rgb = np.stack([columns * 5, rows * 12, (columns + rows) * 3], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(base / "scene.png")
    config = ExperimentConfig(FAST_DEV_RUN=True, SR_BACKEND="LANCZOS", OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL")
    captured = []
    original_close = _PipelineRun.close

    def close(self):
        captured.append(self)
        original_close(self)

    _PipelineRun.close = close
    try:
        result = run_pipeline(base / "scene.png", config, base / "runs", fast_dev_max_side=96)
    finally:
        _PipelineRun.close = original_close
    return base, result, captured[0]


def namespace_for(pipeline_state, tmp_path):
    """Notebook globals after the TIFF export cell, built from a copy of the tiny run."""
    base, result, state = pipeline_state
    root = tmp_path / "runs" / result.root.name
    shutil.copytree(result.root, root)
    groups = state.groups
    by_field = {name: {record.field: record for record in groups[name]} for name in groups}
    logger = logging.getLogger(f"synthraw.test.{tmp_path.name}")
    logger.addHandler(logging.NullHandler())
    adc_mosaic = np.load(root / "raw" / "sensor_mosaic_uint16.npy", mmap_mode="r")
    return {
        "__name__": "__main__",
        "Path": Path,
        "LibRawValidator": LibRawValidator,
        "config": state.config,
        "run": RunLayout(root),
        "environment": state.environment,
        "audit": state.audit,
        "geometry": state.geometry_plan,
        "source_original": root / "source" / state.source_original.name,
        "execution_log": logger,
        "adc_mosaic": adc_mosaic,
        "dng_path": root / "raw" / "synthetic_50mp.dng",
        "dng_metadata": state.dng_records,
        "camera_profile": state.camera_profile,
        "sr_result": state.sr_result,
        "colour_resolution": SimpleNamespace(records=lambda: groups["colour_space"]),
        "white_balance": state.white_balance,
        "exposure_records": state.exposure.records,
        "camera_geometry": SimpleNamespace(records=by_field["camera_geometry"], fallback_reason=None),
        "inverse_isp_records": by_field["inverse_isp"],
        "colour_model": state.colour_model,
        "spectral_records": [record for record in groups["colour_model"] if record.field.startswith("spectral_")],
        "optical_records": state.optical_records,
        "hdr_result": state.hdr_report,
        "tiff_path": root / "masters" / "developed_50mp_16bit.tif",
        "preview_path": root / "masters" / "preview_8bit.png",
        "ACCEPT_SR_MODEL_LICENCE": False,
        "SCENE_EV100": 10.0,
        "MANUAL_EXPOSURE": None,
        "MANUAL_WB": None,
        "MANUAL_FOV_DEGREES": None,
        "SENSOR_EXPOSURE_SCALE": 1.0,
        "SENSOR_STRIP_ROWS": 128,
    }


def run_cell(source, namespace):
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)


def test_cells_validate_compare_and_package_a_tiny_run(cells, pipeline_state, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setattr("synthraw.sony_registry.shutil.which", lambda _: None)
    _, result, _ = pipeline_state
    namespace = namespace_for(pipeline_state, tmp_path)
    root = namespace["run"].root
    expected_metadata = json.loads((result.root / "metadata" / "metadata.json").read_text())
    (root / "metadata" / "metadata.json").unlink()
    (root / "validation" / "run_report.json").unlink()

    run_cell(cells["roundtrip"], namespace)
    assert json.loads((root / "validation" / "libraw_roundtrip.json").read_text())["libraw_reopened"] is True

    run_cell(cells["validation"], namespace)
    # The notebook freezes the same metadata.json that run_pipeline writes.
    assert json.loads((root / "metadata" / "metadata.json").read_text()) == expected_metadata
    assert namespace["fallbacks"] == []
    validation = json.loads((root / "validation" / "validation_results.json").read_text())
    assert validation["status"] == namespace["validation_run"].status == "PASSED"
    report = namespace["validation_report_path"]
    assert report == root / "validation" / "validation_report.html"
    assert "PASSED" in report.read_text()

    run_cell(cells["comparison"], namespace)
    comparison = namespace["comparison"]
    assert comparison.status == "COMPLETED"
    assert comparison.locked_mismatches == {}
    assert comparison.low_source_derived is True
    assert comparison.high.root == root
    assert (root / "validation" / "comparison_2k_vs_4k.html").is_file()
    data = json.loads((root / "validation" / "comparison_2k_vs_4k.json").read_text())
    assert data["low_source_size"] == [24, 10] and data["high_source_size"] == [48, 20]
    # The second run lives beside the main run, so the bundle stays one run.
    assert comparison.low.root.parent == root.parent / f"{root.name}_comparison"

    run_cell(cells["packaging"], namespace)
    run_report = json.loads((root / "validation" / "run_report.json").read_text())
    assert run_report["status"] == "SUCCESS" and run_report["validation"] == "PASSED"
    assert run_report["comparison_2k_vs_4k"]["status"] == "COMPLETED"
    assert namespace["bundle_path"].is_file()


def test_failed_hard_gate_never_packages_a_success_report(cells, pipeline_state, tmp_path, capsys):
    import matplotlib

    matplotlib.use("Agg")
    namespace = namespace_for(pipeline_state, tmp_path)
    root = namespace["run"].root
    radiance = np.load(root / "intermediates" / "04_hdr_radiance.npy", mmap_mode="r+")
    radiance[3, 5, 1] = -0.5
    radiance.flush()
    del radiance

    run_cell(cells["validation"], namespace)
    assert namespace["validation_run"].status == "FAILED"
    assert "FAILED" in namespace["validation_report_path"].read_text()
    namespace["comparison"] = None
    run_cell(cells["packaging"], namespace)
    run_report = json.loads((root / "validation" / "run_report.json").read_text())
    assert run_report["status"] == "FAILED"
    assert "hdr_negative_radiance" in run_report["hard_failures"]
    assert "VALIDATION FAILED" in capsys.readouterr().out


def test_disabled_validation_and_comparison_are_reported_not_run(cells, pipeline_state, tmp_path):
    from dataclasses import replace

    namespace = namespace_for(pipeline_state, tmp_path)
    namespace["config"] = replace(namespace["config"], WRITE_VALIDATION_REPORT=False)
    run_cell(cells["validation"], namespace)
    assert namespace["validation_run"] is None
    root = namespace["run"].root
    assert not (root / "validation" / "validation_report.html").exists()
    source = cells["comparison"].replace(
        "RUN_SOURCE_RESOLUTION_COMPARISON = True", "RUN_SOURCE_RESOLUTION_COMPARISON = False"
    )
    run_cell(source, namespace)
    assert namespace["comparison"] is None
    run_cell(cells["packaging"], namespace)
    run_report = json.loads((root / "validation" / "run_report.json").read_text())
    assert run_report["status"] == "SUCCESS" and run_report["validation"] == "NOT RUN"
    assert run_report["comparison_2k_vs_4k"] is None
