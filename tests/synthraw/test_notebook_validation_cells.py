"""Execute only the notebook's Validation suite, 2K/4K and packaging cells on a real tiny PipelineRun.

The earlier sections are replaced by running the same PipelineRun stages on a 48 x 20 PNG, so these cells
run against genuine run state without executing Colab code or downloading anything.
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.pipeline import STAGES, PipelineRun

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "synthetic_camera_raw_50mp_colab.ipynb"
CELL_IDS = {"validation": "d2942ebd", "comparison": "8764ae54", "packaging": "1c797b38"}


@pytest.fixture(scope="module")
def cells():
    import nbformat

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.id: cell.source for cell in notebook.cells}
    return {name: by_id[cell_id] for name, cell_id in CELL_IDS.items()}


@pytest.fixture
def notebook_state(tmp_path, monkeypatch):
    """Notebook globals after the 16-bit TIFF export section, from a real tiny PipelineRun."""

    def build(**settings):
        import matplotlib

        matplotlib.use("Agg")
        monkeypatch.setattr("synthraw.sony_registry.shutil.which", lambda _: None)
        monkeypatch.setitem(sys.modules, "google.colab", None)
        rows, columns = np.mgrid[0:20, 0:48]
        rgb = np.stack([columns * 5, rows * 12, (columns + rows) * 3], axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save(tmp_path / "scene.png")
        config = ExperimentConfig(
            FAST_DEV_RUN=True, SR_BACKEND="LANCZOS", OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", **settings
        )
        pipeline = PipelineRun.create(tmp_path / "scene.png", config, tmp_path / "runs", fast_dev_max_side=96)
        opened.append(pipeline)
        for name in STAGES[: STAGES.index("provenance")]:
            pipeline.run_stage(name)
        return {
            "__name__": "__main__",
            "Path": Path,
            "config": config,
            "environment": pipeline.environment,
            "pipeline": pipeline,
            "run": pipeline.run,
            "execution_log": pipeline.logger,
        }

    opened = []
    yield build
    for pipeline in opened:
        pipeline.close()


def run_cell(source, namespace):
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)


def test_failed_hard_gate_never_packages_a_success_report(cells, notebook_state, capsys):
    namespace = notebook_state()
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
    assert run_report["status"] == "FAILED" and run_report["validation"] == "FAILED"
    assert "hdr_negative_radiance" in run_report["hard_failures"]
    assert run_report["stages"] == list(STAGES)
    assert "VALIDATION FAILED" in capsys.readouterr().out


def test_disabled_validation_and_comparison_are_reported_not_run(cells, notebook_state):
    namespace = notebook_state(WRITE_VALIDATION_REPORT=False)
    run_cell(cells["validation"], namespace)
    assert namespace["validation_run"] is None
    root = namespace["run"].root
    assert (root / "metadata" / "provenance.json").is_file()
    assert not (root / "validation" / "validation_report.html").exists()
    source = cells["comparison"].replace(
        "RUN_SOURCE_RESOLUTION_COMPARISON = True", "RUN_SOURCE_RESOLUTION_COMPARISON = False"
    )
    run_cell(source, namespace)
    assert namespace["comparison"] is None
    assert not (root.parent / f"{root.name}_comparison").exists()
    run_cell(cells["packaging"], namespace)
    run_report = json.loads((root / "validation" / "run_report.json").read_text())
    assert run_report["status"] == "SUCCESS" and run_report["validation"] == "NOT RUN"
    assert run_report["comparison_2k_vs_4k"] is None
    # Off Colab the Drive mount fails: the run still succeeds and the report warns where the arrays live.
    (warning,) = run_report["warnings"]
    assert "exist only on this Colab runtime" in warning


def test_failed_drive_copy_warns_and_is_recorded_without_failing_the_run(cells, notebook_state, tmp_path, monkeypatch):
    (tmp_path / "drive" / "MyDrive").mkdir(parents=True)
    namespace = notebook_state(WRITE_VALIDATION_REPORT=False, DRIVE_MOUNT_PATH=str(tmp_path / "drive"))
    run_cell(cells["validation"], namespace)
    namespace["comparison"] = None

    def quota(*args, **kwargs):
        raise OSError(28, "Drive storage quota exceeded")

    monkeypatch.setattr("synthraw.bundle.shutil.copytree", quota)
    run_cell(cells["packaging"], namespace)
    run_report = json.loads((namespace["run"].root / "validation" / "run_report.json").read_text())
    assert run_report["status"] == "SUCCESS"
    (warning,) = run_report["warnings"]
    assert "quota exceeded" in warning and "exist only on this Colab runtime" in warning
    drive_zip = tmp_path / "drive" / "MyDrive" / "synthraw_runs" / namespace["bundle_path"].name
    assert namespace["delivery"]["drive_copy"] == str(drive_zip)
    # The delivered zip carries the same report, warning included.
    for bundle in (namespace["bundle_path"], drive_zip):
        with zipfile.ZipFile(bundle) as archive:
            delivered = json.loads(archive.read(f"{namespace['run'].root.name}/validation/run_report.json"))
        assert delivered == run_report
