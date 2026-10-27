"""Runtime > Run all of the Colab notebook on a tiny PNG with Colab mocked; never downloads anything.

Every cell after the pip install runs, in order, on a 48 x 20 PNG with FAST_DEV_RUN at 96 px, so the
whole notebook finishes in seconds on a laptop.
"""

import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rawpy
import tifffile
from PIL import Image
from test_synthraw_pipeline import PRD_TREE

from synthraw.bundle import LARGE_ARRAY_DIRECTORIES
from synthraw.pipeline import STAGES

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "synthetic_camera_raw_50mp_colab.ipynb"


@pytest.fixture
def colab(monkeypatch, tmp_path):
    """Fake google.colab: upload saves a 48 x 20 PNG to the working directory, download is recorded."""
    upload_dir = tmp_path / "content"
    upload_dir.mkdir()
    monkeypatch.chdir(upload_dir)
    rows, columns = np.mgrid[0:20, 0:48]
    rgb = np.stack([columns * 5, rows * 12, (columns + rows) * 3], axis=-1).astype(np.uint8)
    buffer = tmp_path / "scene.png"
    Image.fromarray(rgb).save(buffer)
    downloads = []

    def upload():
        (upload_dir / "scene.png").write_bytes(buffer.read_bytes())
        return {"scene.png": buffer.read_bytes()}

    files = types.SimpleNamespace(upload=upload, download=downloads.append)
    google = types.ModuleType("google")
    module = types.ModuleType("google.colab")
    module.files = files
    module.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = module
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", module)
    return types.SimpleNamespace(downloads=downloads, buffer=buffer)


@pytest.mark.parametrize("geometry", ["SONY_A1II_NATIVE", "CINEMA_50MP_VIRTUAL"])
def test_run_all_produces_prd_tree_and_bundle(colab, monkeypatch, tmp_path, geometry):
    import matplotlib
    import nbformat

    from synthraw.exporters import adobe_rgb_icc
    from synthraw.provenance import SCIENTIFIC_BOUNDARY
    from synthraw.run_layout import RunLayout

    matplotlib.use("Agg")
    runs = tmp_path / "runs"
    create = RunLayout.create.__func__
    # Redirect only the notebook's Colab run root; the 2K/4K comparison run keeps its own folder.
    monkeypatch.setattr(
        RunLayout,
        "create",
        classmethod(
            lambda cls, out, **kwargs: create(cls, runs if str(out) == "/content/synthraw_runs" else out, **kwargs)
        ),
    )
    drive = tmp_path / "drive"
    (drive / "MyDrive").mkdir(parents=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(code[1:], start=1):  # Skip the Colab pip install.
        exec(compile(cell.source, f"{NOTEBOOK}:code{index}", "exec"), namespace)
        if "CONFIG_VALUES" in namespace and index == 1:
            namespace["CONFIG_VALUES"].update(
                FAST_DEV_RUN=True,
                OUTPUT_GEOMETRY=geometry,
                SR_BACKEND="LANCZOS",
                INPUT_MODE="UPLOAD",
                DRIVE_MOUNT_PATH=str(drive),
            )
            namespace["controls"] = {}
        if index == 4:  # Input audit: FAST_DEV_RUN at 96 px keeps every array tiny on the Mac.
            namespace["pipeline"].options["fast_dev_max_side"] = 96

    (root,) = (path for path in runs.iterdir() if path.is_dir() and not path.name.endswith("_comparison"))
    produced = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert PRD_TREE <= produced
    assert not any(name.startswith("raw/.") for name in produced)
    run_report = json.loads((root / "validation" / "run_report.json").read_text())
    assert run_report["status"] == "SUCCESS" and run_report["validation"] == "PASSED"
    assert run_report["comparison_2k_vs_4k"]["status"] == "COMPLETED"
    # The notebook runs the same stage code as run_pipeline, in the same order.
    assert run_report["stages"] == list(STAGES)
    comparison = json.loads((root / "validation" / "comparison_2k_vs_4k.json").read_text())
    assert comparison["locked_mismatches"] == {}
    for name in ("validation_results.json", "validation_report.html", "comparison_2k_vs_4k.html"):
        assert f"validation/{name}" in produced

    mosaic = np.load(root / "raw" / "sensor_mosaic_uint16.npy")
    with rawpy.imread(str(root / "raw" / "synthetic_50mp.dng")) as raw:
        np.testing.assert_array_equal(raw.raw_image_visible, mosaic)
    with tifffile.TiffFile(root / "masters" / "developed_50mp_16bit.tif") as tiff:
        page = tiff.pages[0]
        assert page.dtype == np.uint16 and page.shape == (*mosaic.shape, 3)
        assert page.tags["InterColorProfile"].value == adobe_rgb_icc()

    provenance = json.loads((root / "metadata" / "provenance.json").read_text())
    assert provenance["scientific_boundary"] == SCIENTIFIC_BOUNDARY
    assert provenance["dng_identity"]["Make"] == "Synthetic"
    assert provenance["source"]["file"] == "source/source_original.png"
    assert provenance["c2pa"]["status"].startswith("C2PA")
    assert "raw/synthetic_50mp.dng" in provenance["outputs"]
    metadata = json.loads((root / "metadata" / "metadata.json").read_text())
    assert {"dng", "exposure_posterior", "optics", "camera_profile", "sensor_profile"} <= set(metadata["groups"])
    log = (root / "logs" / "execution.log").read_text()
    assert "Input audit" in log and "Bundle written" in log

    bundle = runs / f"{root.name}.zip"
    assert colab.downloads == [str(bundle)]
    drive_runs = drive / "MyDrive" / "synthraw_runs"
    assert (drive_runs / bundle.name).read_bytes() == bundle.read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        names = {name.removeprefix(f"{root.name}/") for name in archive.namelist()}
        assert archive.read(f"{root.name}/source/source_original.png") == colab.buffer.read_bytes()
    large = {name for name in produced if name.split("/")[0] in LARGE_ARRAY_DIRECTORIES and name.endswith(".npy")}
    assert large and not large & names
    assert names == (produced - large) | {"logs/bundle_manifest.json"}
    # The browser zip is light; the Drive copy of the run folder keeps every full-resolution intermediate.
    for name in large | {"raw/synthetic_50mp.dng"}:
        assert (drive_runs / root.name / name).read_bytes() == (root / name).read_bytes()
