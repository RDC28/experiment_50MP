"""Execute Notebook 01 as a user would, with a small synthetic crowd."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SYNTHETIC_RAW_LAB_HEAVY_TESTS") != "1",
    reason="notebook execution is opt-in: set SYNTHETIC_RAW_LAB_HEAVY_TESTS=1",
)

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_Human_Crowd_50MP.ipynb"


def test_human_notebook_synthetic_quick(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameters = next(cell for cell in notebook.cells if "parameters" in cell.metadata.get("tags", []))
    parameters.source += f"""
INPUT_MODE = "synthetic"
FACE_DETECTOR = "none"
SR_BACKEND = "lanczos_stand_in"
GENERAL_SR_STRENGTH = 0.5
EDGE_SHARPENING = 0.2
SKIN_TEXTURE_SMOOTHING = 0.4
GRAIN = 0.3
TONAL_CONTRAST = 0.15
COLOR_BALANCE = [1.02, 1.0, 0.98]
RUNTIME_PRESET = "quick"
OUTPUT_ROOT = {str(tmp_path / "outputs")!r}
REPORTS_ROOT = {str(tmp_path / "reports")!r}
INPUTS_ROOT = {str(tmp_path / "inputs")!r}
"""
    nbclient.NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    manifests = list((tmp_path / "reports").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "complete"
    assert manifest["config"]["general_sr_strength"] == 0.5
    assert manifest["config"]["edge_sharpening"] == 0.2
    assert {"general_sr_strength", "edge_sharpening"} <= {s["name"] for s in manifest["stages"]}
    assert manifest["config"]["tonal_contrast"] == 0.15
    assert manifest["config"]["colour_balance"] == [1.02, 1.0, 0.98]
    controls = {s["name"]: s for s in manifest["stages"]}
    assert controls["skin_texture_smoothing"]["parameters"]["strength"] == 0.4
    assert controls["skin_texture_smoothing"]["classification"] == "Simulation"
    grain = controls["grain"]
    assert grain["classification"] == "Simulation"
    assert grain["parameters"]["strength"] == 0.3
    assert grain["parameters"]["high_frequency_energy_after"] > grain["parameters"]["high_frequency_energy_before"]
    assert controls["tonal_contrast"]["classification"] == "Simulation"
    assert controls["colour_balance"]["parameters"]["skin_tone_shift_ciede2000"] > 0
    assert manifest["config"]["face_detector"] == "none"
    assert manifest["models"] == []
    assert manifest["notebook_version"] == "01_Human_Crowd_50MP/1.1"
    assert [row["face_count"] for row in manifest["identity"]["per_tier"]] == [1, 1, 1]
    assert len(manifest["config"]["manual_face_boxes"]) == 3
    assert manifest["config"]["primary_subject_index"] is None
    assert manifest["primary_subject"]["index"] == 0
    assert manifest["primary_subject"]["rule"] == "highest_source_tier_then_largest_area_then_first_observation"
    assert len(manifest["comparisons"]["figures"]) == 9
    assert "face_comparison_grid" in manifest["comparisons"]["figures"]
    for figure in manifest["comparisons"]["figures"].values():
        assert Path(figure["path"]).is_file()
    for kind, sample_type in (("tiff16", "uint16"), ("exr", "float16")):
        exported = manifest["exports"][kind]
        assert exported["valid"] and exported["sample_type"] == sample_type
        assert Path(exported["path"]).is_file()
    assert Path(manifest["review"]["path"]).is_file()
    report = manifests[0].with_name("report.md").read_text()
    assert "Tonal and colour controls (Simulation)" in report
    assert "skin_tone_shift_ciede2000" in report
    assert "Per-tier identity metrics" in report
    assert "requires human review" in report
    assert "no automatic detector configured" in report
    assert "Skin controls touch estimated exposed neck, hand and arm skin" in report
    assert "stand-in" in report
    assert list((tmp_path / "reports").glob("*.zip"))
