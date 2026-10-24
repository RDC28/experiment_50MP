"""Phase 6 run collection on the captain's notebook 00 Colab runs (evidence/colab_runs) and tiny local runs."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from synthetic_raw_lab import evaluation_config as ec
from synthetic_raw_lab import night_processing as night
from synthetic_raw_lab import run_collection as rc
from synthetic_raw_lab import universal_config as uc
from synthetic_raw_lab import universal_pipeline as up

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "colab_runs"
PORTRAIT = "20261024T090241Z_40e30440_baselines"
NIGHT = "20261024T093611Z_5fe78cd3_baselines"
PARAMS = ec.load_evaluation_config(ROOT / "configs" / "evaluation_default.yaml").aggregation


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    return rc.collect([EVIDENCE], tmp_path_factory.mktemp("work"), PARAMS)


def test_the_captain_runs_are_recognised_and_labelled(evidence):
    assert [r.run_id for r in evidence] == [NIGHT, PORTRAIT]
    night_run, portrait = evidence
    assert portrait.notebook == "00" and portrait.label == "portrait (red saree, festival)"
    assert night_run.label == "night forest path" and night_run.has_report
    assert [o.name for o in portrait.outputs] == ["lanczos", "bicubic", "sr_conservative", "sr_balanced"]


def test_notebook_00_outputs_carry_the_measured_numbers(evidence):
    night_run, portrait = evidence
    by = {o.name: o for o in portrait.outputs}
    assert by["sr_conservative"].consistency["psnr_db"] == pytest.approx(66.91, abs=0.01)
    assert by["lanczos"].consistency["psnr_db"] == pytest.approx(55.23, abs=0.01)
    assert by["sr_conservative"].seam_passed is True
    assert len(by["sr_balanced"].detail["crop_hf_energy"]) == 4
    balanced_night = next(o for o in night_run.outputs if o.name == "sr_balanced")
    assert balanced_night.seam_passed is False  # the balanced mode failed the seam gate on the night image
    assert portrait.performance["gpu"] == "NVIDIA L4" and portrait.performance["total_seconds"] == pytest.approx(
        1121.9, abs=0.1
    )
    assert portrait.performance["sr_seconds"] == pytest.approx(395.07 + 401.67, abs=0.1)
    assert portrait.performance["output_megapixels"] == pytest.approx(49.96, abs=0.01)


def test_tables_have_one_row_per_output_and_run(evidence):
    header, rows = rc.outputs_table(evidence)
    assert len(rows) == 8 and header[4] == "PSNR dB"
    assert any(row[-1] == "FAIL" for row in rows)
    header, rows = rc.performance_table(evidence)
    assert len(rows) == 2 and rows[0][3] == "NVIDIA L4"
    table = rc.markdown_table(*rc.inventory_table(evidence))
    assert table.count("\n") == 3 and "—" not in table


def test_zips_and_folders_are_collected_once(tmp_path):
    archive = tmp_path / "colab_run.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in (EVIDENCE / PORTRAIT).iterdir():
            bundle.write(path, f"{PORTRAIT}/{path.name}")
    runs = rc.collect([archive, EVIDENCE], tmp_path / "work")
    assert sorted(r.run_id for r in runs) == sorted([NIGHT, PORTRAIT])
    drop = tmp_path / "drop"
    drop.mkdir()
    shutil.copy(archive, drop / "a.zip")
    (drop / "notes.json").write_text("[]")  # unrelated JSON is ignored
    assert [r.run_id for r in rc.collect([drop], tmp_path / "work2")] == [PORTRAIT]


def test_unsafe_zip_members_are_refused(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape/manifest.json", "{}")
    with pytest.raises(ValueError, match="outside the folder"):
        rc.collect([archive], tmp_path / "work")


def test_missing_paths_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        rc.collect([tmp_path / "nothing.zip"], tmp_path / "work")


def test_universal_night_run_links_its_delegated_notebook_02_run(project):
    path = night.write_night_scene(project / "night.png", 192, 128)
    config = uc.load_universal_config(
        ROOT / "configs" / "universal_default.yaml",
        target_megapixels=0.098304,
        crop_size=48,
        review_max_edge=256,
        output_root=str(project / "outputs"),
        reports_root=str(project / "reports"),
        inputs_root=str(project / "inputs"),
        super_resolution={"enabled": False},
        analysis={"face_detection": False},
        human={"profile": str(ROOT / uc.HUMAN_PROFILE), "overrides": {}},
        night={"profile": str(ROOT / uc.NIGHT_PROFILE), "overrides": {}},
    )
    result = up.run_universal(path, config, project, log=lambda _: None)
    archive = up.archive_reports(result.manifest, result.reports_dir)
    runs = rc.collect([archive], project / "work")
    by = {r.notebook: r for r in runs}
    assert set(by) == {"02", "03"} and by["02"].parent == by["03"].run_id
    universal_final = next(o for o in by["03"].outputs if o.name == "universal_final")
    assert universal_final.detail["category"] == "night" and universal_final.consistency["psnr_db"] > 20
    night_final = next(o for o in by["02"].outputs if o.name == "night_final")
    assert night_final.detail["metrics_before"] and night_final.tiff
    header, rows = rc.night_table(runs)
    assert any("shadow chroma" in row[3] for row in rows)


def test_review_sheet_from_masters_or_previews(tmp_path, evidence):
    _, portrait = evidence
    run = rc.Run(**{**portrait.__dict__, "folder": tmp_path / "run"})
    figures = run.folder / "figures"
    figures.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("source", "lanczos", "sr_conservative"):
        Image.fromarray((rng.random((300, 700, 3)) * 255).astype(np.uint8)).save(figures / f"{name}_preview.jpg")
    run.outputs = [rc.normalise(portrait.manifest, run.folder, PARAMS).outputs[i] for i in (0, 2)]
    masters = tmp_path / "drive"
    masters.mkdir()
    master_name = Path(run.outputs[1].tiff).name
    import tifffile

    tifffile.imwrite(masters / master_name, (rng.random((460, 1085, 3)) * 65535).astype(np.uint16))
    sheet = rc.review_sheet([run], tmp_path / "sheet.png", PARAMS, masters_roots=[masters])
    assert Path(sheet["path"]).exists() and len(sheet["crops"]) == 4
    scales = {row["row"]: row.get("scale") for row in sheet["rows"]}
    assert scales["00 lanczos (preview)"] == "preview" and scales["00 sr_conservative"] == "100%"
    assert sheet["rows"][0]["row"] == "source"


def test_configured_review_crops_replace_the_recorded_ones(evidence):
    params = ec.AggregationParams(review_crops=((0.1, 0.1, 0.2, 0.2),))
    assert rc.crop_boxes(evidence, params) == [("crop_1", (0.1, 0.1, 0.2, 0.2))]
    recorded = rc.crop_boxes(evidence, PARAMS)
    assert 1 <= len(recorded) <= 4 and all(0 <= v <= 1 for _, box in recorded for v in box)


def test_locate_finds_a_master_by_name_under_a_root(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    target = tmp_path / "a" / "b" / "final.tif"
    target.write_bytes(b"x")
    assert rc.locate("/content/gone/final.tif", [tmp_path]) == target
    assert rc.locate(None) is None and rc.locate("/content/gone/other.tif", [tmp_path]) is None


def test_manifests_with_unknown_schemas_are_ignored(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"schema": "other/1", "run_id": "x"}))
    assert rc.collect([tmp_path], tmp_path / "work") == []
