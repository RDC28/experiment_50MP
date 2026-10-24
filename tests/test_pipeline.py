import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from synthetic_raw_lab import environment, exporters, pipeline
from synthetic_raw_lab.config import ExperimentConfig, load_config
from synthetic_raw_lab.test_images import make_test_image, write_test_image

ALLOWED_CLASSES = {pipeline.PRESERVATION, pipeline.INTERPOLATION, pipeline.PACKAGING, pipeline.MEASUREMENT}


@pytest.fixture
def source(tmp_path):
    return write_test_image(tmp_path / "scene.png", 320, 180)


def small_config(**overrides) -> ExperimentConfig:
    values = {
        "runtime_preset": "quick",
        "quick_target_megapixels": 0.4,
        "crop_size": 48,
        "crop_count": 2,
        "review_max_edge": 256,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_quick_run_produces_validated_masters_and_report(source, project):
    result = pipeline.run(source, small_config(), project, log=lambda _: None)
    manifest = result.manifest

    assert manifest["status"] == "complete"
    target = manifest["target_resolution"]
    assert abs(target["pixel_count_error"]) <= 0.001
    assert set(manifest["baselines"]) == {"lanczos", "bicubic"}
    for summary in manifest["baselines"].values():
        assert summary["exports"]["tiff16"]["valid"] and summary["exports"]["exr"]["valid"]
        assert Path(summary["exports"]["tiff16"]["path"]).exists()
        assert summary["source_consistency"]["psnr_db"] > 20
        assert summary["source_consistency"]["ssim"] > 0.8
    assert {s["classification"] for s in manifest["stages"]} <= ALLOWED_CLASSES
    assert all(s["error"] is None for s in manifest["stages"])

    assert result.masters_dir == project / "outputs" / "universal" / result.run_id
    assert result.reports_dir == project / "reports" / result.run_id
    assert json.loads(result.manifest_path.read_text())["run_id"] == result.run_id
    report = result.report_path.read_text()
    for heading in (
        "Run summary",
        "Input validation",
        "Processing log",
        "Visual results",
        "Quantitative results",
        "Information provenance",
        "Conclusions",
        "Provenance record",
    ):
        assert heading in report
    assert "Phase 0 exit criterion not assessed" in report
    subject_crop = manifest["crops"][0]
    assert subject_crop["name"] in {"face", "face_eyes", "subject", "center"} and subject_crop["label"]
    assert manifest["subject"]["detector"] and "classification" in manifest["subject"]
    for figure in (
        "source_histograms.png",
        "source_preview.jpg",
        "lanczos_preview.jpg",
        "delta_e_bicubic.png",
        f"crop_{subject_crop['name']}.png",
        "crop_detail_1.png",
        "final_preview.jpg",
    ):
        assert (result.reports_dir / "figures" / figure).exists()
    final = manifest["final_output"]
    assert final["method"] == "lanczos" and final["rule"] == "auto"
    lanczos = manifest["baselines"]["lanczos"]["exports"]
    for kind in ("tiff16", "exr"):
        path = Path(final["files"][kind]["path"])
        assert path.parent == result.masters_dir / "final" and path.name.startswith("final_")
        assert final["files"][kind]["file_sha256"] == lanczos[kind]["file_sha256"]
    assert "### Final 50 MP image" in report and "![Final image](figures/final_preview.jpg)" in report
    assert "Final 50 MP image: lanczos" in report
    assert set(manifest["output_checksums"]) == {Path(p).name for p in _export_paths(manifest)}
    assert (project / "inputs" / "universal").is_dir()


def test_research_preset_checks_reproducibility_and_benchmarks_memory(source, project):
    config = small_config(
        runtime_preset="research",
        target_megapixels=0.3,
        baselines=("bicubic",),
        export_exr=False,
        save_intermediates=False,
    )
    manifest = pipeline.run(source, config, project, log=lambda _: None).manifest
    assert manifest["baselines"]["bicubic"]["reproducibility"]["identical"] is True
    assert len(manifest["memory"]["benchmark"]) == 6
    assert "exr" not in manifest["baselines"]["bicubic"]["exports"]
    assert "intermediates" not in manifest


def test_failed_stage_keeps_earlier_outputs_and_reports(source, project, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(exporters, "write_tiff16", broken)
    with pytest.raises(pipeline.StageError) as failure:
        pipeline.run(source, small_config(), project, log=lambda _: None)

    assert failure.value.stage == "export_tiff16_lanczos"
    manifest = json.loads(failure.value.manifest_path.read_text())
    assert manifest["status"] == "failed in stage export_tiff16_lanczos"
    assert manifest["stages"][-1]["error"] == "OSError: disk full"
    partial = manifest["baselines"]["lanczos"]
    assert partial["exports"]["exr"]["valid"]  # written before the failure and still recorded
    assert "tiff16" not in partial["exports"] and "source_consistency" not in partial
    assert Path(partial["exports"]["exr"]["path"]).exists()
    assert "bicubic" not in manifest["baselines"]
    assert "disk full" in (failure.value.manifest_path.parent / "report.md").read_text()


def test_failure_before_any_output_writes_nothing(tmp_path, project):
    with pytest.raises(pipeline.StageError) as failure:
        pipeline.run(tmp_path / "missing.png", small_config(), project, log=lambda _: None)
    assert failure.value.stage == "ingest_and_validate"
    assert failure.value.manifest_path is None


def test_identical_runs_produce_identical_pixels(source, project):
    first = pipeline.run(source, small_config(baselines=("lanczos",)), project, run_id="a", log=lambda _: None)
    second = pipeline.run(source, small_config(baselines=("lanczos",)), project, run_id="b", log=lambda _: None)
    for key in ("working_pixel_sha256",):
        assert first.manifest["baselines"]["lanczos"][key] == second.manifest["baselines"]["lanczos"][key]
    a, b = (r.manifest["baselines"]["lanczos"]["exports"] for r in (first, second))
    assert a["tiff16"]["pixel_sha256"] == b["tiff16"]["pixel_sha256"]
    assert a["exr"]["pixel_sha256"] == b["exr"]["pixel_sha256"]


def test_test_image_is_deterministic():
    a = make_test_image(200, 120, seed=7)
    assert a.shape == (120, 200, 3) and a.dtype == np.uint8
    assert np.array_equal(a, make_test_image(200, 120, seed=7))
    assert not np.array_equal(a, make_test_image(200, 120, seed=8))
    with pytest.raises(ValueError):
        make_test_image(32, 32)


def test_yaml_config_and_category_route_the_run(tmp_path, project):
    path = tmp_path / "c.yaml"
    path.write_text("quick_target_megapixels: 0.1\ncrop_size: 32\ncrop_count: 1\nreview_max_edge: 256\n")
    config = load_config(path, runtime_preset="quick", input_category="night")
    image = write_test_image(tmp_path / "night.png", 160, 90)
    result = pipeline.run(image, config, project, log=lambda _: None)
    assert result.masters_dir.parent == project / "outputs" / "night"
    assert list(result.masters_dir.glob("*_srgb16.tif"))
    assert (project / "inputs" / "night").is_dir()


def test_storage_check_measures_the_disk_of_a_new_output_folder(source, project, tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir()
    real_disk_usage = shutil.disk_usage

    def disk_usage(path):
        usage = real_disk_usage(path)
        return usage._replace(free=1000) if Path(path) == drive.resolve() else usage

    monkeypatch.setattr(environment.shutil, "disk_usage", disk_usage)
    config = small_config(baselines=("bicubic",), output_root=str(drive / "new" / "masters"))
    manifest = pipeline.run(source, config, project, log=lambda _: None).manifest
    assert manifest["runtime"]["disk_free_gb"] == 0.0
    assert "below the uncompressed upper bound" in manifest["storage"]["warning"]


def test_installed_package_records_the_project_commit(source, project, monkeypatch):
    def git(*args):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init")
    head = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    monkeypatch.setattr(
        pipeline, "__file__", str(project.parent / "site-packages" / "synthetic_raw_lab" / "pipeline.py")
    )
    config = small_config(baselines=("bicubic",))
    manifest = pipeline.run(source, config, project, log=lambda _: None).manifest
    assert manifest["package"]["git"] == {"commit": head.strip(), "dirty": False}


def test_storage_counts_the_hard_linked_final_image_once(source, project):
    manifest = pipeline.run(source, small_config(baselines=("bicubic",)), project, log=lambda _: None).manifest
    final = manifest["final_output"]["files"]["tiff16"]
    if final["how"] != "hard link":
        pytest.skip("this file system does not support hard links")
    masters = Path(final["path"]).parent.parent
    files = {f.stat().st_ino: f.stat().st_size for f in masters.rglob("*") if f.is_file()}
    reports = Path(manifest["crop_panels"][0]["path"]).parent.parent
    files.update({f.stat().st_ino: f.stat().st_size for f in reports.rglob("*") if f.is_file()})
    assert manifest["storage_used_bytes"] <= sum(files.values())


def test_a_detected_face_frames_the_first_crops(source, project, monkeypatch):
    from synthetic_raw_lab import subjects

    face = subjects.Face(100, 20, 120, 150)
    monkeypatch.setattr(
        subjects, "analyse", lambda *args: subjects.SubjectAnalysis(320, 180, (face,), "test", "one test face")
    )
    manifest = pipeline.run(source, small_config(crop_count=3), project, log=lambda _: None).manifest
    assert [c["name"] for c in manifest["crops"]] == ["face_eyes", "face_nose_mouth", "detail_1"]
    assert manifest["subject"]["primary_face"] == face.to_dict() and manifest["subject"]["high_information_face"]
    stage = next(s for s in manifest["stages"] if s["name"] == "subject_analysis")
    assert stage["classification"] == pipeline.MEASUREMENT and "one test face" in stage["notes"]


def test_manual_crops_keep_their_place_and_the_subject_is_still_analysed(source, project):
    config = small_config(crops=((0, 0, 32, 32),))
    manifest = pipeline.run(source, config, project, log=lambda _: None).manifest
    assert [c["name"] for c in manifest["crops"]] == ["manual_1"]
    assert "subject" in manifest and manifest["final_output"]["method"] == "lanczos"


def test_final_output_override_picks_that_output(source, project):
    manifest = pipeline.run(source, small_config(final_output="bicubic"), project, log=lambda _: None).manifest
    final = manifest["final_output"]
    assert (final["method"], final["rule"]) == ("bicubic", "override")
    assert Path(final["files"]["tiff16"]["path"]).name.startswith("final_scene_bicubic_")


def _export_paths(manifest):
    paths = [item["path"] for item in manifest.get("intermediates", [])]
    for summary in manifest["baselines"].values():
        paths += [export["path"] for export in summary["exports"].values()]
    paths += [item["path"] for item in manifest["final_output"]["files"].values()]
    return paths
