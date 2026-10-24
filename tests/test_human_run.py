"""Small file-to-report runs exercise actual master encoding and review packaging."""

import json
import threading
from dataclasses import astuple
from pathlib import Path

import pytest
from PIL import Image

from synthetic_raw_lab import exporters, test_images
from synthetic_raw_lab.human_config import HumanConfig
from synthetic_raw_lab.human_run import render_human_report, run_human_crowd
from synthetic_raw_lab.pipeline import StageError


def configuration():
    _, faces = test_images.make_crowd_test_image()
    return HumanConfig(
        face_detector="none",
        runtime_preset="quick",
        quick_target_megapixels=0.1536,
        manual_face_boxes=tuple(astuple(face) for face in faces),
        primary_subject_index=0,
    )


def source_file(tmp_path):
    pixels, _ = test_images.make_crowd_test_image()
    path = tmp_path / "crowd.png"
    Image.fromarray(pixels).save(path)
    return path


def test_human_run_packages_real_masters_and_tier_report(tmp_path):
    result = run_human_crowd(source_file(tmp_path), configuration(), tmp_path, log=lambda _: None)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest == result.manifest
    assert manifest["status"] == "complete"
    assert [row["face_count"] for row in manifest["identity"]["per_tier"]] == [1, 1, 1]
    assert all(item["valid"] for item in manifest["exports"].values())
    assert manifest["exports"]["tiff16"]["sample_type"] == "uint16"
    assert manifest["exports"]["exr"]["sample_type"] == "float16"
    for item in manifest["exports"].values():
        assert Path(item["path"]).with_suffix(Path(item["path"]).suffix + ".json").exists()
    assert len(manifest["comparisons"]["figures"]) == 9
    assert all(Path(item["path"]).exists() for item in manifest["comparisons"]["figures"].values())
    assert Path(manifest["review"]["path"]).suffix == ".jpg"
    report = result.report_path.read_text()
    for label in ("Per-tier identity metrics", "requires human review", "Failure-rule events", "Stage classification"):
        assert label in report
    assert all(stage["seconds"] >= 0 and stage["rss_peak_mb"] > 0 for stage in manifest["stages"])
    assert manifest["output_checksums"]
    assert not any(t.name == "memory-monitor" for t in threading.enumerate())


def test_export_failure_preserves_completed_master_and_report(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("test TIFF writer failure")

    monkeypatch.setattr(exporters, "write_tiff16", fail)
    with pytest.raises(StageError) as caught:
        run_human_crowd(source_file(tmp_path), configuration(), tmp_path, log=lambda _: None)
    assert caught.value.stage == "export_tiff16"
    manifest = json.loads(caught.value.manifest_path.read_text())
    assert manifest["status"] == "failed in stage export_tiff16"
    assert Path(manifest["exports"]["exr"]["path"]).exists()
    assert caught.value.manifest_path.with_name("report.md").exists()
    assert not any(t.name == "memory-monitor" for t in threading.enumerate())


def test_unset_primary_is_selected_and_reported_end_to_end(tmp_path):
    config = configuration().replace(primary_subject_index=None)
    result = run_human_crowd(source_file(tmp_path), config, tmp_path, log=lambda _: None)
    choice = result.manifest["primary_subject"]
    assert choice["index"] == 0
    assert choice["rule"] == "highest_source_tier_then_largest_area_then_first_observation"
    report = result.report_path.read_text()
    assert "Selected face: 0" in report
    assert choice["rule"] in report
    observations = next(s for s in result.manifest["stages"] if s["name"] == "face_observations")
    assert observations["parameters"]["primary_selection_rule"] == choice["rule"]


def test_report_is_readable_evidence_without_manifest_dump(tmp_path):
    result = run_human_crowd(source_file(tmp_path), configuration(), tmp_path, log=lambda _: None)
    report = result.report_path.read_text()
    assert "```json" not in report
    assert "| Check | Result | Threshold | Faces affected |" in report
    assert "Mean edge displacement" in report
    assert "Faces flagged" in report
    assert "Landmark status" in report
    assert "- extra eyes or teeth" in report
    assert "[manifest.json](manifest.json)" in report
    assert "Added noise or grain" in report

    # Exercise the same report path with precise evidence and a failed stage.
    manifest = result.manifest
    manifest["identity"]["per_tier"][0]["mean_ssim"] = 0.987654321
    stage = manifest["stages"][0]
    stage.update(notes=["first note", "second note"], error="test error")
    rendered = render_human_report(manifest)
    assert "| 0.9877 |" in rendered
    assert "first note; second note; test error" in rendered
    assert "[]None" not in rendered
    assert "['first note'" not in rendered
    assert manifest["identity"]["per_tier"][0]["mean_ssim"] == 0.987654321
    stage.update(notes=[], error=None)
    assert "[]None" not in render_human_report(manifest)


def test_unstaged_exception_writes_failed_diagnostics(tmp_path, monkeypatch):
    from synthetic_raw_lab import human_run

    def fail(*args, **kwargs):
        raise ValueError("test provenance failure")

    monkeypatch.setattr(human_run, "_provenance", fail)
    with pytest.raises(StageError) as caught:
        run_human_crowd(source_file(tmp_path), configuration(), tmp_path, log=lambda _: None)
    manifest = json.loads(caught.value.manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == "ValueError: test provenance failure"
    assert "failed" in caught.value.manifest_path.with_name("report.md").read_text()
    assert not any(t.name == "memory-monitor" for t in threading.enumerate())


def test_report_lists_failure_steps_and_flags_by_tier(tmp_path):
    config = configuration().replace(identity={"max_color_difference": 0.0, "max_edge_displacement": 0.0})
    result = run_human_crowd(source_file(tmp_path), config, tmp_path, log=lambda _: None)
    assert result.manifest["failure_events"]
    report = result.report_path.read_text()
    assert "| Face | Tier | Step | Metrics | Action |" in report
    for event in result.manifest["failure_events"]:
        for index, step in enumerate(event["steps"]):
            assert f"| {event['face_index']} | {event['tier']} | {index + 1} |" in report
            assert step["action"] in report
    assert "failed gates:" in report


def test_automatic_observations_without_manual_boxes_end_to_end(tmp_path, monkeypatch):
    from synthetic_raw_lab import face_preservation

    face = face_preservation.FaceBox(60, 40, 100, 100, 0.6)

    def fake_yunet(self, source):
        return face_preservation.FaceDetections(
            (face,), None, "yunet", "Inference", "Test inference: 1 detections; never identity proof."
        )

    monkeypatch.setattr(face_preservation.YuNetFaceDetector, "__call__", fake_yunet)
    config = configuration().replace(face_detector="yunet", manual_face_boxes=[], primary_subject_index=None)
    result = run_human_crowd(source_file(tmp_path), config, tmp_path, log=lambda _: None)
    assert result.manifest["primary_subject"]["index"] == 0
    assert len(result.manifest["identity"]["faces"]) == 1
    assert result.manifest["identity"]["faces"][0]["primary"]
    assert result.manifest["models"][0]["sha256"]
    assert "Test inference: 1 detections" in result.report_path.read_text()
    manual_only = run_human_crowd(
        source_file(tmp_path),
        config.replace(face_detector="none"),
        tmp_path,
        run_id="manual-only",
        log=lambda _: None,
    )
    assert manual_only.manifest["models"] == []
    assert manual_only.manifest["identity"]["faces"] == []


@pytest.mark.parametrize("data_attribute", [True, False])
def test_missing_haar_build_completes_manual_run(tmp_path, monkeypatch, data_attribute):
    from types import SimpleNamespace

    import cv2

    if data_attribute:
        monkeypatch.setattr(cv2, "data", SimpleNamespace(haarcascades=str(tmp_path)), raising=False)
    else:
        monkeypatch.delattr(cv2, "data", raising=False)
    config = configuration().replace(face_detector="haar")
    result = run_human_crowd(source_file(tmp_path), config, tmp_path, log=lambda _: None)
    assert result.manifest["status"] == "complete"
    model = result.manifest["models"][0]
    assert model["availability"] == "unavailable on this OpenCV build"
    assert model["sha256"] is None
    assert [row["face_count"] for row in result.manifest["identity"]["per_tier"]] == [1, 1, 1]
    assert "unavailable on this OpenCV build" in result.report_path.read_text()


def test_default_yunet_unavailable_completes_manual_run(tmp_path, monkeypatch):
    from synthetic_raw_lab import human_models

    def unavailable():
        raise RuntimeError("offline test artifact unavailable")

    monkeypatch.setattr(human_models, "ensure_yunet_model", unavailable)
    config = configuration().replace(face_detector=HumanConfig().face_detector)
    assert config.face_detector == "yunet"
    result = run_human_crowd(source_file(tmp_path), config, tmp_path, log=lambda _: None)
    assert result.manifest["status"] == "complete"
    assert result.manifest["models"] == [human_models.yunet_model_metadata()]
    assert len(result.manifest["identity"]["faces"]) == 3
    assert "YuNet unavailable: offline test artifact unavailable" in result.report_path.read_text()


def test_empty_tiers_report_missing_metrics_as_dash_end_to_end(tmp_path):
    config = configuration()
    config = config.replace(manual_face_boxes=config.manual_face_boxes[:1])
    result = run_human_crowd(source_file(tmp_path), config, tmp_path, log=lambda _: None)
    report = result.report_path.read_text()
    tiers = result.manifest["identity"]["per_tier"]
    assert [row["face_count"] for row in tiers] == [1, 0, 0]
    for tier in (2, 3):
        assert f"| {tier} | 0 | - | - | - | none | no faces |" in report
    # Display placeholders must not replace the unavailable evidence in the manifest.
    for row in tiers[1:]:
        assert row["mean_ssim"] is None
        assert row["mean_ciede2000"] is None
