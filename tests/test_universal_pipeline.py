"""The Phase 4 universal run: routing, SR ladder gates, re-runs after failed seam gates, final image delivery.

Every run here is tiny (a 192x128 or 384x256 source, SR replaced by a stand-in where it is exercised), so the
suite stays fast and never loads a network; the real SR path runs on Colab (see the notebook tests).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from synthetic_raw_lab import (
    baselines,
    final_output,
    ingestion,
    quality_metrics,
    sr_models,
    test_images,
    universal_report,
)
from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import night_processing as night
from synthetic_raw_lab import universal_config as uc
from synthetic_raw_lab import universal_pipeline as up
from synthetic_raw_lab.face_preservation import FaceBox, FaceDetections
from synthetic_raw_lab.pipeline import StageError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "universal_default.yaml"
SWINIR_CLASSICAL = sr_models.BACKENDS["swinir"]["fidelity"]


def config_for(project: Path, *, megapixels: float = 0.098304, sr: bool = False, **sections) -> uc.UniversalConfig:
    analysis = {"face_detection": False, **sections.pop("analysis", {})}
    return uc.load_universal_config(
        DEFAULT,
        target_megapixels=megapixels,
        crop_size=48,
        review_max_edge=256,
        output_root=str(project / "outputs"),
        reports_root=str(project / "reports"),
        inputs_root=str(project / "inputs"),
        super_resolution={"enabled": sr, **sections.pop("super_resolution", {})},
        analysis=analysis,
        human={"profile": str(ROOT / uc.HUMAN_PROFILE), "overrides": {}},
        night={"profile": str(ROOT / uc.NIGHT_PROFILE), "overrides": {}},
        **sections,
    )


@pytest.fixture
def day(project) -> Path:
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((128, 192, 3)).astype(np.float32), (0, 0), 2)
    pixels = np.clip(0.2 + 0.6 * (base - base.min()) / (base.max() - base.min()), 0, 1)
    path = project / "day.png"
    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(path)
    return path


@pytest.fixture
def dark(project) -> Path:
    return night.write_night_scene(project / "night.png", 192, 128)


def check_final(result: up.UniversalRunResult) -> dict:
    final = result.final
    assert set(final["files"]) == {"tiff16", "exr"}
    for record in final["files"].values():
        path = Path(record["path"])
        assert path.parent == result.masters_dir / "final" and path.name.startswith("final_")
        assert ingestion.sha256_file(path) == record["file_sha256"] and record["verified"]
        assert Path(record["path"] + ".json").exists()
    assert result.report_path.exists() and result.manifest_path.exists()
    assert json.loads(result.manifest_path.read_text())["status"] == "complete"
    return final


# --------------------------------------------------------------------------- measurements


def test_detail_energy_in_bands_matches_the_whole_image_measurement():
    rng = np.random.default_rng(0)
    image = rng.random((37, 29, 3)).astype(np.float32) * 1.2 - 0.1
    whole = quality_metrics.high_frequency_energy(cm.linear_to_srgb(np.clip(image, 0, 1)))
    assert up.detail_energy(image, rows=5) == pytest.approx(whole, rel=1e-9)


def test_clipped_fraction_counts_samples_at_the_16_bit_limits():
    image = np.full((4, 5, 3), 0.5, np.float32)
    image[0, 0] = 1.2
    image[1, 1, 0] = -0.1
    image[2, 2, 1] = 0.0
    assert up.clipped_fraction(image, rows=2) == pytest.approx(5 / 60)


def test_improvement_needs_every_gate():
    gates = uc.ImprovementGates()
    lanczos = {
        "detail_energy": 1.0,
        "clipped_fraction": 0.01,
        "source_consistency": {"ssim": 0.99, "delta_e_mean": 0.5},
    }
    good = {**lanczos, "detail_energy": 1.1}
    assert up.improvement(good, lanczos, gates)["passed"]
    assert up.improvement(good, lanczos, gates, {"passed": True, "max_score": 0.2})["passed"]
    seams = up.improvement(good, lanczos, gates, {"passed": False, "max_score": 3.0})
    assert not seams["passed"] and seams["failed"] == ["seam gate (Phase 2)"]
    flat = up.improvement({**lanczos, "detail_energy": 1.01}, lanczos, gates)
    assert flat["failed"] == ["detail gain over Lanczos"]
    drift = {**good, "source_consistency": {"ssim": 0.95, "delta_e_mean": 1.5}}
    assert up.improvement(drift, lanczos, gates)["failed"] == ["CIEDE2000 mean increase", "SSIM drop"]
    clipped = {**good, "clipped_fraction": 0.02}
    assert up.improvement(clipped, lanczos, gates)["failed"] == ["clipped sample increase"]


# --------------------------------------------------------------------------- face masks


def test_face_masks_grow_scale_and_clip_to_the_output():
    found = FaceDetections((FaceBox(10, 20, 40, 40), FaceBox(0, 0, 20, 20)), 0, "manual", "Preservation", "")
    masks = up.face_masks(found, (200, 100), (400, 200), uc.FaceProtection(margin=0.25, feather=0.15))
    assert masks[0]["box"] == [0, 20, 120, 140] and masks[0]["feather_px"] == 12
    assert masks[1]["box"] == [0, 0, 50, 50]


def test_mask_weight_ramps_inside_the_image_but_not_at_its_border():
    weight = up.mask_weight([0, 10, 40, 50], 8, (100, 100))
    assert weight.shape == (40, 40)
    assert weight[20, 0] == 1.0  # the left edge is the image border: no ramp
    assert weight[0, 20] < 0.05 and weight[20, -1] < 0.05  # ramps at the top and right edges
    assert weight[20, 20] == 1.0
    assert np.all(np.diff(weight[:8, 20]) > 0)


def test_face_policy_drops_only_detail_operations():
    params = uc.load_universal_config(DEFAULT).night_config(ROOT).night
    safe = up.face_safe(params)
    assert safe.texture.sharpening_strength == 0 and safe.noise.denoise_source == "off"
    assert safe.shadows == params.shadows and safe.highlights == params.highlights and safe.color == params.color


# --------------------------------------------------------------------------- routes


def test_general_scene_runs_the_common_pipeline_to_a_final_image(project, day):
    result = up.run_universal(day, config_for(project), project, log=lambda _: None)
    manifest = result.manifest
    assert manifest["scene"]["classification"]["category"] == "general"
    assert manifest["route"]["selection"]["input"] == "lanczos"
    assert manifest["route"]["selection"]["why"] == "super-resolution is disabled"
    final = check_final(result)
    assert final["method"] == "universal_lanczos" and final["category"] == "general"
    assert final["files"]["tiff16"]["width"] == 384 and final["files"]["tiff16"]["height"] == 256
    evaluation = manifest["evaluation"]
    assert evaluation["size"] == [384, 256]
    assert {c["check"] for c in evaluation["improvement"]["checks"]} == {
        "detail gain over Lanczos",
        "CIEDE2000 mean increase",
        "SSIM drop",
        "clipped sample increase",
    }
    assert "_lanczos" not in manifest["route"]
    assert manifest["figures"]["final"].endswith("final_preview.jpg")
    report = result.report_path.read_text()
    assert "## 1. Final full-size image" in report and "Standardised evaluation" in report
    assert "figures/scene.png" in report and "\u2014" not in report


def test_faces_are_protected_and_diagnosed_in_the_common_pipeline(project, day):
    config = config_for(project, analysis={"manual_face_boxes": [[70, 40, 40, 40]]})
    result = up.run_universal(day, config, project, log=lambda _: None)
    route = result.manifest["route"]
    assert result.manifest["scene"]["classification"]["category"] == "general"
    assert len(route["masks"]["faces"]) == 1 and route["masks"]["face_policy"] == up.FACE_SAFE_NOTE
    assert route["faces"]["faces"][0]["box"] == [70, 40, 40, 40]
    assert route["crop_panels"][0]["crop"] == "face"
    check_final(result)


def test_night_with_faces_uses_the_night_policy_outside_the_masks(project, dark):
    config = config_for(project, analysis={"manual_face_boxes": [[60, 30, 40, 40]]})
    result = up.run_universal(dark, config, project, log=lambda _: None)
    assert result.manifest["scene"]["classification"]["category"] == "night_with_faces"
    processing = result.manifest["route"]["processing"]
    assert processing["params"]["shadows"]["lift_strength"] == "low"  # the night profile, not the universal one
    assert check_final(result)["method"] == "universal_lanczos"


def test_night_scene_is_handed_to_the_night_pipeline(project, dark):
    result = up.run_universal(dark, config_for(project), project, log=lambda _: None)
    route = result.manifest["route"]
    assert result.manifest["scene"]["classification"]["category"] == "night"
    assert route["attempts"] == [route["attempts"][0]] and route["attempts"][0]["gate"] is None
    assert route["method"] == "night_lanczos"
    delegated = json.loads(Path(route["delegate"]["manifest"]).read_text())
    assert delegated["status"] == "complete" and delegated["night"]["input"]["slot"] == "lanczos"
    final = check_final(result)
    assert final["files"]["tiff16"]["file_sha256"] == delegated["final"]["exports"]["tiff16"]["file_sha256"]
    archive = up.archive_reports(result.manifest, result.reports_dir)
    names = zipfile.ZipFile(archive).namelist()
    assert any(name.startswith(f"{result.run_id}/") for name in names)
    assert any(name.startswith(f"{route['delegate']['run_id']}/") for name in names)


def test_human_scene_is_handed_to_the_human_pipeline(project):
    pixels, faces = test_images.make_crowd_test_image(384, 256)
    path = project / "crowd.png"
    Image.fromarray(pixels).save(path)
    boxes = [[f.x, f.y, f.width, f.height] for f in faces]
    config = config_for(project, megapixels=0.393216, analysis={"manual_face_boxes": boxes})
    result = up.run_universal(path, config, project, log=lambda _: None)
    route = result.manifest["route"]
    assert result.manifest["scene"]["classification"]["category"] == "human"
    assert route["method"] == "human_lanczos" and route["attempts"][0]["sr_backend"] == "lanczos_stand_in"
    final = check_final(result)
    assert final["files"]["tiff16"]["width"] == 768


# --------------------------------------------------------------------------- SR ladder and seam gates


def fake_sr(passes: dict, calls: list):
    """A Phase 2 stand-in: an unsharp-masked Lanczos image and a seam record that passes or fails per mode."""

    def run(mode, source_linear, target, config, weights_dir, loaded, manifest, figures_dir, stage, log):
        calls.append(mode)
        image = baselines.resample(source_linear, target.width, target.height, "lanczos")
        image += 0.6 * (image - cv2.GaussianBlur(image, (0, 0), 1.5))
        manifest["super_resolution"]["modes"][mode] = {
            "weights": SWINIR_CLASSICAL,
            "seams": {"passed": passes[mode], "max_score": 0.1 if passes[mode] else 4.0},
        }
        return image

    return run


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    monkeypatch.setattr(up, "sr_setup", lambda config, project_root: tmp_path)
    monkeypatch.setattr(up, "release_sr_models", lambda loaded, config, remaining: None)


def test_a_rung_that_fails_its_seam_gate_never_reaches_the_final_image(project, day, monkeypatch, no_network):
    calls = []
    monkeypatch.setattr(up, "run_sr_mode", fake_sr({"conservative": False}, calls))
    result = up.run_universal(day, config_for(project, sr=True), project, log=lambda _: None)
    selection = result.manifest["route"]["selection"]
    assert calls == ["conservative"]
    assert selection["input"] == "lanczos"
    assert "seam gate (Phase 2)" in selection["candidates"]["sr_conservative"]["verdict"]["failed"]
    assert check_final(result)["method"] == "universal_lanczos"


def test_the_most_conservative_passing_rung_is_kept(project, day, monkeypatch, no_network):
    calls = []
    monkeypatch.setattr(up, "run_sr_mode", fake_sr({"conservative": True, "balanced": True}, calls))
    config = config_for(project, sr=True, super_resolution={"modes": ["conservative", "balanced"]})
    result = up.run_universal(day, config, project, log=lambda _: None)
    selection = result.manifest["route"]["selection"]
    assert selection["candidates"]["sr_conservative"]["verdict"]["passed"], selection
    assert calls == ["conservative"]  # the ladder stops at the first rung that passes
    assert selection["input"] == "sr_conservative"
    assert result.manifest["notice"] == up.NOTICE_SR
    sidecar = json.loads(Path(result.final["files"]["tiff16"]["path"] + ".json").read_text())
    assert sidecar["provenance"]["model_weights_sha256"] == sr_models.WEIGHTS[SWINIR_CLASSICAL].sha256


def test_balanced_is_tried_only_after_conservative_fails(project, day, monkeypatch, no_network):
    calls = []
    monkeypatch.setattr(up, "run_sr_mode", fake_sr({"conservative": False, "balanced": True}, calls))
    config = config_for(project, sr=True, super_resolution={"modes": ["conservative", "balanced"]})
    result = up.run_universal(day, config, project, log=lambda _: None)
    assert calls == ["conservative", "balanced"]
    assert result.manifest["route"]["selection"]["input"] == "sr_balanced"


def test_night_run_with_a_failed_sr_gate_is_repeated_without_sr(project, dark, monkeypatch):
    real = up.run_night
    seen = []

    def run_night(source_path, config, project_root, **kwargs):
        sr = config.night.super_resolution
        seen.append(sr.enabled)
        result = real(
            source_path,
            night.NightConfig(config.experiment, config.night.replace(super_resolution={"enabled": False})),
            project_root,
            **kwargs,
        )
        if sr.enabled:  # pretend the conservative SR input ran and failed its seam gate
            result.manifest["night"]["input"]["slot"] = f"{sr.mode}_sr"
            result.manifest["super_resolution"] = {"modes": {sr.mode: {"seams": {"passed": False, "max_score": 5.0}}}}
        return result

    monkeypatch.setattr(up, "run_night", run_night)
    result = up.run_universal(dark, config_for(project, sr=True), project, log=lambda _: None)
    attempts = result.manifest["route"]["attempts"]
    assert seen == [True, False]
    assert attempts[0]["gate"] == {
        "mode": "conservative",
        "passed": False,
        "seams": {"passed": False, "max_score": 5.0},
    }
    assert attempts[1]["gate"] is None and attempts[1]["run_id"].endswith("_night_lanczos")
    assert result.manifest["route"]["delegate"]["run_id"] == attempts[1]["run_id"]
    conclusions = universal_report.universal_conclusions(result.manifest)
    assert any("failed its seam gate" in line for line in conclusions)
    check_final(result)


def test_human_run_with_a_failed_sr_gate_is_repeated_with_the_stand_in(project, monkeypatch):
    pixels, faces = test_images.make_crowd_test_image(384, 256)
    path = project / "crowd.png"
    Image.fromarray(pixels).save(path)
    real = up.run_human_crowd
    backends = []

    def run_human_crowd(source_path, config, project_root, **kwargs):
        backends.append(config.sr_backend)
        result = real(source_path, config.replace(sr_backend="lanczos_stand_in"), project_root, **kwargs)
        if config.sr_backend != "lanczos_stand_in":
            result.manifest["super_resolution"]["modes"]["conservative"]["seams"] = {"passed": False, "max_score": 3.0}
        return result

    monkeypatch.setattr(up, "run_human_crowd", run_human_crowd)
    boxes = [[f.x, f.y, f.width, f.height] for f in faces]
    config = config_for(project, megapixels=0.393216, sr=True, analysis={"manual_face_boxes": boxes})
    result = up.run_universal(path, config, project, log=lambda _: None)
    assert backends == ["swinir", "lanczos_stand_in"]
    assert result.manifest["route"]["method"] == "human_lanczos"
    check_final(result)


# --------------------------------------------------------------------------- failure and delivery


def test_a_failing_stage_keeps_a_manifest_and_report(project, day, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("processing exploded")

    monkeypatch.setattr(night, "process_night", broken)
    with pytest.raises(StageError) as caught:
        up.run_universal(day, config_for(project), project, log=lambda _: None)
    assert caught.value.stage == "universal_processing"
    manifest = json.loads(caught.value.manifest_path.read_text())
    assert manifest["status"] == "failed in stage universal_processing"
    report = (caught.value.manifest_path.parent / "report.md").read_text()
    assert "processing exploded" in report and "No final image" in report


def test_the_final_image_is_published_and_delivered_by_the_shared_step(project, day, tmp_path):
    result = up.run_universal(day, config_for(project), project, log=lambda _: None)
    final = result.final
    assert final["rule"] == "universal" and final["preview"] == "figures/final_preview.jpg"
    naive = sum(f.stat().st_size for d in (result.masters_dir, result.reports_dir) for f in d.rglob("*") if f.is_file())
    assert all(f["how"] == "hard link" for f in final["files"].values())
    final_bytes = sum(f["size_bytes"] for f in final["files"].values())
    assert result.manifest["storage_used_bytes"] <= naive - final_bytes  # hard-linked final files counted once
    messages = []
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    done = final_output.deliver(
        result.manifest, drive_dir=drive_root / "final", drive_root=drive_root, log=messages.append
    )
    copied = Path(done["copied_to"])
    assert copied == drive_root / "final" / result.run_id
    assert sorted(p.name for p in copied.iterdir()) == sorted(
        p.name for p in (result.masters_dir / final_output.FINAL_DIR).iterdir()
    )
    assert done["downloaded"] is None and any(m.startswith("not in Colab") for m in messages)
