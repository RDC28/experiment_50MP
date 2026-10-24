"""The Phase 6 ablation engine (E02, E07, E09, PRD 31 matrix) on tiny inputs; SR is a stand-in, never a network."""

from __future__ import annotations

import json
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from helpers import evaluation_config, fake_sr
from synthetic_raw_lab import ablation, baselines, test_images
from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import night_processing as night
from synthetic_raw_lab.face_preservation import FaceBox, FaceDetections

ROOT = Path(__file__).resolve().parents[1]


def night_linear(width=192, height=128) -> np.ndarray:
    return cm.srgb_to_linear(night.make_night_scene(width, height).astype(np.float32) / 255).astype(np.float32)


def no_faces() -> FaceDetections:
    return FaceDetections((), None, "none", "Measurement", "no detector")


def make_scene(project, kind="night", reference=None, **sections):
    config = evaluation_config(project, ROOT, **sections)
    source = night_linear()
    return ablation.Scene(
        kind,
        source,
        (288, 192),
        no_faces(),
        ablation.scene_params(kind, config, ROOT),
        config.universal.human_config(ROOT),
        config.universal.face_protection,
        reference=reference,
    ), config


def test_neutral_layer_5b_pass_is_an_identity():
    source = night_linear()
    upscaled = baselines.resample(source, 288, 192, "lanczos")
    result = night.process_night(upscaled.copy(), night.segment_regions(source), source, ablation.neutral_params())
    assert np.abs(result.image - upscaled).max() < 1e-5


def test_every_full_minus_configuration_differs_from_full_in_exactly_its_stage():
    full = ablation.CONFIGURATIONS["full"]
    for configuration in ablation.CONFIGURATIONS.values():
        if not configuration.removes:
            continue
        changed = [
            name
            for name in ("upscale", "specialized", "face_handling", "grain", "optics", "sensor")
            if getattr(configuration, name) != getattr(full, name)
        ]
        assert len(changed) == 1, configuration.name
    assert ablation.CONFIGURATIONS["full_no_face_handling"].needs_faces
    assert {c.upscale for c in ablation.CONFIGURATIONS.values()} <= set(ablation.UPSCALES)


def test_engine_parameters_switch_one_stage_at_a_time(tmp_path):
    scene, config = make_scene(tmp_path)
    engine = ablation.Engine(scene, config, sr_runner=fake_sr)
    assert engine.engine_params(ablation.CONFIGURATIONS["sr_conservative"]) is None
    grain = engine.engine_params(ablation.CONFIGURATIONS["grain"])
    assert grain.noise.grain_amount == config.ablation.grain_amount and grain.optics.bloom_amount == 0
    assert grain.texture.sharpening_strength == 0 and not grain.failure_rule.enabled
    optics = engine.engine_params(ablation.CONFIGURATIONS["optics"])
    assert optics.optics.bloom_amount == config.ablation.bloom_amount and optics.noise.grain_amount == 0
    tone = engine.engine_params(ablation.CONFIGURATIONS["tone"])
    assert tone.texture.sharpening_strength == 0 and tone.shadows == scene.params.shadows
    specialised = engine.engine_params(ablation.CONFIGURATIONS["specialized"])
    assert specialised.texture == scene.params.texture and specialised.noise.grain_amount == 0


def test_upscales_run_once_and_are_shared(tmp_path):
    calls = []

    def counting(mode, linear, width, height):
        calls.append(mode)
        return fake_sr(mode, linear, width, height)

    scene, config = make_scene(tmp_path)
    engine = ablation.Engine(scene, config, sr_runner=counting)
    records, _ = ablation.run_configurations(
        engine, ["sr_conservative", "tone", "full", "full_no_grain"], [], log=print
    )
    assert calls == ["conservative"]
    assert all("measurements" in r for r in records)


def test_sr_configurations_are_skipped_with_the_reason_when_sr_is_unavailable(tmp_path):
    scene, config = make_scene(tmp_path)
    engine = ablation.Engine(scene, config, sr_runner=None, sr_unavailable="super-resolution unavailable: no torch")
    records, _ = ablation.run_configurations(engine, ["lanczos", "sr_conservative", "full_no_sr"], [], log=print)
    by = {r["name"]: r for r in records}
    assert by["sr_conservative"]["skipped"] == "super-resolution unavailable: no torch"
    assert "measurements" in by["lanczos"] and "measurements" in by["full_no_sr"]
    assert by["full_no_sr"]["vs_lanczos"]["source_psnr_db"] != 0


def test_one_failing_configuration_keeps_the_others(tmp_path):
    def broken(mode, linear, width, height):
        if mode == "balanced":
            raise RuntimeError("CUDA out of memory")
        return fake_sr(mode, linear, width, height)

    scene, config = make_scene(tmp_path)
    engine = ablation.Engine(scene, config, sr_runner=broken)
    records, _ = ablation.run_configurations(engine, ["lanczos", "sr_balanced", "sr_conservative"], [], log=print)
    by = {r["name"]: r for r in records}
    assert "CUDA out of memory" in by["sr_balanced"]["error"]
    assert "measurements" in by["sr_conservative"]
    attempts = []

    def counting(mode, linear, width, height):
        attempts.append(mode)
        return broken(mode, linear, width, height)

    engine = ablation.Engine(scene, config, sr_runner=counting)
    records, _ = ablation.run_configurations(engine, ["sr_balanced", "full_balanced"], [], log=print)
    assert attempts == ["balanced"]  # a failed upscale is not run again
    assert "failed earlier in this run" in records[1]["error"]


def test_reference_metrics_score_truth_perfectly_and_interpolation_near_one():
    rng = np.random.default_rng(1)
    truth = np.clip(baselines.resample(night_linear(96, 64), 288, 192, "lanczos"), 0, 1)
    truth = np.clip(truth + 0.05 * rng.standard_normal(truth.shape).astype(np.float32), 0, 1).astype(np.float32)
    source = baselines.downsample_area(truth, 144, 96)
    perfect = ablation.reference_metrics(truth, truth, (144, 96))
    assert perfect["hf_nmse"] == pytest.approx(0, abs=1e-9) and perfect["hf_correlation"] == pytest.approx(1)
    lanczos = ablation.reference_metrics(baselines.resample(source, 288, 192, "lanczos"), truth, (144, 96))
    assert 0.8 < lanczos["hf_nmse"] < 1.05 and lanczos["hf_energy_ratio"] < 0.5
    assert lanczos["psnr_db"] < perfect["psnr_db"] and lanczos["lowband_psnr_db"] > lanczos["psnr_db"]
    noisy = baselines.resample(source, 288, 192, "lanczos") + 0.1 * rng.standard_normal(truth.shape).astype(np.float32)
    assert ablation.reference_metrics(noisy, truth, (144, 96))["hf_nmse"] > 1  # invented detail is worse than none
    faces = ablation.reference_metrics(truth, truth, (144, 96), [(10, 10, 60, 60), (0, 0, 4, 4)])["faces"]
    assert len(faces) == 1 and faces[0]["ssim"] == pytest.approx(1)


def test_reference_scene_adds_accuracy_to_every_record(tmp_path):
    source = night_linear()
    truth = baselines.resample(source, 288, 192, "bicubic")
    scene, config = make_scene(tmp_path, reference=truth)
    engine = ablation.Engine(scene, config, sr_runner=fake_sr)
    records, _ = ablation.run_configurations(engine, ["lanczos", "sr_conservative"], [], log=print)
    assert all("reference" in r for r in records)
    assert records[1]["vs_lanczos"]["reference_psnr_db"] == pytest.approx(
        records[1]["reference"]["psnr_db"] - records[0]["reference"]["psnr_db"]
    )


def test_crop_box_and_faces_in_crop():
    assert ablation.crop_box(1000, 500, 300, (20, 480)) == (0, 200, 300, 500)
    assert ablation.crop_box(1000, 500, 0, (0, 0)) == (0, 0, 1000, 500)
    assert ablation.crop_box(200, 100, 300, (100, 50)) == (0, 0, 200, 100)
    detections = FaceDetections(
        (FaceBox(10, 10, 20, 20), FaceBox(250, 210, 40, 40), FaceBox(290, 290, 30, 30)), 1, "manual", "M", "d"
    )
    inside = ablation.faces_in_crop(detections, (200, 200, 300, 300))
    assert [(f.x, f.y, f.width) for f in inside.faces] == [(50, 10, 40)] and inside.primary_index == 0


def write_crowd(tmp_path) -> tuple[Path, list]:
    pixels, faces = test_images.make_crowd_test_image()
    path = tmp_path / "crowd.png"
    Image.fromarray(pixels).save(path)
    return path, [list(astuple(f)) for f in faces]


def test_run_ablation_on_a_human_crop_writes_manifest_report_and_sheets(tmp_path):
    source, boxes = write_crowd(tmp_path)
    config = evaluation_config(
        tmp_path,
        ROOT,
        universal_overrides={"analysis": {"manual_face_boxes": boxes}},
        ablation={"crop_size": 160, "target_megapixels": 0.1536},
    )
    result = ablation.run_ablation(source, config, tmp_path, sr_runner=fake_sr, log=lambda _: None)
    manifest = result.manifest
    assert manifest["status"] == "complete" and manifest["schema"] == ablation.SCHEMA
    assert manifest["scene"]["kind"] == "human" and manifest["crop"]["faces_in_crop"] >= 1
    assert manifest["crop"]["scale"] == [1.25, 1.25] and manifest["crop"]["output_size"] == [200, 200]
    names = [r["name"] for r in manifest["configurations"]]
    assert names == list(ablation.DEFAULT_MATRIX)
    assert all("measurements" in r for r in manifest["configurations"])
    full = next(r for r in manifest["configurations"] if r["name"] == "full")
    assert [a["stage"] for a in full["applied"]] == [
        "sr_conservative",
        "human_region_detail",
        "layer_5b",
        "identity_recovery",
        "sensor_round_trip",
    ]
    assert any("face masks cover" in note for note in full["notes"])
    assert {c["stage"] for c in manifest["contributions"]} >= {"super-resolution", "the sensor round trip"}
    assert manifest["bit_depth"]["expansion"]["information_added"] is False
    figures = manifest["figures"]
    assert Path(figures["review_sheet"]).exists() and Path(figures["blind_sheet"]).exists()
    key = json.loads((result.reports_dir / "figures" / "blind_key.json").read_text())
    assert sorted(key.values()) == sorted(names)
    on_disk = json.loads(result.manifest_path.read_text())
    assert on_disk["run_id"] == result.run_id
    report = result.report_path.read_text()
    assert "## 3. Configurations" in report and "## 4. Stage contributions" in report and "—" not in report


def test_run_ablation_without_sr_skips_learned_configurations(tmp_path):
    path = night.write_night_scene(tmp_path / "night.png", 320, 180)
    config = evaluation_config(
        tmp_path,
        ROOT,
        universal_overrides={"super_resolution": {"enabled": False}},
        ablation={
            "crop_size": 128,
            "target_megapixels": 0.2304,
            "configurations": ["lanczos", "sr_conservative", "full_no_sr"],
        },
    )
    result = ablation.run_ablation(path, config, tmp_path, log=lambda _: None)
    by = {r["name"]: r for r in result.manifest["configurations"]}
    assert result.manifest["scene"]["kind"] == "night"
    assert "disabled" in by["sr_conservative"]["skipped"] and "measurements" in by["full_no_sr"]
    assert result.manifest["bit_depth"]["image"] == "lanczos"


def test_failed_run_names_the_stage_and_keeps_a_report(tmp_path, monkeypatch):
    path = night.write_night_scene(tmp_path / "night.png", 320, 180)
    config = evaluation_config(tmp_path, ROOT, ablation={"crop_size": 128, "target_megapixels": 0.2304})

    def boom(*args, **kwargs):
        raise RuntimeError("bad masks")

    monkeypatch.setattr(ablation, "Scene", boom)
    with pytest.raises(ablation.StageError, match="prepare_scene") as raised:
        ablation.run_ablation(path, config, tmp_path, sr_runner=fake_sr, log=lambda _: None)
    manifest = json.loads(raised.value.manifest_path.read_text())
    assert manifest["status"] == "failed in stage prepare_scene"
    assert (raised.value.manifest_path.parent / "report.md").exists()
