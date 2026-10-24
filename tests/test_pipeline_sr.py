"""Phase 2 in the pipeline: preset wiring, exports, provenance and report, with a stand-in network.

``sr_models.load_model`` is replaced by a small local numpy model, so these tests need neither PyTorch
nor weights. The real network runs in ``test_sr_models.py`` when its weights are cached.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from synthetic_raw_lab import pipeline, sr_models
from synthetic_raw_lab import super_resolution as sr
from synthetic_raw_lab.config import ExperimentConfig
from synthetic_raw_lab.test_images import write_test_image

ALLOWED_CLASSES = {
    pipeline.PRESERVATION,
    pipeline.INTERPOLATION,
    pipeline.INFERENCE,
    pipeline.PACKAGING,
    pipeline.MEASUREMENT,
    pipeline.SETUP,
}


class FakeNetwork:
    """Stands in for sr_models.TorchModel: x4 cubic upscale plus a weights-dependent sharpening."""

    scale = 4
    precision = "float32"
    architecture = "SwinIR"

    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = device
        self.weights = sr_models.WEIGHTS[name]
        self.amount = 0.3 if self.weights.role == "fidelity" else 0.9

    def __call__(self, encoded):
        h, w = encoded.shape[:2]
        up = cv2.resize(encoded, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        soft = cv2.GaussianBlur(up, (0, 0), 2.0)
        return np.clip(up + self.amount * (up - soft), 0, 1)

    def fallback(self):
        return None

    def release_cache(self):
        pass

    def describe(self):
        return {
            "weights": self.weights.to_dict(),
            "architecture": self.architecture,
            "scale": 4,
            "device": {"device": self.device, "name": "test CPU"},
            "precision": self.precision,
            "peak_device_memory_mb": None,
        }


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def load_model(name, weights_dir, **kwargs):
        calls.append({"name": name, "weights_dir": Path(weights_dir), **kwargs})
        return FakeNetwork(name)

    monkeypatch.setattr(sr_models, "load_model", load_model)
    monkeypatch.setattr(sr_models, "available", lambda: True)
    return calls


@pytest.fixture
def source(tmp_path):
    return write_test_image(tmp_path / "scene.png", 320, 180)


def sr_config(**overrides) -> ExperimentConfig:
    values = {
        "runtime_preset": "quick",
        "quick_target_megapixels": 0.4,
        "crop_size": 48,
        "crop_count": 2,
        "review_max_edge": 256,
        "baselines": ("lanczos",),
        "enable_super_resolution": True,
        "sr_tile_size": 128,
        "sr_tile_overlap": 64,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def assert_run_labels(manifest, *, sr_masters):
    """Run-level provenance follows the masters actually written, not the SR setting."""
    if sr_masters:
        expected = (pipeline.PHASE_SR, pipeline.RUN_NOTICE_SR, pipeline.SEEDS_SR)
    else:
        expected = (pipeline.PHASE_01, pipeline.NOTICE, pipeline.SEEDS_UNUSED)
    assert (manifest["phase"], manifest["notice"], manifest["random_seeds"]["used_by"]) == expected


def test_sr_modes_are_exported_measured_and_reported(source, project, loads):
    config = sr_config(sr_modes=("conservative", "balanced", "aggressive"))
    result = pipeline.run(source, config, project, log=lambda _: None)
    manifest = result.manifest

    assert manifest["status"] == "complete"
    assert_run_labels(manifest, sr_masters=True)
    modes = manifest["super_resolution"]["modes"]
    assert list(modes) == ["conservative", "balanced", "aggressive"]
    # Preset wiring: the fidelity network for Mode A, the perceptual one for B and C, loaded once each.
    assert [modes[m]["weights"] for m in modes] == [
        "swinir_m_x4_classical",
        "swinir_m_x4_realsr_gan",
        "swinir_m_x4_realsr_gan",
    ]
    assert [c["name"] for c in loads] == ["swinir_m_x4_classical", "swinir_m_x4_realsr_gan"]
    assert loads[0]["weights_dir"] == project / "weights"
    assert {c["device"] for c in loads} == {"auto"} and {c["seed"] for c in loads} == {42}
    assert [modes[m]["consistency_strength"] for m in modes] == [1.0, 1.0, 0.0]
    assert "source_projection" in modes["balanced"] and "source_projection" not in modes["aggressive"]

    for mode, summary in modes.items():
        assert summary["classification"] == pipeline.INFERENCE
        assert summary["preset"] == sr.PRESETS[mode].to_dict()
        assert summary["exports"]["tiff16"]["valid"] and summary["exports"]["exr"]["valid"]
        assert summary["tiling"]["tile_size"] == 128 and summary["tiling"]["tiles"] > 1
        assert summary["seams"]["passed"]
        assert summary["source_consistency"]["psnr_db"] > 25
        assert summary["interpolation_difference"]["mean"] > 0
        sidecar = json.loads(Path(summary["exports"]["tiff16"]["path"] + ".json").read_text())
        assert sidecar["provenance"]["notice"] == pipeline.SR_NOTICE
        assert sidecar["provenance"]["model_weights_sha256"] == sr_models.WEIGHTS[summary["weights"]].sha256
        assert pipeline.INFERENCE in sidecar["provenance"]["stage_classification"]
        assert (result.reports_dir / "figures" / f"sr_difference_{mode}.png").exists()
        assert (result.reports_dir / "figures" / f"sr_{mode}_preview.jpg").exists()

    projected = modes["conservative"]["source_projection"]
    assert projected["mean_abs_residual_linear_after"] < projected["mean_abs_residual_linear_before"]
    assert [m["weights"]["name"] for m in manifest["models"]] == [modes[m]["weights"] for m in modes]
    assert {s["classification"] for s in manifest["stages"]} <= ALLOWED_CLASSES
    assert {"load_model_swinir_m_x4_classical", "super_resolution_balanced"} <= {s["name"] for s in manifest["stages"]}
    assert all(f"sr_{m}" in p["methods"] for p in manifest["crop_panels"] for m in modes)
    # TIFF + EXR per method, the intermediate, and the final image's TIFF + EXR
    assert len(manifest["output_checksums"]) == 2 * (1 + 3) + 1 + 2
    final = manifest["final_output"]
    assert final["method"] == "sr_balanced" and "general scene" in final["reason"]  # no face in the test chart
    assert final["files"]["tiff16"]["file_sha256"] == modes["balanced"]["exports"]["tiff16"]["file_sha256"]
    for mode in ("conservative", "balanced"):
        resample = modes[mode]["tiling"]["resample_range"]
        assert resample["min_linear_after"] >= 0 and resample["min_linear_before"] <= resample["min_linear_after"]
        assert modes[mode]["source_projection"]["min_linear_after"] >= 0
        assert min(modes[mode]["statistics"]["min"].values()) >= 0

    report = result.report_path.read_text()
    assert report.startswith("# Phase 0 to 2 report")
    assert "Pretrained super-resolution (PRD 10, experiment E02)" in report
    assert "| Metric | lanczos | sr_conservative | sr_balanced | sr_aggressive |" in report
    assert "partly true by construction" in report
    assert "synthesised" in report and "swinir_m_x4_classical" in report
    assert "Phase 2 exit criterion not assessed" in report
    assert "Linear min: resampled -> anti-ringing -> corrected" in report
    assert "Source PSNR (dB) (higher is better), best first: " in report and "sr_aggressive " in report
    assert "Final 50 MP image: sr_balanced" in report


def test_the_human_category_makes_the_conservative_mode_final(source, project, loads):
    config = sr_config(input_category="human")
    final = pipeline.run(source, config, project, log=lambda _: None).manifest["final_output"]
    assert final["method"] == "sr_conservative" and "human" in final["reason"]
    assert Path(final["files"]["exr"]["path"]).name.startswith("final_scene_sr_conservative_")


def test_a_skipped_sr_override_falls_back_to_a_baseline(source, project, loads):
    config = sr_config(quick_target_megapixels=0.04, pixel_count_tolerance=0.01, final_output="sr_conservative")
    manifest = pipeline.run(source, config, project, log=lambda _: None).manifest
    assert manifest["super_resolution"]["skipped"]
    final = manifest["final_output"]
    assert final["method"] == "lanczos" and "could not be used" in final["notes"][0]


def test_disabled_projection_and_strength_reach_the_run(source, project, loads):
    config = sr_config(sr_modes=("conservative",), preserve_source_structure=False, sr_strength=0.0)
    summary = pipeline.run(source, config, project, log=lambda _: None).manifest["super_resolution"]["modes"][
        "conservative"
    ]
    assert summary["consistency_strength"] == 0.0 and "source_projection" not in summary
    # strength 0 is pure Lanczos, so nothing departs from the interpolation
    assert summary["interpolation_difference"]["p99"] == 0


def test_backend_and_weights_folder_are_configurable(source, project, loads, tmp_path):
    config = sr_config(sr_modes=("balanced",), sr_backend="realesrgan", sr_weights_dir=str(tmp_path / "cache"))
    pipeline.run(source, config, project, log=lambda _: None)
    assert [(c["name"], c["weights_dir"]) for c in loads] == [("realesrgan_x4plus", tmp_path / "cache")]


def test_sr_is_skipped_when_the_target_is_not_an_enlargement(source, project, loads):
    result = pipeline.run(
        source, sr_config(quick_target_megapixels=0.04, pixel_count_tolerance=0.01), project, log=lambda _: None
    )
    manifest = result.manifest
    assert manifest["status"] == "complete"
    assert "not an enlargement" in manifest["super_resolution"]["skipped"]
    assert not loads and manifest["models"] == []
    assert_run_labels(manifest, sr_masters=False)
    report = result.report_path.read_text()
    assert pipeline.NOTICE in report and pipeline.RUN_NOTICE_SR not in report


def test_failure_in_sr_keeps_baselines_and_names_the_stage(source, project, loads, monkeypatch):
    def broken(*args, **kwargs):
        raise sr_models.DeviceOutOfMemory("cuda: simulated, no fallback device")

    monkeypatch.setattr(sr, "super_resolve", broken)
    with pytest.raises(pipeline.StageError) as failure:
        pipeline.run(source, sr_config(), project, log=lambda _: None)
    assert failure.value.stage == "super_resolution_conservative"
    manifest = json.loads(failure.value.manifest_path.read_text())
    assert manifest["baselines"]["lanczos"]["exports"]["tiff16"]["valid"]
    assert_run_labels(manifest, sr_masters=False)
    report = (failure.value.manifest_path.parent / "report.md").read_text()
    assert "super_resolution_conservative" in report and pipeline.RUN_NOTICE_SR not in report


def test_failure_after_an_sr_master_keeps_the_sr_labels(source, project, loads, monkeypatch):
    calls = []
    real = sr.super_resolve

    def second_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise sr_models.DeviceOutOfMemory("cuda: simulated, no fallback device")
        return real(*args, **kwargs)

    monkeypatch.setattr(sr, "super_resolve", second_fails)
    with pytest.raises(pipeline.StageError) as failure:
        pipeline.run(source, sr_config(sr_modes=("conservative", "balanced")), project, log=lambda _: None)
    assert failure.value.stage == "super_resolution_balanced"
    manifest = json.loads(failure.value.manifest_path.read_text())
    assert manifest["super_resolution"]["modes"]["conservative"]["exports"]["tiff16"]["valid"]
    assert_run_labels(manifest, sr_masters=True)


def test_sr_without_torch_reports_how_to_install(source, project, monkeypatch):
    monkeypatch.setattr(sr_models, "available", lambda: False)
    with pytest.raises(pipeline.StageError, match=r"\[sr\]") as failure:
        pipeline.run(source, sr_config(), project, log=lambda _: None)
    assert failure.value.stage == "super_resolution_setup"
    manifest = json.loads(failure.value.manifest_path.read_text())
    assert manifest["status"] == "failed in stage super_resolution_setup"
    assert_run_labels(manifest, sr_masters=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sr_modes": ["fast"]},
        {"sr_backend": "esrgan"},
        {"sr_tile_size": 32},
        {"sr_tile_size": 96, "sr_tile_overlap": 64},
        {"sr_tile_overlap": 32},
        {"sr_strength": 1.5},
        {"sr_device": "tpu"},
    ],
)
def test_invalid_sr_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        sr_config(**overrides)
