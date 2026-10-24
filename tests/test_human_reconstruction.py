"""Integration checks for the configured human reconstruction graph."""

import threading
from dataclasses import astuple

import numpy as np
import pytest

from synthetic_raw_lab import baselines, color_management, test_images
from synthetic_raw_lab.human_config import HumanConfig
from synthetic_raw_lab.human_reconstruction import reconstruct_human_crowd
from synthetic_raw_lab.pipeline import StageError


def test_synthetic_crowd_reconstruction_preserves_tiers_and_provenance():
    encoded, faces = test_images.make_crowd_test_image()
    source = color_management.srgb_to_linear(encoded.astype(np.float32) / 255).astype(np.float32)
    before = source.copy()
    config = HumanConfig(
        sr_backend="lanczos_stand_in",
        face_detector="haar",
        manual_face_boxes=tuple(astuple(f) for f in faces),
        primary_subject_index=0,
    )
    result = reconstruct_human_crowd(source, 480, 320, config, log=lambda _: None)
    np.testing.assert_array_equal(source, before)
    assert result.final.shape == (320, 480, 3)
    assert result.final.dtype == np.float32 and np.isfinite(result.final).all()
    expected = baselines.resample(source, 480, 320, "lanczos")
    for candidate in result.candidates.values():
        np.testing.assert_array_equal(candidate, expected)
    assert [r["face_count"] for r in result.identity["per_tier"]] == [1, 1, 1]
    assert result.detections.primary_index == 0
    assert result.crowd["requires_human_review"]
    assert result.crowd["new_face_like_detections"] == []
    assert len(result.stages) == 15
    assert all(s.seconds >= 0 and s.rss_peak_mb > 0 for s in result.stages)
    assert all(s.classification and s.notes for s in result.stages)
    assert result.peak_rss_mb > 0
    assert not any(t.name == "memory-monitor" for t in threading.enumerate())


@pytest.mark.parametrize("kind", ["haar", "yunet"])
def test_crowd_check_reuses_configured_detector_at_source_scale(monkeypatch, kind):
    from synthetic_raw_lab import face_preservation

    calls = []

    def detector(image):
        calls.append(image.shape)
        # Only the final output contains a new, unmatched face-like observation.
        faces = (face_preservation.FaceBox(16, 4, 8, 8),) if len(calls) == 3 else ()
        return face_preservation.FaceDetections(faces, None, "fake", "Inference", "Local test detector")

    monkeypatch.setattr(
        face_preservation, "HaarFaceDetector" if kind == "haar" else "YuNetFaceDetector", lambda: detector
    )
    source = np.full((16, 24, 3), 0.2, np.float32)
    result = reconstruct_human_crowd(
        source, 48, 32, HumanConfig(sr_backend="lanczos_stand_in", face_detector=kind), log=lambda _: None
    )
    assert calls == [source.shape] * 3
    assert result.crowd["new_face_like_detections"][0]["box"] == [16, 4, 8, 8]
    assert "no automatic detector" not in result.crowd["detection_note"]

    calls.clear()
    disabled = reconstruct_human_crowd(
        source, 48, 32, HumanConfig(sr_backend="lanczos_stand_in", face_detector="none"), log=lambda _: None
    )
    assert calls == []
    assert disabled.crowd["new_face_like_detections"] is None
    assert "no automatic detector configured" in disabled.crowd["detection_note"]


def test_empty_annotations_remain_explicit():
    source = np.full((16, 24, 3), 0.2, np.float32)
    result = reconstruct_human_crowd(
        source, 24, 16, HumanConfig(sr_backend="lanczos_stand_in", face_detector="haar"), log=lambda _: None
    )
    assert result.identity["faces"] == []
    assert all(row["face_count"] == 0 for row in result.identity["per_tier"])
    assert result.failure_events == []


def test_stage_failure_stops_memory_sampler():
    config = HumanConfig(sr_backend="lanczos_stand_in", face_detector="haar", manual_face_boxes=((0, 0, 99, 99),))
    with pytest.raises(StageError) as error:
        reconstruct_human_crowd(np.zeros((16, 24, 3), np.float32), 24, 16, config, log=lambda _: None)
    assert error.value.stage == "face_observations"
    assert not any(t.name == "memory-monitor" for t in threading.enumerate())


@pytest.mark.parametrize("width,height", [(12, 16), (24, 8), (True, 16), (24, 16.0)])
def test_invalid_target_fails_before_processing(width, height):
    with pytest.raises(ValueError):
        reconstruct_human_crowd(
            np.zeros((16, 24, 3), np.float32),
            width,
            height,
            HumanConfig(sr_backend="lanczos_stand_in", face_detector="haar"),
        )


def test_altered_face_triggers_configured_recovery(monkeypatch):
    from synthetic_raw_lab import crowd_processing
    from synthetic_raw_lab.face_preservation import IdentityThresholds

    source = np.random.default_rng(7).uniform(0.1, 0.6, (32, 48, 3)).astype(np.float32)
    original_enhance = crowd_processing.enhance_regions

    def altered(*args, **kwargs):
        result = original_enhance(*args, **kwargs)
        result.linear[8:24, 8:24] = 0.95
        return result

    monkeypatch.setattr(crowd_processing, "enhance_regions", altered)
    config = HumanConfig(
        sr_backend="lanczos_stand_in",
        face_detector="haar",
        manual_face_boxes=((8, 8, 16, 16),),
        primary_subject_index=0,
        identity=IdentityThresholds(max_color_difference=0.01, max_skin_shift=0.01),
    )
    result = reconstruct_human_crowd(source, 48, 32, config, log=lambda _: None)
    assert result.failure_events[0]["steps"][-1]["action"] == "original_derived_lanczos_fallback"
    np.testing.assert_array_equal(result.final[8:24, 8:24], source[8:24, 8:24])
    assert result.identity["faces"][0]["human_reviewer_judgment"] is None


def test_unset_primary_receives_primary_mask_and_leaves_background_faces():
    encoded, faces = test_images.make_crowd_test_image()
    source = color_management.srgb_to_linear(encoded.astype(np.float32) / 255).astype(np.float32)
    config = HumanConfig(
        sr_backend="lanczos_stand_in", face_detector="haar", manual_face_boxes=tuple(astuple(f) for f in faces)
    )
    result = reconstruct_human_crowd(source, 384, 256, config, log=lambda _: None)
    assert result.detections.primary_index == 0
    face = faces[0]
    center = (face.y + face.height // 2, face.x + face.width // 2)
    assert result.masks.masks["primary_face"][center] > 0.99
    assert result.masks.masks["background_faces"][center] == 0


def test_nonzero_sharpening_protects_complete_observed_box():
    from synthetic_raw_lab.crowd_processing import RegionDetailConfig

    source = np.random.default_rng(12).uniform(0.1, 0.6, (24, 32, 3)).astype(np.float32)
    config = HumanConfig(
        sr_backend="lanczos_stand_in",
        face_detector="none",
        manual_face_boxes=((8, 6, 12, 12),),
        regions=RegionDetailConfig(max_detail=0),
        edge_sharpening=1,
        general_sr_strength=0.5,
    )
    result = reconstruct_human_crowd(source, 32, 24, config, log=lambda _: None)
    np.testing.assert_array_equal(result.final[6:18, 8:20], source[6:18, 8:20])
    assert np.any(result.final[:4] != source[:4])
    records = {s.name: s for s in result.stages}
    assert records["edge_sharpening"].classification == "Simulation"
    assert records["general_sr_strength"].parameters["strength"] == 0.5


def test_colour_shift_unavailable_without_skin_and_recovery_checks_tone():
    from synthetic_raw_lab.face_preservation import IdentityThresholds

    source = np.full((16, 24, 3), 0.4, np.float32)
    config = HumanConfig(
        sr_backend="lanczos_stand_in",
        face_detector="none",
        manual_face_boxes=((4, 4, 8, 8),),
        tonal_contrast=0.5,
        colour_balance=(1.5, 0.5, 1),
        identity=IdentityThresholds(max_color_difference=0.001),
    )
    result = reconstruct_human_crowd(source, 24, 16, config, log=lambda _: None)
    stage = next(s for s in result.stages if s.name == "colour_balance")
    assert stage.parameters["skin_tone_shift_ciede2000"] is None
    assert "unavailable" in stage.parameters["skin_shift_status"]
    assert result.failure_events
    np.testing.assert_array_equal(result.final[4:12, 4:12], source[4:12, 4:12])
    assert np.any(result.final[0] != source[0])


def test_smoothing_uses_facial_skin_samples_with_complete_box_caps(monkeypatch):
    from synthetic_raw_lab import human_reconstruction, region_masks
    from synthetic_raw_lab.crowd_processing import RegionDetailConfig

    source = np.random.default_rng(18).uniform(0.3, 0.34, (24, 32, 3)).astype(np.float32)
    estimate = region_masks.estimate_masks

    def masks_with_skin(*args, **kwargs):
        result = estimate(*args, **kwargs)
        # Emulate chroma-positive facial skin removed by mask precedence.
        result.skin_samples[:] = 1
        return result

    monkeypatch.setattr(region_masks, "estimate_masks", masks_with_skin)
    real_smooth = human_reconstruction.smooth_skin
    captured = {}

    def capture(image, strength, skin, limits):
        captured.update(image=image.copy(), skin=skin.copy(), limits=limits.copy())
        captured["output"] = real_smooth(image, strength, skin, limits)
        return captured["output"]

    monkeypatch.setattr(human_reconstruction, "smooth_skin", capture)
    reconstruct_human_crowd(
        source,
        64,
        48,
        HumanConfig(
            sr_backend="lanczos_stand_in",
            face_detector="none",
            manual_face_boxes=((4, 4, 8, 8), (20, 4, 6, 6)),
            skin_texture_smoothing=1,
            regions=RegionDetailConfig(primary_face=0.1),
        ),
        log=lambda _: None,
    )
    assert np.all(captured["skin"] == 1)
    assert np.all(captured["limits"][8:24, 8:24] == np.float32(0.1))
    assert np.all(captured["limits"][8:20, 40:52] == 0)
    np.testing.assert_array_equal(captured["output"][8:20, 40:52], captured["image"][8:20, 40:52])
    assert np.max(np.abs(captured["output"][8:24, 8:24] - captured["image"][8:24, 8:24])) <= 0.0020001
    assert np.any(captured["output"] != captured["image"])


def test_yunet_reconstruction_reports_landmarks_without_network(monkeypatch):
    from synthetic_raw_lab import face_preservation as fp

    monkeypatch.setattr(fp, "YuNetFaceDetector", lambda: fp.ManualFaceDetector())
    crops = []

    def landmarks(crop):
        crops.append(crop.shape)
        return np.arange(10).reshape(5, 2)

    monkeypatch.setattr(fp, "YuNetLandmarkEstimator", lambda: landmarks)
    source = np.full((32, 40, 3), 0.2, np.float32)
    config = HumanConfig(sr_backend="lanczos_stand_in", face_detector="yunet", manual_face_boxes=((4, 4, 24, 24, 1.0),))
    result = reconstruct_human_crowd(source, 80, 64, config, log=lambda _: None)
    assert crops and all(shape == (24, 24, 3) for shape in crops)
    assert result.identity["faces"][0]["landmark_displacement_px"] == {"mean": 0.0, "max": 0.0}
    assert result.identity["per_tier"][2]["landmark_face_count"] == 1


def test_small_crowd_pixels_match_before_intermediate_release():
    """Stored float32 checksum from the pre-lifetime-change pipeline, OpenCV 5."""
    import hashlib

    encoded, faces = test_images.make_crowd_test_image()
    source = color_management.srgb_to_linear(encoded.astype(np.float32) / 255).astype(np.float32)
    config = HumanConfig(
        sr_backend="lanczos_stand_in",
        face_detector="none",
        manual_face_boxes=tuple(astuple(f) for f in faces),
        skin_texture_smoothing=0.2,
        edge_sharpening=0.3,
        tonal_contrast=0.1,
        colour_balance=(1.05, 0.95, 1),
        grain=0.1,
    )
    result = reconstruct_human_crowd(source, 480, 320, config, log=lambda _: None)
    assert hashlib.sha256(result.final.tobytes()).hexdigest() == (
        "93e0aba531727ebbdb4f497e8109f1e609a483d543fdac816e5eb331e6459388"
    )


@pytest.mark.parametrize("fail_balanced", [False, True])
def test_learned_reconstruction_uses_conservative_and_releases_models(monkeypatch, tmp_path, fail_balanced):
    import weakref
    from types import SimpleNamespace

    from synthetic_raw_lab import crowd_processing, sr_models
    from synthetic_raw_lab import super_resolution as sr

    loads, releases, selected = [], [], []
    raw_outputs = []

    class FakeModel:
        scale, alignment = 4, 1
        device, precision, architecture = "cpu", "float32", "fake"

        def __init__(self, name):
            self.name = name

        def describe(self):
            return {"architecture": self.architecture, "scale": self.scale}

        def release_cache(self):
            releases.append(self.name)

    def load(name, weights_dir, **kwargs):
        assert weights_dir == tmp_path
        loads.append(name)
        return FakeModel(name)

    def resolve(linear, width, height, model, **kwargs):
        assert all(reference() is None for reference in raw_outputs)
        conservative = model.name == sr_models.BACKENDS["swinir"]["fidelity"]
        if fail_balanced and not conservative:
            raise RuntimeError("fake balanced failure")
        output = np.full((height, width, 3), 0.21 if conservative else 0.7, np.float32)
        raw_outputs.append(weakref.ref(output))
        return output, SimpleNamespace(
            final_device="cpu",
            boundaries=[],
            to_dict=lambda: {"overlap_disagreement": {"blend_error_p99_abs_max": 0}},
        )

    enhance = crowd_processing.enhance_regions

    def capture(linear, *args):
        selected.append(linear.copy())
        return enhance(linear, *args)

    monkeypatch.setattr(sr_models, "available", lambda: True)
    monkeypatch.setattr(sr_models, "load_model", load)
    monkeypatch.setattr(sr, "super_resolve", resolve)
    monkeypatch.setattr(crowd_processing, "enhance_regions", capture)
    source = np.full((16, 24, 3), 0.2, np.float32)
    config = HumanConfig(
        face_detector="none",
        manual_face_boxes=((4, 4, 8, 8),),
        sr_weights_dir=str(tmp_path),
        sr_tile_size=256,
        preserve_source_structure=False,
    )
    if fail_balanced:
        with pytest.raises(StageError, match="fake balanced failure"):
            reconstruct_human_crowd(source, 48, 32, config, log=lambda _: None)
        assert not selected
    else:
        result = reconstruct_human_crowd(source, 48, 32, config, log=lambda _: None)
        np.testing.assert_allclose(selected[0], 0.21)
        np.testing.assert_allclose(result.candidates["balanced"], 0.7)
        np.testing.assert_allclose(result.final[8:24, 8:24], 0.21)
        records = [s for s in result.stages if s.name.startswith("human_upscale_")]
        assert len(records) == 2 and all(s.classification == "Inference" for s in records)
        assert all(s.parameters["seams"]["passed"] for s in records)
    assert loads == list(sr_models.BACKENDS["swinir"].values())
    assert sorted(releases) == sorted(loads)
    assert not any(t.name == "memory-monitor" for t in threading.enumerate())
