"""In-memory human reconstruction used by Notebook 01 run orchestration.

Candidates are retained for mandatory comparisons. Callers should release this
result after packaging figures and masters. Learned SR synthesizes detail;
source consistency is not identity proof. File ingestion, target planning and packaging belong to the run layer.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import baselines, crowd_processing, environment, face_preservation, quality_metrics, region_masks
from .human_config import HumanConfig
from .human_pipeline import (
    STAND_IN_DESCRIPTION,
    Phase2Upscaler,
    _validate_image,
    add_grain,
    adjust_tonal_contrast,
    balance_colour,
    blend_upscale,
    get_upscaler,
    sharpen_edges,
    smooth_skin,
)
from .pipeline import INFERENCE, INTERPOLATION, MEASUREMENT, SR_NOTICE, StageRecord, _stage


@dataclass(frozen=True)
class HumanReconstruction:
    candidates: dict[str, np.ndarray]
    final: np.ndarray
    detections: face_preservation.FaceDetections
    masks: region_masks.RegionMasks
    identity: dict
    failure_events: list[dict]
    crowd: dict
    stages: list[StageRecord]
    peak_rss_mb: float


def reconstruct_human_crowd(
    source: np.ndarray,
    width: int,
    height: int,
    config: HumanConfig,
    *,
    log: Callable[[str], None] = print,
    project_root: str | Path = ".",
) -> HumanReconstruction:
    """Connect configured protection stages, retaining source-scale annotations.

    Target dimensions must be at least source size, as required by identity and
    crowd measurements. The memory sampler always stops, including on failure;
    stage failures use the shared StageError convention.
    """
    _validate_image(source)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (width, height)):
        raise ValueError("target dimensions must be integers")
    if width < source.shape[1] or height < source.shape[0]:
        raise ValueError("target must be at least source size on both axes")
    stages = []
    learned = config.sr_backend != "lanczos_stand_in"
    sr_notice = SR_NOTICE if learned else STAND_IN_DESCRIPTION
    with environment.MemoryMonitor() as monitor:

        def stage(record):
            return _stage(record, stages, monitor, log)

        with stage(StageRecord("face_observations", "Preservation")) as record:
            detector = {
                "haar": face_preservation.HaarFaceDetector,
                "yunet": face_preservation.YuNetFaceDetector,
                "none": lambda: None,
            }[config.face_detector]()
            detections = face_preservation.observe_faces(
                source,
                face_preservation.ManualFaceDetector(
                    tuple(face_preservation.FaceBox(*box) for box in config.manual_face_boxes),
                    config.primary_subject_index,
                ),
                detector,
            )
            detections = face_preservation.select_primary_subject(detections, config.face_tiers)
            record.__dict__.update(detections.stage_record().__dict__)
        with stage(face_preservation.tier_stage_record(detections, config.face_tiers)):
            # Tier assignments are source-scale measurements, never output-scale claims.
            tiers = [face_preservation.face_tier(face, config.face_tiers) for face in detections.faces]
        with stage(StageRecord("region_masks", MEASUREMENT)) as record:
            masks = region_masks.estimate_masks(source, detections)
            record.__dict__.update(masks.stage_record().__dict__)
        candidates = {}
        with stage(
            StageRecord(
                "lanczos_baseline",
                INTERPOLATION,
                {"width": width, "height": height},
                "float32 linear",
                "float32 linear",
                notes=["Source-derived Lanczos interpolation; no recovered detail."],
            )
        ):
            candidates["lanczos"] = baselines.resample(source, width, height, "lanczos")
        upscaler = Phase2Upscaler(config, project_root, log=log) if learned else get_upscaler(config.sr_backend)
        with stage(
            StageRecord(
                "general_sr_strength",
                INFERENCE if learned else INTERPOLATION,
                {"strength": config.general_sr_strength, "modes": ["conservative", "balanced"]},
                "float32 linear",
                "float32 linear",
                notes=["Blend Lanczos baseline with selected upscaler. " + sr_notice],
            )
        ):
            try:
                for mode in ("conservative", "balanced"):
                    with stage(StageRecord(f"human_upscale_{mode}", INFERENCE if learned else INTERPOLATION)) as record:
                        result = upscaler(source, width, height, mode)
                        record.__dict__.update(result.stage_record().__dict__)
                    # Blend and release the raw SR buffer before generating the next mode.
                    candidates[mode] = blend_upscale(candidates["lanczos"], result.linear, config.general_sr_strength)
                    del result
            finally:
                if learned:
                    upscaler.close()
        with stage(StageRecord("region_detail", "Simulation")) as record:
            # Conservative is the default face-preserving starting point.
            enhanced = crowd_processing.enhance_regions(candidates["conservative"], masks, config.regions)
            record.__dict__.update(enhanced.stage_record().__dict__)
        with stage(
            StageRecord(
                "skin_texture_smoothing",
                "Simulation",
                {
                    "strength": config.skin_texture_smoothing,
                    "diameter_output_pixels": 5,
                    "sigma_colour_linear": 0.05,
                    "sigma_space_output_pixels": 2,
                    "max_channel_delta": 0.02,
                    "primary_face_cap": config.regions.primary_face,
                    "background_face_cap": config.regions.background_faces,
                },
                "float32 linear",
                "float32 linear",
                notes=[
                    "Skin-weighted bilateral smoothing, not recovered texture; zero is an exact no-op. "
                    "Pre-partition skin samples include facial skin; complete face boxes cap blending "
                    "by their configured face strength before identity recovery."
                ],
            )
        ):
            skin = masks.masks["skin"]
            if masks.skin_samples is not None:
                skin = np.maximum(skin, masks.skin_samples)
            skin = cv2.resize(skin, (width, height), interpolation=cv2.INTER_LINEAR)
            face_limit = np.ones((height, width), np.float32)
            for index, face in enumerate(detections.faces):
                x0 = int(np.floor(face.x * width / source.shape[1]))
                y0 = int(np.floor(face.y * height / source.shape[0]))
                x1 = int(np.ceil((face.x + face.width) * width / source.shape[1]))
                y1 = int(np.ceil((face.y + face.height) * height / source.shape[0]))
                cap = (
                    config.regions.primary_face
                    if index == detections.primary_index
                    else config.regions.background_faces
                )
                face_limit[y0:y1, x0:x1] = np.minimum(face_limit[y0:y1, x0:x1], cap)
            smoothed = smooth_skin(enhanced.linear, config.skin_texture_smoothing, skin, face_limit)
            del enhanced, skin, face_limit
        with stage(
            StageRecord(
                "edge_sharpening",
                "Simulation",
                {"strength": config.edge_sharpening, "sigma_output_pixels": 1.0, "max_detail": 0.02},
                "float32 linear",
                "float32 linear",
                notes=[
                    "Bounded unsharp mask; complete observed face boxes excluded. "
                    "Zero is an exact no-op; no recovered detail."
                ],
            )
        ):
            exclusion = np.zeros((height, width), dtype=bool)
            for face in detections.faces:
                x0 = int(np.floor(face.x * width / source.shape[1]))
                y0 = int(np.floor(face.y * height / source.shape[0]))
                x1 = int(np.ceil((face.x + face.width) * width / source.shape[1]))
                y1 = int(np.ceil((face.y + face.height) * height / source.shape[0]))
                exclusion[y0:y1, x0:x1] = True
            sharpened = sharpen_edges(smoothed, config.edge_sharpening, exclusion)
            del smoothed
        with stage(
            StageRecord(
                "tonal_contrast",
                "Simulation",
                {"strength": config.tonal_contrast, "pivot_linear": 0.18, "gain_bounds": [0.5, 2.0]},
                "float32 linear",
                "float32 linear",
                notes=[
                    "Classical log-luminance contrast; zero is an exact no-op. No recovered detail. "
                    "Applied before identity recovery, which can undo face changes."
                ],
            )
        ):
            toned = adjust_tonal_contrast(sharpened, config.tonal_contrast)
            del sharpened
        with stage(
            StageRecord(
                "colour_balance",
                "Simulation",
                {"linear_rgb_gains": list(config.colour_balance)},
                "float32 linear",
                "float32 linear",
                notes=[
                    "Classical linear RGB gains; unit gains are an exact no-op. No recovered detail. "
                    "Skin shift measures this operation before identity recovery; "
                    "final identity metrics remain separate."
                ],
            )
        ) as record:
            balanced = balance_colour(toned, config.colour_balance)
            # Measure at source scale to avoid another output-sized perceptual map.
            before = baselines.downsample_area(toned, source.shape[1], source.shape[0])
            after = baselines.downsample_area(balanced, source.shape[1], source.shape[0])
            weights = masks.skin_samples
            shift = None
            if weights is not None and weights.sum() > 0:
                _, delta = quality_metrics.source_consistency(before, after)
                shift = float(np.average(delta, weights=weights))
            record.parameters["skin_tone_shift_ciede2000"] = shift
            record.parameters["skin_shift_status"] = (
                "measured on estimated source-scale skin samples"
                if shift is not None
                else "unavailable: no estimated skin samples"
            )
            del toned, before, after
            if shift is not None:
                del delta
        with stage(
            StageRecord(
                "grain",
                "Simulation",
                {
                    "strength": config.grain,
                    "seed": config.random_seed,
                    "spatial_scale_output_pixels": 1,
                    "chromatic": False,
                    "max_amplitude_linear": 0.02,
                    "luminance_dependence": "sqrt(clipped linear luma)",
                },
                "float32 linear",
                "float32 linear",
                notes=[
                    "Random texture, never recovered detail; observed face boxes excluded. "
                    "Laplacian-luma variance measured before recovery; recovery may undo face changes."
                ],
            )
        ) as record:
            grained = add_grain(balanced, config.grain, exclusion, config.random_seed)
            record.parameters["high_frequency_energy_before"] = quality_metrics.high_frequency_energy(balanced)
            record.parameters["high_frequency_energy_after"] = quality_metrics.high_frequency_energy(grained)
            del balanced, exclusion
        with stage(StageRecord("identity_failure_recovery", "Simulation")) as record:
            recovery = face_preservation.preserve_face_identity(
                source,
                grained,
                detections,
                thresholds=config.identity,
                tier_config=config.face_tiers,
                skin_mask=masks.skin_samples,
                landmark_estimator=(
                    face_preservation.YuNetLandmarkEstimator() if config.face_detector == "yunet" else None
                ),
            )
            record.__dict__.update(recovery.stage.__dict__)
            del grained
        with stage(
            StageRecord(
                "crowd_consistency",
                MEASUREMENT,
                {"tiers": tiers},
                "float32 linear",
                "source-scale warning metrics",
                notes=[sr_notice, "Unavailable anatomy checks require human review."],
            )
        ):
            crowd = crowd_processing.evaluate_crowd_consistency(
                source,
                recovery.linear,
                detections,
                detector=detector,
                skin_mask=masks.skin_samples,
            )
    return HumanReconstruction(
        candidates,
        recovery.linear,
        detections,
        masks,
        recovery.evaluation,
        recovery.events,
        crowd,
        stages,
        monitor.peak_rss / environment.MB,
    )
