"""Phase 4 universal run (notebook 03): analyse, classify, route, process, evaluate, deliver one final image.

1. Ingest and validate the source once, then analyse the scene (:mod:`scene_analysis`): statistics, optional
   face detection, and the category with the cues and reasons behind it (PRD 20.2).
2. Hand the scene to the pipeline for its category, with the preset :mod:`universal_config` derives:

   - ``human``: the notebook 01 human and crowd pipeline (:func:`human_run.run_human_crowd`), whose final
     image is conservative SR with identity protection.
   - ``night``: the notebook 02 night pipeline (:func:`night_pipeline.run_night`) with the first rung of the
     SR ladder (conservative by default) as its input.
   - ``night_with_faces`` and ``general``: the universal common pipeline below.

   No route keeps an SR result that failed a seam gate: a specialised run whose SR input failed is re-run
   without SR (the Lanczos input), and the common pipeline only accepts a rung that passes the seam gate.
3. The common pipeline: the Lanczos baseline, then the SR ladder (Phase 2 backend, one network at a time),
   each rung measured against Lanczos and kept only when it passes the seam gate and the measurable-improvement
   gates (most conservative rung first, PRD 4.3); combined masks (the PRD 8.3 luminance regions plus feathered
   face masks); universal tone, colour and region-aware detail on the Layer 5B engine with the PRD 12.6 failure
   rule, and for a night scene with faces the night policies outside the face masks; face diagnostics;
   16-bit TIFF and EXR masters.
4. Standardised evaluation of the final master against the Lanczos baseline, the same for every route.
5. One final full-size image: the route's masters saved again under ``final/`` and verified by SHA-256 by the
   shared final-output step (``final_output.publish``), which the notebook then delivers with
   ``final_output.deliver``.

If a stage fails, outputs already written stay on disk and the manifest and report name the failing stage.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from . import (
    __version__,
    baselines,
    comparison,
    environment,
    exporters,
    face_preservation,
    final_output,
    ingestion,
    quality_metrics,
    report_builder,
    scene_analysis,
    sr_models,
    subjects,
)
from . import color_management as cm
from . import night_processing as night
from .human_run import run_human_crowd
from .night_pipeline import fallback_figure, region_figure, run_night
from .pipeline import (
    INTERPOLATION,
    MEASUREMENT,
    PACKAGING,
    PRESERVATION,
    SETUP,
    StageError,
    StageRecord,
    _exr_attributes,
    _stage,
    _stem,
    _without_histograms,
    make_run_id,
    release_sr_models,
    resolve_roots,
    run_sr_mode,
    sr_run_record,
    sr_setup,
)
from .resolution import compute_target_resolution
from .universal_config import UniversalConfig

SCHEMA = "synthetic_raw_lab.phase4_universal.manifest/1"
PHASE = "Phase 4 (universal notebook)"
NOTICE = (
    "Synthetic camera-grade master reconstructed from an AI-generated source by classical interpolation and "
    "universal tone, colour and detail processing. It is not a camera capture or camera RAW file and contains "
    "no recovered sensor measurements."
)
NOTICE_SR = (
    "Synthetic camera-grade master reconstructed from an AI-generated source by pretrained super-resolution and "
    "universal tone, colour and detail processing. Detail above the source sampling limit was synthesised by a "
    "learned model, not recovered; it is not a camera capture or camera RAW file."
)
DELEGATED = "Delegated (the specialised pipeline logs its own stages in its manifest)"
UNIVERSAL_PROCESSING = "Layer 5B engine with universal or night policies (per-step classes in the route record)"
COMMON = ("night_with_faces", "general")
# Linear values whose 16-bit sRGB code is 0 or 65535: what a clipped sample of the TIFF master is.
_LOW_CLIP = float(cm.srgb_to_linear(np.float32(0.5 / 65535)))
_HIGH_CLIP = 1.0 - 0.5 / 65535
_ROWS = 512
FACE_SAFE_NOTE = (
    "Inside the feathered face masks the processing runs without sharpening, atmospheric detail, denoising, "
    "grain or bloom (PRD 20.2: reduced enhancement in face regions); tone and colour are the same as outside, "
    "so the masks leave no tonal seam."
)


@dataclass
class UniversalRunResult:
    run_id: str
    manifest: dict
    manifest_path: Path
    report_path: Path
    masters_dir: Path
    reports_dir: Path

    @property
    def final(self) -> dict:
        """The final image record: ``files`` maps tiff16 and exr to their verified paths."""
        return self.manifest.get("final_output") or {}


# --------------------------------------------------------------------------- scene analysis


def analyse_scene(
    source_linear: np.ndarray, config: UniversalConfig, project_root: str | Path
) -> tuple[dict, face_preservation.FaceDetections]:
    """Statistics, faces and classification of a source (PRD 20.2); the same call the run makes."""
    analysis = config.analysis
    sr = config.super_resolution
    weights_dir = Path(sr.weights_dir) if sr.weights_dir else sr_models.default_weights_dir(Path(project_root))
    detections = scene_analysis.observe_faces(source_linear, analysis.detector, weights_dir, analysis.manual_face_boxes)
    statistics = scene_analysis.scene_statistics(source_linear, detections, analysis.thresholds)
    classification = scene_analysis.classify(
        statistics,
        analysis.thresholds,
        override=analysis.manual_override,
        automatic=analysis.automatic_scene_classification,
        night_detection=analysis.nighttime_detection,
        face_detection=analysis.face_detection,
    )
    return {"statistics": statistics, "classification": classification}, detections


def preset_record(config: UniversalConfig, classification: dict) -> dict:
    """The automatic preset selection (PRD 4.3): which pipeline and settings process this scene, and why."""
    category = classification["category"]
    ladder = list(config.sr_modes) if config.sr_enabled else []
    sr = (
        f"SR ladder {', '.join(ladder)} ({config.super_resolution.backend}); a rung is kept only if it passes "
        "the seam gate and the measurable-improvement gates, otherwise Lanczos"
        if ladder
        else "super-resolution disabled: Lanczos interpolation"
    )
    settings = {
        "human": (
            f"{config.human['profile']} with {len(config.human['overrides'])} override(s): conservative SR as the "
            "final image, identity protection, crowd consistency; re-run with the Lanczos stand-in if the "
            "conservative SR fails its seam gate"
        ),
        "night": (
            f"{config.night['profile']} with {len(config.night['overrides'])} override(s) and "
            f"{ladder[0] if ladder else 'no'} SR as the night input; re-run without SR if that input fails its "
            "seam gate"
        ),
        "night_with_faces": (
            f"{sr}; the night tone, colour and detail of {config.night['profile']} outside the face masks and "
            "the same without detail operations inside them"
        ),
        "general": f"{sr}; conservative universal tone, colour and region-aware detail; faces protected",
    }[category]
    why = {
        "manual": "set by manual_override",
        "low": "low-confidence or disabled classification: conservative general settings (PRD 20.2)",
        "high": f"classified as {category}",
    }[classification["confidence"]]
    return {"category": category, "pipeline": classification["pipeline"], "settings": settings, "why": why}


def scene_figure(
    source_linear: np.ndarray, detections: face_preservation.FaceDetections, path: Path, max_edge: int = 1600
) -> Path:
    """The source with the counted and uncounted face boxes, beside its PRD 8.3 luminance regions."""
    height, width = source_linear.shape[:2]
    scale = min(1.0, max_edge / 2 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reduced = baselines.downsample_area(source_linear, *size) if scale < 1 else source_linear
    picture = cm.srgb_uint16_to_uint8(cm.encode_srgb_uint16(reduced)[0]).copy()
    for face in detections.faces:
        x0, y0 = int(face.x * scale), int(face.y * scale)
        x1, y1 = int((face.x + face.width) * scale), int((face.y + face.height) * scale)
        cv2.rectangle(picture, (x0, y0), (max(x0 + 1, x1), max(y0 + 1, y1)), (255, 220, 0), 2)
    regions = night.segment_regions(np.ascontiguousarray(reduced, dtype=np.float32)).visualize(reduced)
    gutter = np.full((size[1], 8, 3), 24, np.uint8)
    body = np.concatenate([picture, gutter, regions], axis=1)
    header = np.full((28, body.shape[1], 3), 24, np.uint8)
    for x, text in ((6, f"source, {len(detections.faces)} face box(es)"), (size[0] + 14, "PRD 8.3 luminance regions")):
        cv2.putText(header, text, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return exporters.write_review_image(path, np.concatenate([header, body], axis=0))


# --------------------------------------------------------------------------- measurements


def detail_energy(linear: np.ndarray, rows: int = _ROWS) -> float:
    """Variance of the Laplacian of sRGB luma (``quality_metrics.high_frequency_energy``), in row bands.

    Higher means more fine detail, but ringing, noise and seams raise it too; it is a measurement of change,
    not of correctness, so the improvement gates pair it with source consistency and clipping.
    """
    height = linear.shape[0]
    total = total_sq = 0.0
    count = 0
    for start in range(0, height, rows):
        top, bottom = max(0, start - 1), min(height, start + rows + 1)
        y = quality_metrics.luma(cm.linear_to_srgb(np.clip(linear[top:bottom], 0, 1))).astype(np.float64)
        laplace = ndimage.laplace(y)[start - top : start - top + min(rows, height - start)]
        total += float(laplace.sum())
        total_sq += float(np.square(laplace).sum())
        count += laplace.size
    mean = total / count
    return max(total_sq / count - mean * mean, 0.0)


def clipped_fraction(linear: np.ndarray, rows: int = _ROWS) -> float:
    """Share of samples a 16-bit sRGB master stores at code 0 or 65535."""
    clipped = 0
    for start in range(0, linear.shape[0], rows):
        block = linear[start : start + rows]
        clipped += int(np.count_nonzero((block <= _LOW_CLIP) | (block >= _HIGH_CLIP)))
    return clipped / linear.size


def reduce_to_source(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-average reduction to the source size, one channel at a time (no second full-size buffer)."""
    reduced = np.empty((height, width, 3), dtype=np.float32)
    for channel in range(3):
        plane = np.clip(image[..., channel], 0, 1)[..., None]
        reduced[..., channel] = baselines.downsample_area(plane, width, height)[..., 0]
        del plane
    return reduced


def measure(image: np.ndarray, source_linear: np.ndarray) -> dict:
    """The standardised measurements of one full-size candidate (PRD 23): detail, consistency, clipping."""
    height, width = source_linear.shape[:2]
    consistency, _ = quality_metrics.source_consistency(source_linear, reduce_to_source(image, width, height))
    return {
        "detail_energy": detail_energy(image),
        "clipped_fraction": clipped_fraction(image),
        "source_consistency": {
            "psnr_db": consistency["psnr_db"],
            "ssim": consistency["ssim"],
            "delta_e_mean": consistency["color_difference_ciede2000"]["mean"],
            "delta_e_p95": consistency["color_difference_ciede2000"]["p95"],
            "edge_displacement": consistency["edge_displacement"],
        },
    }


def improvement(candidate: dict, lanczos: dict, gates, seams: dict | None = None) -> dict:
    """Whether ``candidate`` is a measurable improvement over Lanczos under the provisional gates.

    ``seams`` is a Phase 2 seam record for an SR candidate; ``None`` skips the seam check.
    """
    gain = candidate["detail_energy"] / lanczos["detail_energy"] - 1 if lanczos["detail_energy"] > 0 else 0.0
    cand, base = candidate["source_consistency"], lanczos["source_consistency"]
    checks = []
    if seams is not None:
        checks.append(
            {
                "check": "seam gate (Phase 2)",
                "value": seams.get("max_score"),
                "limit": "passed",
                "passed": bool(seams.get("passed")),
            }
        )
    checks += [
        {
            "check": "detail gain over Lanczos",
            "value": gain,
            "limit": f">= {gates.min_detail_gain:g}",
            "passed": gain >= gates.min_detail_gain,
        },
        {
            "check": "CIEDE2000 mean increase",
            "value": cand["delta_e_mean"] - base["delta_e_mean"],
            "limit": f"<= {gates.max_delta_e_increase:g}",
            "passed": cand["delta_e_mean"] - base["delta_e_mean"] <= gates.max_delta_e_increase,
        },
        {
            "check": "SSIM drop",
            "value": base["ssim"] - cand["ssim"],
            "limit": f"<= {gates.max_ssim_drop:g}",
            "passed": base["ssim"] - cand["ssim"] <= gates.max_ssim_drop,
        },
        {
            "check": "clipped sample increase",
            "value": candidate["clipped_fraction"] - lanczos["clipped_fraction"],
            "limit": f"<= {gates.max_clipping_increase:g}",
            "passed": candidate["clipped_fraction"] - lanczos["clipped_fraction"] <= gates.max_clipping_increase,
        },
    ]
    failed = [c["check"] for c in checks if not c["passed"]]
    return {"detail_gain": gain, "checks": checks, "passed": not failed, "failed": failed}


def standard_evaluation(
    source_linear: np.ndarray,
    final_tiff: Path,
    gates,
    lanczos: dict | None = None,
    *,
    preview: Path | None = None,
    max_edge: int = 2048,
) -> dict:
    """The final master as delivered (its 16-bit TIFF) against the Lanczos baseline, the same for every route.

    ``lanczos`` reuses a Lanczos measurement made at the same size; otherwise the baseline is rebuilt.
    ``preview`` writes the review JPEG of the final image from the same TIFF.
    """
    import tifffile

    encoded = tifffile.imread(final_tiff)
    if preview is not None:
        report_builder.write_preview(encoded, preview, max_edge)
    height, width = encoded.shape[:2]
    final = np.empty((height, width, 3), dtype=np.float32)
    for start in range(0, height, _ROWS):
        final[start : start + _ROWS] = cm.srgb_uint16_to_linear(encoded[start : start + _ROWS])
    del encoded
    measured = measure(final, source_linear)
    del final
    gc.collect()
    if lanczos is None:
        baseline = baselines.resample(source_linear, width, height, "lanczos")
        lanczos = measure(baseline, source_linear)
        del baseline
        gc.collect()
    return {
        "compared": "final 16-bit TIFF master vs the Lanczos baseline at the same size",
        "size": [width, height],
        "final": measured,
        "lanczos": lanczos,
        "improvement": improvement(measured, lanczos, gates),
        "note": (
            "Detail energy is the variance of the Laplacian of sRGB luma; it also rises with ringing, noise and "
            "seams, so read it with source consistency, clipping and the crops at 100%. Deliberate tone changes "
            "(night shadow lift, highlight shoulder) show up as colour difference."
        ),
    }


# --------------------------------------------------------------------------- the run


def run_universal(
    source_path: str | Path,
    config: UniversalConfig,
    project_root: str | Path,
    *,
    notebook_version: str | None = None,
    run_id: str | None = None,
    log: Callable[[str], None] = print,
) -> UniversalRunResult:
    """Run the Phase 4 universal pipeline on one source image and end with one final full-size image."""
    experiment = config.experiment
    project_root = Path(project_root).resolve()
    inputs_root, output_root, reports_root = resolve_roots(experiment, project_root)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    monitor = environment.MemoryMonitor().start()
    stages: list[StageRecord] = []
    manifest: dict = {
        "schema": SCHEMA,
        "phase": PHASE,
        "notice": NOTICE,
        "started_at": started_at.isoformat(timespec="seconds"),
        "package": {
            "name": "synthetic_raw_lab",
            "version": __version__,
            "git": environment.git_revision(Path(__file__).parent) or environment.git_revision(project_root),
        },
        "notebook_version": notebook_version,
        "models": [],
        "random_seeds": {"random_seed": experiment.random_seed, "used_by": "simulated dither (and grain if set)"},
        "config": config.to_dict(),
        "runtime": environment.inspect_runtime(output_root),
        "stages": [],
        "status": "running",
    }
    paths: dict[str, Path] = {}

    def stage(record: StageRecord):
        return _stage(record, stages, monitor, log)

    def fail(error: StageError) -> StageError:
        manifest["status"] = f"failed in stage {error.stage}"
        manifest["error"] = {"stage": error.stage, "message": str(error), "traceback": traceback.format_exc()}
        error.manifest_path = _finish(manifest, stages, monitor, started, paths, log)
        return error

    try:
        with stage(
            StageRecord(
                "ingest_and_validate",
                PRESERVATION,
                {"source_color_space": experiment.source_color_space},
                "source file",
                "float32 linear working space",
            )
        ) as record:
            source = ingestion.ingest(
                source_path, inputs_root, experiment.input_category, source_color_space=experiment.source_color_space
            )
            record.notes = list(source.warnings)
        run_id = run_id or make_run_id(source.sha256, experiment.profile, started_at)
        masters_dir = output_root / experiment.input_category / run_id
        reports_dir = reports_root / run_id
        figures_dir = reports_dir / "figures"
        paths.update(masters=masters_dir, reports=reports_dir, figures=figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        masters_dir.mkdir(parents=True, exist_ok=True)
        manifest.update(
            run_id=run_id,
            source={
                "original_path": str(source.original_path),
                "stored_copy": str(source.stored_path),
                "sha256": source.sha256,
                "pixel_sha256": source.pixel_sha256,
                "width": source.width,
                "height": source.height,
                "color_space": source.color_space.to_dict(),
            },
            validation=_without_histograms(source.validation),
        )
        report_builder.histogram_figure(source.validation["histograms"], figures_dir / "source_histograms.png")

        with stage(
            StageRecord(
                "scene_analysis",
                "Measurement and optional face inference (no image data)",
                {"face_detector": config.analysis.detector, "manual_override": config.analysis.manual_override},
                "float32 linear",
                "statistics, face boxes, category",
            )
        ) as record:
            manifest["scene"], detections = analyse_scene(source.linear, config, project_root)
            classification = manifest["scene"]["classification"]
            manifest["preset"] = preset_record(config, classification)
            manifest["figures"] = {"scene": str(scene_figure(source.linear, detections, figures_dir / "scene.png"))}
            record.notes = classification["reasons"]
        category = classification["category"]
        log(f"scene: {category} ({classification['confidence']} confidence) -> {classification['pipeline']}")

        if category == "human":
            route = _human_route(source_path, config, project_root, run_id, manifest, stage, log, notebook_version)
        elif category == "night":
            route = _night_route(source_path, config, project_root, run_id, manifest, stage, log, notebook_version)
        else:
            route = _common_route(source, detections, config, project_root, run_id, manifest, paths, stage, log)
        manifest["notice"] = route["notice"]

        with stage(
            StageRecord(
                "standard_evaluation",
                MEASUREMENT,
                {"reference": "Lanczos baseline at the final size", "gates": config.improvement.to_dict()},
                "uint16 sRGB TIFF",
                "metrics",
            )
        ):
            manifest["evaluation"] = standard_evaluation(
                source.linear,
                Path(route["exports"]["tiff16"]["path"]),
                config.improvement,
                route.pop("_lanczos", None),
                preview=figures_dir / "final_preview.jpg" if experiment.export_review else None,
                max_edge=experiment.review_max_edge,
            )
        if (figures_dir / "final_preview.jpg").exists():
            manifest["figures"]["final"] = str(figures_dir / "final_preview.jpg")
        with stage(
            StageRecord("final_output", PACKAGING, {"folder": final_output.FINAL_DIR}, "masters", "final TIFF and EXR")
        ):
            manifest["final_output"] = final_image(route, masters_dir, figures_dir)
        manifest["status"] = "complete"
    except StageError as error:
        raise fail(error) from error.__cause__
    except Exception as error:
        last = stages[-1].name if stages else "start"
        raise fail(StageError(f"unlogged step after {last}", error, None)) from error

    manifest_path = _finish(manifest, stages, monitor, started, paths, log)
    return UniversalRunResult(
        manifest["run_id"], manifest, manifest_path, paths["reports"] / "report.md", paths["masters"], paths["reports"]
    )


# --------------------------------------------------------------------------- specialised routes


def _human_route(source_path, config, project_root, run_id, manifest, stage, log, notebook_version) -> dict:
    """The notebook 01 pipeline; its conservative SR must pass the seam gate or the run is repeated without SR."""
    route = manifest["route"] = {"category": "human", "pipeline": scene_analysis.PIPELINES["human"], "attempts": []}
    for sr_enabled in (True, False) if config.sr_enabled else (False,):
        delegate_id = f"{run_id}_human" + ("" if sr_enabled else "_lanczos")
        attempt = {"run_id": delegate_id, "sr_enabled": sr_enabled}
        route["attempts"].append(attempt)
        with stage(StageRecord("human_pipeline", DELEGATED, dict(attempt), "source file", "masters and report")):
            human_config = config.human_config(project_root, sr_enabled=sr_enabled)
            attempt["sr_backend"] = human_config.sr_backend
            try:
                result = run_human_crowd(
                    source_path,
                    human_config,
                    project_root,
                    notebook_version=notebook_version,
                    run_id=delegate_id,
                    log=log,
                )
            except StageError as error:
                attempt.update(status="failed", failed_stage=error.stage, manifest=str(error.manifest_path))
                raise
        delegated = result.manifest
        attempt.update(
            status=delegated["status"], manifest=str(result.manifest_path), report=str(result.report_path), gate=None
        )
        if sr_enabled:
            summary = delegated["super_resolution"]["modes"].get("conservative") or {}
            seams = summary.get("seams") or {}
            attempt["gate"] = {"mode": "conservative", "passed": bool(seams.get("passed")), "seams": seams}
            if not attempt["gate"]["passed"]:
                log("human: conservative SR failed its seam gate; re-running with the Lanczos stand-in")
                continue
        break
    learned = human_config.sr_backend != "lanczos_stand_in"
    manifest["models"] = delegated.get("models", [])
    route.update(
        method="human_sr_conservative" if learned else "human_lanczos",
        label=(
            f"notebook 01: conservative SR ({human_config.sr_backend}) with identity protection"
            if learned
            else "notebook 01: Lanczos stand-in with identity protection"
        ),
        delegate={"run_id": result.run_id, "manifest": str(result.manifest_path), "report": str(result.report_path)},
        target=delegated["target_resolution"],
        exports=delegated["exports"],
        notice=delegated["notice"],
        source_consistency=delegated.get("source_consistency"),
    )
    return route


def _night_route(source_path, config, project_root, run_id, manifest, stage, log, notebook_version) -> dict:
    """The notebook 02 pipeline; an SR night input must pass the seam gate or the run is repeated without SR."""
    route = manifest["route"] = {"category": "night", "pipeline": scene_analysis.PIPELINES["night"], "attempts": []}
    for sr_enabled in (True, False) if config.sr_enabled else (False,):
        delegate_id = f"{run_id}_night" + ("" if sr_enabled else "_lanczos")
        attempt = {"run_id": delegate_id, "sr_enabled": sr_enabled}
        route["attempts"].append(attempt)
        with stage(StageRecord("night_pipeline", DELEGATED, dict(attempt), "source file", "masters and report")):
            night_config = config.night_config(project_root, sr_enabled=sr_enabled)
            attempt["sr_mode"] = night_config.night.super_resolution.mode
            try:
                result = run_night(
                    source_path,
                    night_config,
                    project_root,
                    notebook_version=notebook_version,
                    run_id=delegate_id,
                    log=log,
                )
            except StageError as error:
                attempt.update(status="failed", failed_stage=error.stage, manifest=str(error.manifest_path))
                raise
        delegated = result.manifest
        slot = delegated["night"]["input"]["slot"]
        attempt.update(
            status=delegated["status"],
            manifest=str(result.manifest_path),
            report=str(result.report_path),
            input=slot,
            gate=None,
        )
        if slot.endswith("_sr"):
            mode = slot.removesuffix("_sr")
            seams = delegated["super_resolution"]["modes"][mode].get("seams") or {}
            attempt["gate"] = {"mode": mode, "passed": bool(seams.get("passed")), "seams": seams}
            if not attempt["gate"]["passed"]:
                log(f"night: {mode} SR failed its seam gate; re-running the night pipeline without SR")
                continue
        break
    manifest["models"] = delegated.get("models", [])
    route.update(
        method=f"night_{slot}",
        label=f"notebook 02: {delegated['night']['input']['label']}, then Layer 5B night processing",
        delegate={"run_id": result.run_id, "manifest": str(result.manifest_path), "report": str(result.report_path)},
        target=delegated["target_resolution"],
        exports=delegated["final"]["exports"],
        notice=delegated["notice"],
        source_consistency=delegated["final"].get("source_consistency"),
    )
    return route


# --------------------------------------------------------------------------- the common pipeline


def face_safe(params: night.NightParams) -> night.NightParams:
    """The face policy of the common pipeline: the same tone and colour, no detail-changing operations."""
    return params.replace(
        texture={"sharpening_strength": 0.0, "atmospheric_detail_strength": 0.0, "atmospheric_softness": 0.0},
        noise={"denoise_source": "off", "grain_amount": 0.0},
        optics={"bloom_amount": 0.0},
    )


def face_masks(
    detections: face_preservation.FaceDetections,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    protection,
) -> list[dict]:
    """Output-scale face mask rectangles: each face box grown by the margin, with its feather width."""
    sx, sy = output_size[0] / source_size[0], output_size[1] / source_size[1]
    masks = []
    for face in detections.faces:
        grow_x, grow_y = protection.margin * face.width, protection.margin * face.height
        x0 = max(0, int(np.floor((face.x - grow_x) * sx)))
        y0 = max(0, int(np.floor((face.y - grow_y) * sy)))
        x1 = min(output_size[0], int(np.ceil((face.x + face.width + grow_x) * sx)))
        y1 = min(output_size[1], int(np.ceil((face.y + face.height + grow_y) * sy)))
        feather = max(1, round(protection.feather * min(face.width * sx, face.height * sy)))
        masks.append(
            {"box": [x0, y0, x1, y1], "feather_px": feather, "face": [face.x, face.y, face.width, face.height]}
        )
    return masks


def mask_weight(box: list[int], feather: int, output_size: tuple[int, int]) -> np.ndarray:
    """Weights of a face mask rectangle: 1 inside, a raised-cosine ramp over ``feather`` pixels at its edges
    (no ramp where the rectangle meets the image border)."""
    x0, y0, x1, y1 = box
    width, height = output_size

    def ramp(n: int, at_start: bool, at_end: bool) -> np.ndarray:
        position = np.arange(n, dtype=np.float32) + 0.5
        distance = np.minimum(
            position if not at_start else np.full(n, np.inf, np.float32),
            n - position if not at_end else np.full(n, np.inf, np.float32),
        )
        t = np.clip(distance / feather, 0, 1)
        return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)

    across = ramp(x1 - x0, x0 == 0, x1 == width)
    down = ramp(y1 - y0, y0 == 0, y1 == height)
    return np.minimum(down[:, None], across[None, :])


def _crops(source_linear: np.ndarray, detections, experiment) -> list[comparison.Crop]:
    """The shared subject-first evaluation crops (``comparison.select_crops``), led by the primary observed face.

    The subject analysis is built from this run's own face observations, so no second detector runs.
    """
    encoded = cm.linear_to_srgb(np.clip(source_linear, 0, 1))
    height, width = encoded.shape[:2]
    order = sorted(
        range(len(detections.faces)),
        key=lambda i: (i != detections.primary_index, -detections.faces[i].width * detections.faces[i].height),
    )
    found = tuple(subjects.Face(f.x, f.y, f.width, f.height) for f in (detections.faces[i] for i in order))
    point, basis = (None, "")
    if not found:
        point, basis = subjects.center_of_interest(quality_metrics.luma(encoded), (width, height))
    analysis = subjects.SubjectAnalysis(width, height, found, detections.method, detections.description, point, basis)
    crops = comparison.select_crops(
        encoded, experiment.crop_size, experiment.crop_count, experiment.crops, subjects=analysis
    )
    del encoded
    return crops


def _keep_tiles(tiles: dict, boxes: dict, label: str, image: np.ndarray) -> None:
    for name, (x0, y0, x1, y1) in boxes.items():
        tile, _ = cm.encode_srgb_uint16(image[y0:y1, x0:x1])
        tiles[name][label] = cm.srgb_uint16_to_uint8(tile)


def _common_route(source, detections, config, project_root, run_id, manifest, paths, stage, log) -> dict:
    experiment = config.experiment
    category = manifest["scene"]["classification"]["category"]
    masters_dir, figures_dir = paths["masters"], paths["figures"]
    route = manifest["route"] = {"category": category, "pipeline": scene_analysis.PIPELINES[category]}
    with stage(StageRecord("target_planning", MEASUREMENT, {}, "-", "-")):
        target = compute_target_resolution(
            source.width,
            source.height,
            experiment.effective_target_megapixels,
            mode=experiment.resolution_mode,
            pixel_count_tolerance=experiment.pixel_count_tolerance,
            dimension_multiple=experiment.dimension_multiple,
            preserve_aspect_ratio=experiment.preserve_aspect_ratio,
            max_aspect_ratio_error=experiment.max_aspect_ratio_error,
        )
    size = (target.width, target.height)
    route["target"] = target.to_dict()
    manifest["memory"] = {"estimates": environment.estimate_memory(*size)}
    log(f"source {source.width}x{source.height} -> target {target.width}x{target.height} ({target.megapixels:.3f} MP)")
    with stage(StageRecord("preset", SETUP, {"policy": "universal" if category == "general" else "night"}, "-", "-")):
        params = config.processing if category == "general" else config.night_config(project_root).night
    crops = _crops(source.linear, detections, experiment)
    boxes = {c.name: c.output_box((source.width, source.height), size) for c in crops}
    tiles: dict[str, dict[str, np.ndarray]] = {c.name: {} for c in crops}
    route["crops"] = [c.to_dict() for c in crops]

    with stage(
        StageRecord("lanczos_baseline", INTERPOLATION, {"size": f"{size[0]}x{size[1]}"}, "float32 linear", "float32")
    ):
        image = baselines.resample(source.linear, *size, "lanczos")
    with stage(StageRecord("measure_lanczos", MEASUREMENT, {}, "float32 linear", "metrics")):
        lanczos = measure(image, source.linear)
        _keep_tiles(tiles, boxes, "Lanczos", image)
        if experiment.export_review:
            report_builder.write_preview(
                cm.encode_srgb_uint16(image)[0], figures_dir / "lanczos_preview.jpg", experiment.review_max_edge
            )
    candidates = {"lanczos": {"label": "Lanczos baseline", "measurements": lanczos}}
    chosen, chosen_label = "lanczos", "Lanczos"

    if config.sr_enabled:
        sr_record = manifest["super_resolution"] = sr_run_record(experiment, target)
    else:
        sr_record = manifest["super_resolution"] = {"enabled": False, "modes": {}}
    if config.sr_enabled and "skipped" not in sr_record:
        with stage(StageRecord("super_resolution_setup", SETUP, {"backend": experiment.sr_backend}, "-", "-")):
            weights_dir = sr_setup(experiment, project_root)
        loaded: dict = {}
        try:
            for index, mode in enumerate(config.sr_modes):
                output = run_sr_mode(
                    mode, source.linear, target, experiment, weights_dir, loaded, manifest, figures_dir, stage, log
                )
                release_sr_models(loaded, experiment, config.sr_modes[index + 1 :])
                with stage(StageRecord(f"improvement_gate_{mode}", MEASUREMENT, {}, "float32 linear", "metrics")):
                    measured = measure(output, source.linear)
                    verdict = improvement(measured, lanczos, config.improvement, sr_record["modes"][mode].get("seams"))
                    _keep_tiles(tiles, boxes, f"{mode} SR", output)
                    if experiment.export_review:
                        report_builder.write_preview(
                            cm.encode_srgb_uint16(output)[0],
                            figures_dir / f"sr_{mode}_preview.jpg",
                            experiment.review_max_edge,
                        )
                candidates[f"sr_{mode}"] = {"label": f"{mode} SR", "measurements": measured, "verdict": verdict}
                log(f"{mode} SR: {'kept' if verdict['passed'] else 'rejected (' + ', '.join(verdict['failed']) + ')'}")
                if verdict["passed"]:
                    chosen, chosen_label = f"sr_{mode}", f"{mode} SR"
                    image = output
                    break
                del output
                gc.collect()
        finally:
            release_sr_models(loaded, experiment, ())
            gc.collect()
        manifest["models"] = [m["model"] for m in sr_record["modes"].values() if "model" in m]
    if chosen == "lanczos":
        if not config.sr_enabled:
            why = "super-resolution is disabled"
        elif "skipped" in sr_record:
            why = f"super-resolution was skipped: {sr_record['skipped']}"
        else:
            why = "no SR rung passed the seam and measurable-improvement gates"
    else:
        why = f"{chosen_label} is the most conservative rung that passed the seam and measurable-improvement gates"
    route["selection"] = {"input": chosen, "label": chosen_label, "why": why, "candidates": candidates}
    log(f"common pipeline input: {chosen_label} ({why})")

    with stage(StageRecord("combined_masks", MEASUREMENT, {}, "float32 linear", "region labels and face masks")):
        regions = night.segment_regions(source.linear)
        masks = face_masks(detections, (source.width, source.height), size, config.face_protection)
        route["masks"] = {
            "regions": regions.to_dict(),
            "faces": masks,
            "face_policy": FACE_SAFE_NOTE if masks else None,
        }
        manifest["figures"]["regions"] = str(region_figure(regions, source.linear, figures_dir / "regions.png"))

    processing_record = StageRecord(
        "universal_processing",
        UNIVERSAL_PROCESSING,
        {"policy": "night" if category == "night_with_faces" else "universal", "face_masks": len(masks)},
        "float32 linear",
        "float32 linear",
    )
    with stage(processing_record):
        patches = []
        if masks:
            safe = image.copy()
            night.process_night(safe, regions, source.linear, face_safe(params), seed=experiment.random_seed, out=safe)
            patches = [safe[m["box"][1] : m["box"][3], m["box"][0] : m["box"][2]].copy() for m in masks]
            del safe
            gc.collect()
        result = night.process_night(image, regions, source.linear, params, seed=experiment.random_seed, out=image)
        for mask, patch in zip(masks, patches, strict=True):
            x0, y0, x1, y1 = mask["box"]
            weight = mask_weight(mask["box"], mask["feather_px"], size)[..., None]
            image[y0:y1, x0:x1] += weight * (patch - image[y0:y1, x0:x1])
        del patches
        route["processing"] = {
            "params": params.to_dict(),
            "stages": result.stages,
            "fallbacks": len(result.fallbacks),
            "fallback_fraction": result.fallback_fraction,
            "information": result.information,
            "comparison": result.comparison,
        }
        processing_record.notes = [
            f"failure rule: {len(result.fallbacks)} fallback blocks ({result.fallback_fraction:.3%})"
        ]
        manifest["figures"]["fallbacks"] = str(
            fallback_figure(
                night.fallback_map(result, *size, max_edge=experiment.review_max_edge),
                source.linear,
                result.fallbacks,
                result.fallback_fraction,
                figures_dir / "fallback_map.png",
                enabled=params.failure_rule.enabled,
            )
        )
        del result

    if detections.faces:
        with stage(StageRecord("face_diagnostics", MEASUREMENT, {"faces": len(detections.faces)}, "float32", "-")):
            route["faces"] = face_preservation.evaluate_face_identity(source.linear, image, detections)

    learned = chosen != "lanczos"
    notice = NOTICE_SR if learned else NOTICE
    provenance = {
        "notice": notice,
        "run_id": run_id,
        "source_name": source.original_path.name,
        "source_sha256": source.sha256,
        "source_dimensions": [source.width, source.height],
        "output_dimensions": list(size),
        "category": category,
        "method": f"{chosen_label}, then universal processing ({processing_record.parameters['policy']} policy)",
        "stage_provenance": {s["name"]: s["provenance"] for s in route["processing"]["stages"] if s["enabled"]},
        "information": route["processing"]["information"],
        "face_masks": len(masks),
        "package_version": __version__,
    }
    if learned:
        weights = sr_models.WEIGHTS[sr_record["modes"][chosen.removeprefix("sr_")]["weights"]]
        provenance.update(model_weights_sha256=weights.sha256, model_license=weights.license)
    exports = route["exports"] = {}
    label = f"{size[0]}x{size[1]}"
    stem = f"{_stem(source)}_universal_{label}"
    if experiment.export_exr:
        with stage(StageRecord("export_exr", PACKAGING, {"precision": experiment.exr_precision}, "float32", "EXR")):
            suffix = "half" if experiment.exr_precision == "half" else "f32"
            exported = exporters.write_exr(
                masters_dir / f"{stem}_lin_rec709_{suffix}.exr",
                image,
                _exr_attributes(run_id, source, "Universal master", notice=notice),
                precision=experiment.exr_precision,
                compression=experiment.exr_compression,
            )
            exporters.write_sidecar(exported, provenance)
            exports["exr"] = exported.to_dict()
    with stage(StageRecord("encode_srgb16", PACKAGING, {"transfer": "sRGB"}, "float32", "uint16 sRGB")):
        encoded, clip_stats = cm.encode_srgb_uint16(image)
        route["tiff_clipping"] = clip_stats.to_dict()
        _keep_tiles(tiles, boxes, "final", image)
        del image
        gc.collect()
    # The final image is the TIFF master, so it is written even when export_tiff16 is off for other runs.
    with stage(StageRecord("export_tiff16", PACKAGING, {"compression": experiment.tiff_compression}, "uint16", "TIFF")):
        exported = exporters.write_tiff16(
            masters_dir / f"{stem}_srgb16.tif",
            encoded,
            {**provenance, "software": f"synthetic_raw_lab {__version__}"},
            compression=experiment.tiff_compression,
        )
        exporters.write_sidecar(exported, provenance)
        exports["tiff16"] = exported.to_dict()
        del encoded

    with stage(StageRecord("comparisons", MEASUREMENT, {"crops": len(crops)}, "uint8 tiles", "review PNG figures")):
        source_encoded = cm.linear_to_srgb(np.clip(source.linear, 0, 1))
        panels = []
        for crop in crops:
            tile = source_encoded[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
            source_tile = np.round(tile * 255).astype(np.uint8)
            path = exporters.write_review_image(
                figures_dir / f"crop_{crop.name}.png",
                comparison.crop_panel(source_tile, tiles[crop.name], (chosen_label, "final")),
            )
            panels.append({"crop": crop.name, "path": str(path), "box": list(boxes[crop.name])})
        route["crop_panels"] = panels
        del source_encoded

    route.update(
        method=f"universal_{chosen}",
        label=f"{chosen_label}, then universal processing ({processing_record.parameters['policy']} policy)",
        notice=notice,
        _lanczos=lanczos,
    )
    return route


# --------------------------------------------------------------------------- final image


def final_image(route: dict, masters_dir: Path, figures_dir: Path | None = None) -> dict:
    """The run's final image: the route's masters saved again under ``final/`` by the shared final-output step
    (``final_output.publish``: hard links where possible, verified by SHA-256)."""
    record = {
        "method": route["method"],
        "rule": "universal",
        "reason": (f"the {route['category']} scene's one final master, from the {route['pipeline']}: {route['label']}"),
        "skipped": [],
        "notes": [],
        "category": route["category"],
        "label": route["label"],
    }
    record = final_output.publish(record, route["exports"], masters_dir)
    if figures_dir is not None and (Path(figures_dir) / "final_preview.jpg").exists():
        record["preview"] = "figures/final_preview.jpg"
    return record


def archive_reports(manifest: dict, reports_dir: Path) -> Path:
    """Zip this run's report folder with the delegated specialised run's report folders (PRD 21.5: reports are
    archived separately from the full-resolution masters)."""
    import zipfile

    reports_dir = Path(reports_dir)
    folders = [reports_dir]
    for attempt in manifest.get("route", {}).get("attempts", []):
        if attempt.get("report"):
            folders.append(Path(attempt["report"]).parent)
    archive = reports_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for folder in dict.fromkeys(folders):
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(folder.parent))
    return archive


# --------------------------------------------------------------------------- manifest and report


def _finish(manifest: dict, stages: list[StageRecord], monitor, started: float, paths: dict, log) -> Path | None:
    from .universal_report import write_universal_report

    monitor.stop()
    manifest["stages"] = [s.to_dict() for s in stages]
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["total_seconds"] = round(time.perf_counter() - started, 3)
    manifest.setdefault("memory", {})
    manifest["memory"]["pipeline_peak_rss_mb"] = round(monitor.peak_rss / environment.MB, 1)
    manifest["memory"]["process_lifetime_peak_rss_mb"] = environment.process_peak_rss_mb()
    if "reports" not in paths:
        log("run failed before an output folder existed; nothing written")
        return None
    manifest.get("route", {}).pop("_lanczos", None)
    manifest["storage_used_bytes"] = final_output.storage_bytes(paths["masters"], paths["reports"])
    exports = list(manifest.get("route", {}).get("exports", {}).values())
    exports += list((manifest.get("final_output") or {}).get("files", {}).values())
    manifest["output_checksums"] = {Path(e["path"]).name: e["file_sha256"] for e in exports}
    manifest_path = paths["reports"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    write_universal_report(manifest, paths["reports"] / "report.md")
    log(f"report: {paths['reports'] / 'report.md'}")
    return manifest_path
