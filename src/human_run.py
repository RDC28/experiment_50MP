"""Notebook 01 file-to-report orchestration with backend-specific provenance."""

import json
import time
from pathlib import Path

from . import baselines, environment, exporters, ingestion, quality_metrics, report_builder
from . import color_management as cm
from .face_preservation import haar_model_metadata
from .human_comparisons import write_human_comparisons
from .human_models import yunet_model_metadata
from .human_pipeline import STAND_IN_DESCRIPTION
from .human_reconstruction import reconstruct_human_crowd
from .pipeline import (
    MEASUREMENT,
    NOTICE,
    SR_NOTICE,
    RunResult,
    StageError,
    StageRecord,
    _exr_attributes,
    _provenance,
    _stage,
    _without_histograms,
    make_run_id,
    resolve_roots,
)
from .resolution import compute_target_resolution


def run_human_crowd(source_path, config, project_root, *, notebook_version=None, run_id=None, log=print):
    """Run source face observations through masters and comparisons, preserving partial files.

    Quick enlarges the source 1.5x in pixel count, at most the configured target.
    Detection and source consistency are never identity proof. Failed runs retain
    a manifest, every completed stage record and a review report.
    """
    learned = config.sr_backend != "lanczos_stand_in"
    sr_notice = SR_NOTICE if learned else STAND_IN_DESCRIPTION
    started = time.perf_counter()
    inputs, outputs, reports = resolve_roots(config, Path(project_root).resolve())
    run_id = run_id or make_run_id(ingestion.sha256_file(Path(source_path)), config.profile)
    masters, report_dir = outputs / config.input_category / run_id, reports / run_id
    masters.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "synthetic_raw_lab.human_crowd.manifest/1",
        "run_id": run_id,
        "notice": SR_NOTICE if learned else NOTICE,
        "sr_notice": sr_notice,
        "notebook_version": notebook_version,
        "config": config.to_dict(),
        "models": [],
        "status": "running",
        "exports": {},
    }
    stages = []
    error = None
    with environment.MemoryMonitor() as monitor:

        def stage(name, classification):
            return _stage(StageRecord(name, classification), stages, monitor, log)

        try:
            with stage("ingest_and_validate", "Preservation"):
                source = ingestion.ingest(
                    source_path, inputs, config.input_category, source_color_space=config.source_color_space
                )
                manifest["source"] = {
                    "original_path": str(source.original_path),
                    "stored_copy": str(source.stored_path),
                    "sha256": source.sha256,
                    "pixel_sha256": source.pixel_sha256,
                    "width": source.width,
                    "height": source.height,
                    "color_space": source.color_space.to_dict(),
                }
                manifest["validation"] = _without_histograms(source.validation)
            with stage("target_planning", MEASUREMENT):
                target = compute_target_resolution(
                    source.width,
                    source.height,
                    config.planned_megapixels(source.width, source.height),
                    mode=config.resolution_mode,
                    pixel_count_tolerance=config.pixel_count_tolerance,
                    dimension_multiple=config.dimension_multiple,
                    preserve_aspect_ratio=config.preserve_aspect_ratio,
                    max_aspect_ratio_error=config.max_aspect_ratio_error,
                )
                manifest["target_resolution"] = target.to_dict()
                if target.width < source.width or target.height < source.height:
                    raise ValueError(
                        f"target {target.width} x {target.height} does not enlarge the "
                        f"{source.width} x {source.height} source on both axes"
                    )
                manifest["runtime"] = environment.inspect_runtime(outputs)
            with stage("human_reconstruction", "Simulation"):
                if config.face_detector == "haar":
                    manifest["models"] = [haar_model_metadata()]
                elif config.face_detector == "yunet":
                    manifest["models"] = [yunet_model_metadata()]
                result = reconstruct_human_crowd(
                    source.linear,
                    target.width,
                    target.height,
                    config,
                    log=log,
                    project_root=project_root,
                    stages=stages,
                )
                modes = {
                    record.parameters["mode"]: record.parameters
                    for record in result.stages
                    if record.name.startswith("human_upscale_")
                }
                manifest["super_resolution"] = {
                    "backend": config.sr_backend,
                    "final_mode": "conservative",
                    "balanced_usage": "comparison only",
                    "modes": modes,
                }
                for summary in modes.values():
                    model = summary.get("model")
                    if model is not None and model not in manifest["models"]:
                        manifest["models"].append(model)
                manifest["primary_subject"] = {
                    "index": result.detections.primary_index,
                    "rule": result.detections.primary_selection_rule,
                    "classification": MEASUREMENT,
                    "note": "Protection heuristic only; selection does not establish identity or intended subject.",
                }
                manifest.update(identity=result.identity, failure_events=result.failure_events, crowd=result.crowd)
            provenance = _provenance(run_id, source, "lanczos", target)
            provenance.update(
                notice=SR_NOTICE if learned else NOTICE,
                sr_notice=sr_notice,
                config=config.to_dict(),
                super_resolution=manifest["super_resolution"],
            )
            if learned:
                provenance["method"] = "Conservative learned SR with human protection and Lanczos identity fallback"
                provenance["stage_classification"].insert(1, "Inference")
                provenance["information"]["inferred"] = SR_NOTICE
                provenance["information"]["interpolated"] = (
                    "Lanczos baseline, native x4 SR resampled to target, and original-derived identity fallback"
                )
            provenance["stage_classification"].insert(-1, "Simulation")
            provenance["information"]["simulated"] = (
                "Bounded region microcontrast, optional skin-weighted bilateral smoothing, "
                "face-excluding edge sharpening, "
                "log-luminance tonal contrast, linear RGB colour balance and optional seeded grain "
                "with identity-rule recovery"
            )
            manifest["provenance"] = provenance
            with stage("export_exr", "Packaging"):
                exported = exporters.write_exr(
                    masters / "final_lin_rec709_half.exr",
                    result.final,
                    {
                        **_exr_attributes(run_id, source, "Human and crowd reconstruction", notice=sr_notice),
                        "humanProvenance": json.dumps(provenance),
                    },
                    precision="half",
                    compression=config.exr_compression,
                )
                exporters.write_sidecar(exported, provenance)
                manifest["exports"]["exr"] = exported.to_dict()
            with stage("export_tiff16", "Packaging"):
                encoded, clipping = cm.encode_srgb_uint16(result.final)
                manifest["tiff_clipping"] = clipping.to_dict()
                exported = exporters.write_tiff16(
                    masters / "final_srgb16.tif", encoded, provenance, compression=config.tiff_compression
                )
                exporters.write_sidecar(exported, provenance)
                manifest["exports"]["tiff16"] = exported.to_dict()
            with stage("comparisons_and_review", "Packaging"):
                manifest["comparisons"] = write_human_comparisons(
                    source.linear, result, report_dir / "figures", tier_config=config.face_tiers
                )
                preview = report_dir / "figures" / "final_review.jpg"
                report_builder.write_preview(encoded, preview, config.review_max_edge)
                manifest["review"] = {
                    "path": str(preview),
                    "classification": "Packaging",
                    "notice": sr_notice,
                }
            with stage("source_consistency", MEASUREMENT):
                reduced = baselines.downsample_area(cm.srgb_uint16_to_linear(encoded), source.width, source.height)
                metrics, delta = quality_metrics.source_consistency(source.linear, reduced)
                manifest["source_consistency"] = metrics
                manifest["statistics"] = quality_metrics.image_statistics(result.final)
                report_builder.heatmap_figure(
                    delta,
                    report_dir / "figures" / "delta_e_final.png",
                    "Final reduced to source: CIEDE2000",
                    config.review_max_edge,
                )
            manifest["status"] = "complete"
        except StageError as caught:
            error = caught
            manifest["status"] = f"failed in stage {caught.stage}"
            manifest["error"] = {
                "stage": caught.stage,
                "message": str(caught),
                "retry": _retry_advice(caught),
            }
        except Exception as caught:
            last = stages[-1].name if stages else "start"
            error = StageError(f"unlogged step after {last}", caught, None)
            manifest["status"] = "failed"
            manifest["error"] = {"stage": error.stage, "message": f"{type(caught).__name__}: {caught}"}
    manifest["stages"] = [record.to_dict() for record in stages]
    manifest["total_seconds"] = time.perf_counter() - started
    manifest["memory"] = {"pipeline_peak_rss_mb": monitor.peak_rss / environment.MB}
    manifest["output_checksums"] = {
        str(path): ingestion.sha256_file(path)
        for root in (masters, report_dir)
        for path in root.rglob("*")
        if path.is_file() and path.name not in ("manifest.json", "report.md")
    }
    manifest_path, report_path = report_dir / "manifest.json", report_dir / "report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    report_path.write_text(render_human_report(manifest))
    if error:
        error.manifest_path = manifest_path
        raise error
    return RunResult(run_id, manifest, manifest_path, report_path, masters, report_dir)


def _retry_advice(error: StageError) -> str:
    """Retry guidance for the failing stage and cause, never a generic size reduction."""
    cause = error.__cause__
    if isinstance(cause, MemoryError) or type(cause).__name__ == "OutOfMemoryError":
        return (
            "Out of memory: use runtime_preset=quick (1.5x the source pixel count) "
            "or a smaller sr_tile_size, and check the per-stage peak memory."
        )
    if error.stage.startswith("human_upscale_"):
        return (
            "Learned SR failed: install the [sr] extra, check the verified weights in sr_weights_dir and the "
            "device, or set sr_backend: lanczos_stand_in for explicit interpolation."
        )
    if error.stage == "target_planning":
        return "Set target_megapixels so the target enlarges the source on both axes; quick always does."
    return f"Fix the cause reported for stage {error.stage}; completed stage records are in this manifest."


def _report_numbers(value):
    """Round display values to four significant digits without changing evidence."""
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {key: _report_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_numbers(item) for item in value]
    return value


def render_human_report(manifest):
    """Human evidence report; unavailable checks never imply visual acceptance."""
    manifest = _report_numbers(manifest)

    def table(header, rows):
        return report_builder._table(header, (_report_numbers(row) for row in rows))

    crowd = manifest.get("crowd", {})
    events = manifest.get("failure_events", [])
    faces = manifest.get("identity", {}).get("faces", [])
    flagged = {event["face_index"] for event in events}
    tier_rows = []
    for row in manifest.get("identity", {}).get("per_tier", []):
        selected = [face for face in faces if face["tier"] == row["tier"]]
        edges = [
            face["edge_displacement"]["mean_px"]
            for face in selected
            if face["edge_displacement"]["mean_px"] is not None
        ]
        tier_rows.append(
            (
                row["tier"],
                row["face_count"],
                row["mean_ssim"] if row["mean_ssim"] is not None else "-",
                row["mean_ciede2000"] if row["mean_ciede2000"] is not None else "-",
                sum(edges) / len(edges) if edges else "-",
                ", ".join(str(face["face_index"]) for face in selected if face["face_index"] in flagged) or "none",
                "; ".join(sorted({_landmark_status(face) for face in selected})) or "no faces",
            )
        )
    sections = [
        "# Human and crowd reconstruction",
        manifest["notice"],
        manifest["sr_notice"],
        f"Status: {manifest['status']}; runtime: {manifest['total_seconds']:.3f} s; "
        f"peak RSS: {manifest['memory']['pipeline_peak_rss_mb']:.1f} MB.",
        "## Face detection",
        next(
            (" ".join(s["notes"]) for s in manifest["stages"] if s["name"] == "face_observations"),
            "Face observations unavailable.",
        ),
        "Model artifact versions, SHA-256 and license: see the models list in [manifest.json](manifest.json) "
        "and repository MODELS.md. Detection is never identity proof.",
        "## Super-resolution",
        f"Backend: {manifest.get('super_resolution', {}).get('backend', 'unavailable')}. "
        "Final uses conservative SR; balanced is comparison only. "
        "SR weights and commercial-use terms: repository docs/models.md.",
        table(
            ["Mode", "Preset", "Weights", "Device / precision", "Tiling", "Seam checks"],
            [
                (
                    mode,
                    summary.get("preset", {}).get("label", "Lanczos interpolation"),
                    summary.get("weights", "none"),
                    f"{summary.get('device', '-')} / {summary.get('precision', '-')}",
                    summary.get("tiling", "not applicable"),
                    summary.get("seams", "not applicable"),
                )
                for mode, summary in manifest.get("super_resolution", {}).get("modes", {}).items()
            ],
        ),
        table(
            ["SR weights", "SHA-256", "Licence", "Commercial use"],
            [
                (weights["name"], weights["sha256"], weights["license"], weights["commercial_use"])
                for model in manifest.get("models", [])
                if (weights := model.get("weights")) is not None
            ],
        ),
        "## Primary subject",
        f"Selected face: {manifest.get('primary_subject', {}).get('index', 'unavailable')}; "
        f"selection rule: {manifest.get('primary_subject', {}).get('rule', 'unavailable')}. "
        "Automatic selection prefers highest source tier, then largest area, then first observation. "
        "A manual index overrides this protection heuristic; selection is not identity proof.",
        "## Per-tier identity metrics",
        "Source-scale consistency only. Thresholds require experimental calibration. "
        "Unavailable landmarks, embeddings and human judgments are not passing identity checks.",
        table(
            [
                "Tier",
                "Faces",
                "Mean SSIM",
                "Mean CIEDE2000",
                "Mean edge displacement (source px)",
                "Faces flagged",
                "Landmark status",
            ],
            tier_rows,
        ),
        "## Crowd checks and requires human review",
        f"{crowd.get('classification', MEASUREMENT)}. Warnings are estimates, not identity or anatomy proof. "
        "Unavailable checks are not passing checks.",
        _crowd_table(crowd),
        "### Requires human review",
        "\n".join(f"- {item}" for item in crowd.get("requires_human_review", []))
        or "- Review requirements unavailable.",
        "## Failure-rule events",
        table(
            ["Face", "Tier", "Step", "Metrics", "Action"],
            [
                (event["face_index"], event["tier"], index + 1, _failure_metrics(step), step["action"])
                for event in events
                for index, step in enumerate(event["steps"])
            ],
        )
        if events
        else "No failure-rule events recorded; this does not establish identity acceptance.",
        "## Stage classification and provenance",
        table(
            ["Stage", "Classification", "Seconds", "Peak RSS MB", "Input", "Output", "Notes / errors"],
            [
                (
                    s["name"],
                    s["classification"],
                    s["seconds"],
                    s["rss_peak_mb"],
                    s["input_precision"],
                    s["output_precision"],
                    "; ".join([*s["notes"], *([str(s["error"])] if s["error"] else [])]),
                )
                for s in manifest["stages"]
            ],
        ),
        "## Tonal and colour controls (Simulation)",
        table(
            ["Stage", "Parameters and measurements (before identity recovery)"],
            [
                (s["name"], s["parameters"])
                for s in manifest["stages"]
                if s["name"] in ("tonal_contrast", "colour_balance")
            ],
        ),
        "## Comparisons",
    ]
    for name, figure in manifest.get("comparisons", {}).get("figures", {}).items():
        sections.append(
            f"### {name}\n\n{figure['classification']}: {figure['description']}"
            f"\n\n![{name}](figures/{Path(figure['path']).name})\n"
        )
    information = manifest.get("provenance", {}).get("information", {})
    sections += [
        "## Information provenance (PRD 27.5)",
        table(
            ["Contribution", "Classification", "Evidence / limits"],
            [
                ("Original source data", "Preservation", information.get("preserved", "unavailable")),
                ("Interpolated values", "Interpolation", information.get("interpolated", "unavailable")),
                ("Learned inference", "Inference", information.get("inferred", "unavailable")),
                ("Region detail", "Simulation", information.get("simulated", "unavailable")),
                ("Simulated sensor effects", "Simulation", "none"),
                ("Added noise or grain", "Simulation", _grain_summary(manifest)),
                ("File encoding", "Packaging", ", ".join(manifest.get("exports", {})) or "no completed exports"),
            ],
        ),
        "Visual quality and anatomy require human review; no recovered detail or identity acceptance is established. "
        "Full configuration, metrics, checksums and stage records: [manifest.json](manifest.json).",
    ]
    if "error" in manifest:
        error = manifest["error"]
        retry = f"\n\nRetry: {error['retry']}" if "retry" in error else ""
        sections.append(f"## Failure\n\n{error['stage']}: {error['message']}{retry}")
    return "\n\n".join(sections) + "\n"


def _crowd_table(crowd):
    """Keep every measured pair and face visible, including unavailable results."""
    thresholds = crowd.get("thresholds", {})
    rows = []
    for pair in crowd.get("duplicate_pairs", []):
        available = pair["source_correlation"] is not None and pair["output_correlation"] is not None
        rows.append(
            (
                "Duplicate-looking faces",
                f"flagged={pair['flagged']}; source={pair['source_correlation']}; output={pair['output_correlation']}"
                if available
                else "unavailable",
                f"source < {thresholds.get('source_match')}; output >= {thresholds.get('duplicate')}",
                ", ".join(map(str, pair["face_indices"])),
            )
        )
    for face in crowd.get("skin_texture", []):
        rows.append(
            (
                "Skin texture variance ratio",
                f"{face['variance_ratio']}; flagged={face['flagged']}"
                if face["variance_ratio"] is not None
                else face["note"],
                f"ratio < {thresholds.get('texture_ratio')}",
                face["face_index"],
            )
        )
    new = crowd.get("new_face_like_detections")
    rows.append(
        (
            "New face-like detections",
            crowd.get("detection_note", "unavailable") if new is None else f"{len(new)} unmatched detections",
            f"IoU < {thresholds.get('overlap_iou')}",
            "unavailable" if new is None else "; ".join(str(item["box"]) for item in new) or "none",
        )
    )
    return report_builder._table(["Check", "Result", "Threshold", "Faces affected"], rows)


def _landmark_status(face):
    pixels = face.get("landmark_displacement_px")
    if pixels is None:
        return f"face {face['face_index']}: {face['landmark_note']}"
    normalized = face["landmark_displacement_normalized"]
    return (
        f"face {face['face_index']}: mean/max {pixels['mean']}/{pixels['max']} source px; "
        f"normalized {normalized['mean']}/{normalized['max']}"
    )


def _failure_metrics(step):
    metrics = step["metrics"]
    return (
        f"SSIM={metrics['ssim']}; CIEDE2000={metrics['color_difference_ciede2000']['mean']}; "
        f"skin shift={metrics['skin_tone_shift_ciede2000']}; "
        f"edge px={metrics['edge_displacement']['mean_px']}; "
        f"failed gates: {', '.join(step['failed_metrics']) or 'none'}"
    )


def _grain_summary(manifest):
    grain = next((s for s in manifest.get("stages", []) if s["name"] == "grain"), None)
    if grain is None:
        return "none"
    p = grain["parameters"]
    return (
        f"Seeded random texture, strength={p['strength']}; "
        f"linear-luma Laplacian variance before={p['high_frequency_energy_before']}, "
        f"after={p['high_frequency_energy_after']}. Not recovered detail; measured before recovery."
    )
