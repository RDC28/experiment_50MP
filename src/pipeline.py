"""Phase 0 and Phase 1 run: ingest, plan, interpolate, export, measure, report.

Every stage is logged with its PRD 5.2 classification, parameters, duration,
precision and resident-memory peak. Baselines run one at a time and each 50 MP
buffer is released before the next one starts. If a stage fails, the outputs
already written stay on disk, the failure is recorded in the manifest and
report, and :class:`StageError` names the stage.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from . import (
    __version__,
    baselines,
    comparison,
    environment,
    exporters,
    ingestion,
    quality_metrics,
    report_builder,
    sr_models,
)
from . import color_management as cm
from . import super_resolution as sr
from .config import ExperimentConfig
from .resolution import TargetResolution, compute_target_resolution

PRESERVATION = "Preservation"
INTERPOLATION = "Interpolation"
INFERENCE = "Inference"
PACKAGING = "Packaging"
MEASUREMENT = "Measurement (reads data, produces no image data)"
SETUP = "Setup (loads model weights, produces no image data)"

NOTICE = (
    "Synthetic camera-grade master reconstructed by classical interpolation from an AI-generated source. "
    "It is not a camera capture or camera RAW file and contains no recovered sensor measurements."
)
SR_NOTICE = (
    "Synthetic camera-grade master reconstructed by pretrained super-resolution from an AI-generated source. "
    "Detail above the source sampling limit was synthesised by a learned model, not recovered; it is not a "
    "camera capture or camera RAW file and contains no recovered sensor measurements."
)
RUN_NOTICE_SR = (
    "Synthetic camera-grade masters reconstructed from an AI-generated source by classical interpolation "
    "(baselines) and by pretrained super-resolution (synthesised detail). None is a camera capture or camera "
    "RAW file and none contains recovered sensor measurements."
)
PHASE_01 = "Phase 0 (feasibility) and Phase 1 (baselines)"
PHASE_SR = f"{PHASE_01} and Phase 2 (pretrained super-resolution)"
SEEDS_UNUSED = "nothing (Phase 0/1 is deterministic)"
SEEDS_SR = "torch.manual_seed before SR inference (inference itself draws no random numbers)"
LOW_MEMORY_HEADROOM_GB = 1.0
# Provisional Phase 2 gates for major seams, calibrated on the measurements in docs/models.md (PRD 25.3).
SEAM_SCORE_LIMIT = 1.0  # hard seams: excess step across a boundary line, sRGB 8-bit code values
SEAM_BLEND_ERROR_LIMIT = 2.0  # soft seams: p99 blend error, sRGB 8-bit code values


class StageError(RuntimeError):
    def __init__(self, stage: str, cause: BaseException, manifest_path: Path | None):
        super().__init__(f"stage {stage!r} failed: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.manifest_path = manifest_path


@dataclass
class StageRecord:
    name: str
    classification: str
    parameters: dict = field(default_factory=dict)
    input_precision: str = ""
    output_precision: str = ""
    enabled: bool = True
    seconds: float = 0.0
    rss_peak_mb: float | None = None
    memory_warnings: list[str] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "classification": self.classification,
            "enabled": self.enabled,
            "parameters": self.parameters,
            "input_precision": self.input_precision,
            "output_precision": self.output_precision,
            "seconds": round(self.seconds, 3),
            "rss_peak_mb": None if self.rss_peak_mb is None else round(self.rss_peak_mb, 1),
            "memory_warnings": self.memory_warnings,
            "error": self.error,
            "notes": self.notes,
        }


@dataclass
class RunResult:
    run_id: str
    manifest: dict
    manifest_path: Path
    report_path: Path
    masters_dir: Path
    reports_dir: Path


def make_run_id(source_sha256: str, profile: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{source_sha256[:8]}_{profile}"


def resolve_roots(config: ExperimentConfig, project_root: str | Path) -> tuple[Path, Path, Path]:
    """The inputs, output and reports folders a run uses; empty config values mean folders in the project."""
    project_root = Path(project_root)
    return tuple(
        Path(value) if value else project_root / default
        for value, default in (
            (config.inputs_root, "inputs"),
            (config.output_root, "outputs"),
            (config.reports_root, "reports"),
        )
    )


def run(
    source_path: str | Path,
    config: ExperimentConfig,
    project_root: str | Path,
    *,
    notebook_version: str | None = None,
    run_id: str | None = None,
    benchmark_memory: bool | None = None,
    log: Callable[[str], None] = print,
) -> RunResult:
    project_root = Path(project_root).resolve()
    inputs_root, output_root, reports_root = resolve_roots(config, project_root)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)

    runtime = environment.inspect_runtime(output_root)
    monitor = environment.MemoryMonitor().start()
    stages: list[StageRecord] = []
    manifest: dict = {
        "schema": "synthetic_raw_lab.phase01.manifest/1",
        "phase": PHASE_01,
        "notice": NOTICE,
        "started_at": started_at.isoformat(timespec="seconds"),
        "package": {
            "name": "synthetic_raw_lab",
            "version": __version__,
            "git": environment.git_revision(Path(__file__).parent) or environment.git_revision(project_root),
        },
        "notebook_version": notebook_version,
        "models": [],
        "random_seeds": {"random_seed": config.random_seed, "used_by": SEEDS_UNUSED},
        "config": config.to_dict(),
        "runtime": runtime,
        "stages": [],
        "baselines": {},
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
                {
                    "source_color_space": config.source_color_space,
                    "working_color_space": config.working_color_space,
                },
                "source file",
                "float32 linear working space",
            )
        ) as record:
            source = ingestion.ingest(
                source_path, inputs_root, config.input_category, source_color_space=config.source_color_space
            )
            record.notes = list(source.warnings)
            record.input_precision = f"{source.codes.dtype} ({source.validation['samples']['effective_bit_depth']}-bit)"

        run_id = run_id or make_run_id(source.sha256, config.profile, started_at)
        masters_dir = output_root / config.input_category / run_id
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
        (reports_dir / "input_validation.json").write_text(json.dumps(source.validation, indent=2))

        target = compute_target_resolution(
            source.width,
            source.height,
            config.effective_target_megapixels,
            mode=config.resolution_mode,
            pixel_count_tolerance=config.pixel_count_tolerance,
            dimension_multiple=config.dimension_multiple,
            preserve_aspect_ratio=config.preserve_aspect_ratio,
            max_aspect_ratio_error=config.max_aspect_ratio_error,
        )
        manifest["target_resolution"] = target.to_dict()
        manifest["memory"] = {"estimates": environment.estimate_memory(target.width, target.height)}
        manifest["storage"] = environment.estimate_storage(
            target.width,
            target.height,
            len(config.baselines) + (len(config.sr_modes) if config.enable_super_resolution else 0),
            tiff=config.export_tiff16,
            exr=config.export_exr,
            exr_bytes_per_sample=2 if config.exr_precision == "half" else 4,
            source_pixels=source.width * source.height,
            save_intermediates=config.save_intermediates,
        )
        free = runtime["disk_free_gb"] * 1e9
        if free < manifest["storage"]["upper_bound_bytes"]:
            manifest["storage"]["warning"] = (
                f"free disk ({runtime['disk_free_gb']} GB) is below the uncompressed upper bound"
            )
            log(f"WARNING: {manifest['storage']['warning']}")
        log(
            f"source {source.width}x{source.height} -> target {target.width}x{target.height} "
            f"({target.megapixels:.3f} MP, pixel error {target.pixel_count_error:+.4%}, "
            f"aspect error {target.aspect_ratio_error:+.5%})"
        )

        source_encoded16, _ = cm.encode_srgb_uint16(source.linear)
        source_display = cm.linear_to_srgb(np.clip(source.linear, 0, 1))
        crops = comparison.select_crops(source_display, config.crop_size, config.crop_count, config.crops)
        del source_display
        manifest["crops"] = [c.to_dict() for c in crops]
        crop_tiles: dict[str, dict[str, np.ndarray]] = {c.name: {} for c in crops}
        crop_stats: dict[str, dict[str, dict]] = {c.name: {} for c in crops}

        if config.export_review:
            report_builder.write_preview(source_encoded16, figures_dir / "source_preview.jpg", config.review_max_edge)

        if config.runtime_preset == "research" if benchmark_memory is None else benchmark_memory:
            with stage(
                StageRecord("memory_benchmark", MEASUREMENT, {"width": target.width, "height": target.height}, "-", "-")
            ):
                manifest["memory"]["benchmark"] = environment.benchmark_representations(target.width, target.height)

        if config.save_intermediates:
            with stage(
                StageRecord("save_working_source", PACKAGING, {"format": "OpenEXR float32"}, "float32", "float32")
            ):
                result = exporters.write_exr(
                    masters_dir / "intermediates" / f"{_stem(source)}_source_lin_rec709_f32.exr",
                    source.linear,
                    _exr_attributes(run_id, source, "working-space source (no resampling)"),
                    precision="float",
                    compression=config.exr_compression,
                )
                manifest["intermediates"] = [result.to_dict()]

        for method in config.baselines:
            # Registered before it runs, so a failure still records what this baseline completed.
            summary = manifest["baselines"][method] = {
                "description": baselines.METHODS[method].description,
                "classification": INTERPOLATION,
                "exports": {},
            }
            _run_baseline(
                method,
                summary,
                source,
                target,
                config,
                run_id,
                masters_dir,
                figures_dir,
                crops,
                source_encoded16,
                crop_tiles,
                crop_stats,
                stage,
                log,
            )
            gc.collect()

        if config.enable_super_resolution:
            _run_super_resolution(
                manifest,
                source,
                target,
                config,
                project_root,
                run_id,
                masters_dir,
                figures_dir,
                crops,
                source_encoded16,
                crop_tiles,
                crop_stats,
                stage,
                log,
            )

        with stage(
            StageRecord("crop_comparison", MEASUREMENT, {"crops": len(crops)}, "uint16 sRGB", "uint8 sRGB review PNG")
        ):
            panels = []
            for crop in crops:
                if not crop_tiles[crop.name]:
                    continue
                source_tile = cm.srgb_uint16_to_uint8(
                    source_encoded16[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
                )
                tiles = {m: cm.srgb_uint16_to_uint8(t) for m, t in crop_tiles[crop.name].items()}
                path = exporters.write_review_image(
                    figures_dir / f"crop_{crop.name}.png",
                    comparison.crop_panel(source_tile, tiles, difference=_panel_difference(tiles)),
                )
                panels.append({"crop": crop.name, "path": str(path), "methods": crop_stats[crop.name]})
            manifest["crop_panels"] = panels

        manifest["status"] = "complete"
    except StageError as error:
        raise fail(error) from error.__cause__
    except Exception as error:
        last = stages[-1].name if stages else "start"
        raise fail(StageError(f"unlogged step after {last}", error, None)) from error

    manifest_path = _finish(manifest, stages, monitor, started, paths, log)
    return RunResult(
        manifest["run_id"], manifest, manifest_path, paths["reports"] / "report.md", paths["masters"], paths["reports"]
    )


def _run_baseline(
    method,
    summary: dict,
    source,
    target: TargetResolution,
    config: ExperimentConfig,
    run_id,
    masters_dir,
    figures_dir,
    crops,
    source_encoded16,
    crop_tiles,
    crop_stats,
    stage,
    log,
) -> None:
    info = baselines.METHODS[method]
    size = f"{target.width}x{target.height}"

    with stage(
        StageRecord(
            f"baseline_{method}",
            INTERPOLATION,
            {
                "method": info.description,
                "width": target.width,
                "height": target.height,
                "scale_x": round(target.scale_x, 6),
                "scale_y": round(target.scale_y, 6),
                "space": cm.WORKING_SPACE,
            },
            "float32 linear",
            "float32 linear",
        )
    ):
        output = baselines.resample(source.linear, target.width, target.height, method)
        summary["working_pixel_sha256"] = ingestion.pixel_sha256(output)
        summary["statistics"] = quality_metrics.image_statistics(output)
    log(f"{method}: resampled to {size}")

    if config.runtime_preset == "research":
        with stage(StageRecord(f"reproducibility_{method}", MEASUREMENT, {"repeat": 1}, "float32 linear", "sha256")):
            del output
            gc.collect()
            output = baselines.resample(source.linear, target.width, target.height, method)
            repeat_hash = ingestion.pixel_sha256(output)
            summary["reproducibility"] = {
                "repeat_pixel_sha256": repeat_hash,
                "identical": repeat_hash == summary["working_pixel_sha256"],
            }

    buffer = [output]
    del output  # the helper frees the 50 MP buffer as soon as it is encoded
    _export_and_measure(
        method,
        buffer,
        summary,
        _provenance(run_id, source, method, target),
        _exr_attributes(run_id, source, f"{method} baseline"),
        source,
        target,
        config,
        masters_dir,
        figures_dir,
        crops,
        crop_tiles,
        crop_stats,
        stage,
    )


def _export_and_measure(
    method: str,
    buffer: list,
    summary: dict,
    provenance: dict,
    exr_attributes: dict,
    source,
    target: TargetResolution,
    config: ExperimentConfig,
    masters_dir,
    figures_dir,
    crops,
    crop_tiles,
    crop_stats,
    stage,
) -> None:
    """Exports, source-consistency metrics, crops and preview of one reconstruction (baseline or SR).

    ``buffer`` is a one-item list holding the float32 output; it is emptied so the 50 MP buffer is freed
    as soon as it is encoded. The caller must not keep its own reference.
    """
    size = f"{target.width}x{target.height}"
    output = buffer.pop()
    if config.export_exr:
        with stage(
            StageRecord(
                f"export_exr_{method}",
                PACKAGING,
                {
                    "precision": config.exr_precision,
                    "compression": config.exr_compression,
                },
                "float32 linear",
                f"{config.exr_precision} linear",
            )
        ):
            suffix = "half" if config.exr_precision == "half" else "f32"
            result = exporters.write_exr(
                masters_dir / f"{_stem(source)}_{method}_{size}_lin_rec709_{suffix}.exr",
                output,
                exr_attributes,
                precision=config.exr_precision,
                compression=config.exr_compression,
            )
            exporters.write_sidecar(result, provenance)
            summary["exports"]["exr"] = result.to_dict()

    with stage(
        StageRecord(
            f"encode_srgb16_{method}",
            PACKAGING,
            {"transfer": "sRGB", "clip": "[0, 1]"},
            "float32 linear",
            "uint16 sRGB",
        )
    ):
        encoded, clip_stats = cm.encode_srgb_uint16(output)
        summary["tiff_clipping"] = clip_stats.to_dict()
        del output
        gc.collect()

    if config.export_tiff16:
        with stage(
            StageRecord(
                f"export_tiff16_{method}",
                PACKAGING,
                {"compression": config.tiff_compression},
                "uint16 sRGB",
                "uint16 sRGB TIFF",
            )
        ):
            metadata = {**provenance, "software": f"synthetic_raw_lab {__version__}"}
            result = exporters.write_tiff16(
                masters_dir / f"{_stem(source)}_{method}_{size}_srgb16.tif",
                encoded,
                metadata,
                compression=config.tiff_compression,
            )
            exporters.write_sidecar(result, provenance)
            summary["exports"]["tiff16"] = result.to_dict()

    with stage(
        StageRecord(
            f"source_consistency_{method}",
            MEASUREMENT,
            {
                "reduction": baselines.DOWNSAMPLE_DESCRIPTION,
                "compared": "16-bit sRGB master vs source",
            },
            "uint16 sRGB",
            "metrics",
        )
    ):
        reduced = np.empty((source.height, source.width, 3), dtype=np.float32)
        for channel in range(3):
            plane = cm.srgb_uint16_to_linear(encoded[..., channel])[..., None]
            reduced[..., channel] = baselines.downsample_area(plane, source.width, source.height)[..., 0]
            del plane
        metrics, delta_e = quality_metrics.source_consistency(source.linear, reduced)
        summary["source_consistency"] = metrics
        report_builder.heatmap_figure(
            delta_e,
            figures_dir / f"delta_e_{method}.png",
            f"CIEDE2000, {method} reduced to source vs source",
            config.review_max_edge,
        )
        del reduced, delta_e

    for crop in crops:
        x0, y0, x1, y1 = crop.output_box((source.width, source.height), (target.width, target.height))
        tile = encoded[y0:y1, x0:x1].copy()
        crop_tiles[crop.name][method] = tile
        crop_stats[crop.name][method] = {
            "output_box": [x0, y0, x1, y1],
            "high_frequency_energy": quality_metrics.high_frequency_energy(tile.astype(np.float32) / 65535),
        }
    if config.export_review:
        report_builder.write_preview(encoded, figures_dir / f"{method}_preview.jpg", config.review_max_edge)
    del encoded
    gc.collect()


def _run_super_resolution(
    manifest: dict,
    source,
    target: TargetResolution,
    config: ExperimentConfig,
    project_root: Path,
    run_id,
    masters_dir,
    figures_dir,
    crops,
    source_encoded16,
    crop_tiles,
    crop_stats,
    stage,
    log,
) -> None:
    """Phase 2: one pretrained-SR master per configured mode, measured and exported like the baselines."""
    networks = sr_models.BACKENDS[config.sr_backend]
    record = manifest["super_resolution"] = {
        "backend": config.sr_backend,
        "scale_handling": "network at its native x4, then Lanczos-3 in linear light to the exact target size",
        "modes": {},
    }
    if target.scale_x <= 1 or target.scale_y <= 1:
        record["skipped"] = (
            f"target scale {target.scale_x:.3f} x {target.scale_y:.3f} is not an enlargement; super-resolution "
            "cannot add anything (use the standard preset for the full target)"
        )
        log(f"super-resolution skipped: {record['skipped']}")
        return
    with stage(StageRecord("super_resolution_setup", SETUP, {"backend": config.sr_backend}, "-", "-")):
        if not sr_models.available():
            raise RuntimeError("super-resolution needs PyTorch and spandrel: pip install 'synthetic-raw-50mp-lab[sr]'")
        weights_dir = (
            Path(config.sr_weights_dir) if config.sr_weights_dir else sr_models.default_weights_dir(project_root)
        )
    loaded: dict[str, sr_models.TorchModel] = {}
    for index, mode in enumerate(config.sr_modes):
        preset = sr.PRESETS[mode]
        name = f"sr_{mode}"
        weights = networks[preset.network]
        consistency = preset.consistency_strength if config.preserve_source_structure else 0.0
        summary = record["modes"][mode] = {
            "preset": preset.to_dict(),
            "classification": INFERENCE,
            "weights": weights,
            "strength": config.sr_strength,
            "consistency_strength": consistency,
            "exports": {},
        }
        if weights not in loaded:
            with stage(
                StageRecord(
                    f"load_model_{weights}",
                    SETUP,
                    {"weights": weights, "device": config.sr_device, "precision": config.sr_precision},
                    "weights file",
                    "network",
                )
            ):
                loaded[weights] = sr_models.load_model(
                    weights,
                    weights_dir,
                    device=config.sr_device,
                    precision=config.sr_precision,
                    seed=config.random_seed,
                    log=log,
                )
        model = loaded[weights]
        tile = config.sr_tile_size
        tile_basis = "configured"
        if not tile:
            tile, tile_basis = sr_models.auto_tile_size(
                model.device, model.precision, model.architecture, source_width=source.width, scale=model.scale
            )
        tile = max(tile, 2 * config.sr_tile_overlap)
        resample = target.scale_x / model.scale
        with stage(
            StageRecord(
                f"super_resolution_{mode}",
                INFERENCE,
                {
                    "weights": weights,
                    "native_scale": model.scale,
                    "then": f"Lanczos-3 x{resample:.4f} ({'downscale' if resample < 1 else 'upscale'})",
                    "tile": f"{tile} ({tile_basis})",
                    "overlap": config.sr_tile_overlap,
                    "strength": config.sr_strength,
                    "device": model.device,
                    "precision": model.precision,
                },
                "float32 linear (sRGB-encoded for the network)",
                "float32 linear",
            )
        ) as record_stage:
            output, tiling = sr.super_resolve(
                source.linear,
                target.width,
                target.height,
                model,
                tile_size=tile,
                tile_overlap=config.sr_tile_overlap,
                detail_strength=config.sr_strength,
                log=lambda _: None,
            )
            summary["tiling"] = tiling.to_dict()
            summary["tiling"]["tile_basis"] = tile_basis
            summary["final_resample"] = {"filter": "Lanczos-3, linear light", "factor": resample}
            record_stage.notes = [f["message"] for f in tiling.fallbacks]
            model = loaded[weights] = _current_model(model, tiling)
            summary["model"] = model.describe()
            manifest["models"] = [m["model"] for m in record["modes"].values() if "model" in m]
        log(f"{name}: {tiling.tiles} tiles of {tile} px on {tiling.final_device}, {tiling.inference_seconds:.0f} s")

        if consistency > 0:
            with stage(
                StageRecord(
                    f"source_projection_{mode}",
                    INTERPOLATION,
                    {"strength": consistency, "reduction": "area average", "upsampling": "Keys bicubic"},
                    "float32 linear",
                    "float32 linear",
                )
            ):
                summary["source_projection"] = sr.project_to_source(output, source.linear, consistency)

        with stage(StageRecord(f"seam_check_{mode}", MEASUREMENT, {"boundaries": len(tiling.boundaries)}, "-", "-")):
            seams = sr.seam_scores(output, tiling.boundaries, source.linear)
            disagreement = summary["tiling"]["overlap_disagreement"]
            seams["passed"] = (seams["max_score"] or 0) <= SEAM_SCORE_LIMIT and (
                disagreement["blend_error_p99_abs_max"] or 0
            ) <= SEAM_BLEND_ERROR_LIMIT
            seams["limits"] = {"max_score": SEAM_SCORE_LIMIT, "blend_error_p99_abs_8bit": SEAM_BLEND_ERROR_LIMIT}
            summary["seams"] = seams

        with stage(StageRecord(f"interpolation_difference_{mode}", MEASUREMENT, {"reference": "Lanczos-3"}, "-", "-")):
            difference, summary["interpolation_difference"] = sr.interpolation_difference(output, source.linear)
            report_builder.heatmap_figure(
                _reduce_map(difference, config.review_max_edge),
                figures_dir / f"sr_difference_{mode}.png",
                f"|{name} - Lanczos|, sRGB luma (synthesised detail and drift)",
                config.review_max_edge,
                label="8-bit code values",
            )
            del difference
            summary["working_pixel_sha256"] = ingestion.pixel_sha256(output)
            summary["statistics"] = quality_metrics.image_statistics(output)

        buffer = [output]
        del output
        try:
            _export_and_measure(
                name,
                buffer,
                summary,
                _sr_provenance(run_id, source, target, summary),
                _exr_attributes(run_id, source, f"{name} pretrained super-resolution", notice=SR_NOTICE),
                source,
                target,
                config,
                masters_dir,
                figures_dir,
                crops,
                crop_tiles,
                crop_stats,
                stage,
            )
        finally:
            if summary["exports"]:
                manifest["phase"], manifest["notice"] = PHASE_SR, RUN_NOTICE_SR
                manifest["random_seeds"]["used_by"] = SEEDS_SR
        remaining = {networks[sr.PRESETS[m].network] for m in config.sr_modes[index + 1 :]}
        for unused in set(loaded) - remaining:
            loaded.pop(unused).release_cache()
        gc.collect()


def _current_model(model, tiling):
    """After a device fallback the tiler holds a CPU copy; keep using it for later modes."""
    if tiling.final_device and tiling.final_device != model.device:
        return model.fallback() or model
    return model


def _reduce_map(values: np.ndarray, max_edge: int) -> np.ndarray:
    """Area-reduce a float16 full-resolution map to review size in row bands (no full float32 copy)."""
    height, width = values.shape
    scale = min(1.0, max_edge / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rows = np.empty((height, size[0]), dtype=np.float32)
    for start in range(0, height, 512):
        block = values[start : start + 512].astype(np.float32)[..., None]
        rows[start : start + 512] = baselines.downsample_area(block, size[0], block.shape[0])[..., 0]
    return baselines.downsample_area(rows[..., None], size[0], size[1])[..., 0]


def _panel_difference(tiles: dict) -> tuple[str, str] | None:
    """Crop panels show |first SR mode - Lanczos| when SR ran (PRD 10.5), else the two baselines."""
    reference = "lanczos" if "lanczos" in tiles else next(iter(tiles))
    learned = [name for name in tiles if name.startswith("sr_")]
    return (reference, learned[0]) if learned else None


@contextmanager
def _stage(record: StageRecord, stages: list[StageRecord], monitor: environment.MemoryMonitor, log):
    start = time.perf_counter()
    try:
        with monitor.stage(record.name):
            yield record
    except Exception as error:
        record.error = str(error) if isinstance(error, StageError) else f"{type(error).__name__}: {error}"
        log(f"stage {record.name} FAILED: {record.error}")
        if isinstance(error, StageError):
            raise
        raise StageError(record.name, error, None) from error
    finally:
        record.seconds = time.perf_counter() - start
        record.rss_peak_mb = monitor.stages[-1].rss_peak_mb
        available = psutil.virtual_memory().available / 1e9
        if available < LOW_MEMORY_HEADROOM_GB:
            record.memory_warnings.append(f"only {available:.2f} GB RAM available after this stage")
        stages.append(record)
    log(f"stage {record.name}: {record.seconds:.2f} s, peak RSS {record.rss_peak_mb:.0f} MB")


def _finish(manifest: dict, stages: list[StageRecord], monitor, started: float, paths: dict, log) -> Path | None:
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
    storage = sum(f.stat().st_size for d in (paths["masters"], paths["reports"]) for f in d.rglob("*") if f.is_file())
    manifest["storage_used_bytes"] = storage
    manifest["output_checksums"] = _checksums(manifest)
    manifest_path = paths["reports"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    report_builder.write_report(manifest, paths["reports"] / "report.md")
    log(f"report: {paths['reports'] / 'report.md'}")
    return manifest_path


def _checksums(manifest: dict) -> dict:
    sums = {}
    for item in manifest.get("intermediates", []):
        sums[Path(item["path"]).name] = item["file_sha256"]
    reconstructions = list(manifest.get("baselines", {}).values())
    reconstructions += list(manifest.get("super_resolution", {}).get("modes", {}).values())
    for summary in reconstructions:
        for export in summary.get("exports", {}).values():
            sums[Path(export["path"]).name] = export["file_sha256"]
    return sums


def _provenance(run_id: str, source, method: str, target: TargetResolution) -> dict:
    return {
        "notice": NOTICE,
        "run_id": run_id,
        "source_name": source.original_path.name,
        "source_sha256": source.sha256,
        "source_dimensions": [source.width, source.height],
        "output_dimensions": [target.width, target.height],
        "method": baselines.METHODS[method].description,
        "stage_classification": [PRESERVATION, INTERPOLATION, PACKAGING],
        "information": {
            "preserved": "source structure up to the source sampling limit",
            "interpolated": "every output sample is an interpolated estimate",
            "inferred": "none (no learned models)",
            "simulated": "none",
        },
        "package_version": __version__,
    }


def _sr_provenance(run_id: str, source, target: TargetResolution, summary: dict) -> dict:
    weights = sr_models.WEIGHTS[summary["weights"]]
    return {
        "notice": SR_NOTICE,
        "run_id": run_id,
        "source_name": source.original_path.name,
        "source_sha256": source.sha256,
        "source_dimensions": [source.width, source.height],
        "output_dimensions": [target.width, target.height],
        "method": f"{summary['preset']['label']}: {weights.name} ({weights.architecture}), x4 then Lanczos-3",
        "model_weights_sha256": weights.sha256,
        "model_license": weights.license,
        "stage_classification": [PRESERVATION, INFERENCE, INTERPOLATION, PACKAGING],
        "information": {
            "preserved": "source structure up to the source sampling limit; low-frequency tone and colour are "
            f"pulled back to the source (correction strength {summary['consistency_strength']})",
            "interpolated": "the final resample from the network's x4 grid to the target, the low-frequency "
            f"correction and a {1 - summary['strength']:.0%} Lanczos share",
            "inferred": f"detail above the source sampling limit, synthesised by {weights.name} from learned "
            "priors, not recovered",
            "simulated": "none",
        },
        "package_version": __version__,
    }


def _exr_attributes(run_id: str, source, content: str, *, notice: str = NOTICE) -> dict[str, str]:
    return {
        "comments": notice,
        "colorSpaceDescription": exporters.EXR_COLOR_DESCRIPTION,
        "channelNaming": "R, G, B: linear-light Rec.709 primaries",
        "syntheticRawLabContent": content,
        "syntheticRawLabRunId": run_id,
        "syntheticRawLabSourceSha256": source.sha256,
        "syntheticRawLabVersion": __version__,
    }


def _stem(source) -> str:
    stem = source.original_path.stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return safe or hashlib.sha256(stem.encode()).hexdigest()[:8]


def _without_histograms(validation: dict) -> dict:
    return {k: v for k, v in validation.items() if k != "histograms"}
