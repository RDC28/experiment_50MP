"""Phase 3 nighttime run (notebook 02): ingest, segment, compare, process, export, measure, report.

The run builds on the Phase 0/1 pipeline (stage logging, ingestion, target
resolution, exporters) and adds the Layer 5B night processing of
:mod:`night_processing`:

1. Ingest and validate the source; compute the 50 MP (or reduced) target.
2. Segment the PRD 8.3 luminance regions on the source, apply manual
   overrides and write the region-mask figure.
3. Build every PRD 19.3 comparison image one at a time (bicubic and Lanczos
   baselines, then the conservative and balanced SR slots, which run the
   Phase 2 pretrained-SR backend through ``pipeline.run_sr_mode``: one network
   loaded at a time and released after its slot), measure the PRD 23.5 night
   metrics and source consistency on each, keep the crops and release it.
   The slot that feeds Layer 5B (the configured SR mode, or Lanczos when SR is
   disabled or cannot enlarge) runs last and is kept, so no mode runs twice.
4. Apply Layer 5B in place on that kept image, so a 50 MP run holds one
   full-size float buffer.
5. Export the final master as 16-bit TIFF and OpenEXR (half by default),
   validate both, and write the crop panels, banding diagnostics, halo
   comparisons, failure-rule fallback map, manifest and report.

If a stage fails, outputs already written stay on disk, and the manifest and
report name the failing stage (PRD 22.3).
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import (
    __version__,
    baselines,
    comparison,
    environment,
    exporters,
    ingestion,
    quality_metrics,
    report_builder,
)
from . import color_management as cm
from . import night_processing as night
from .pipeline import (
    MEASUREMENT,
    PACKAGING,
    PRESERVATION,
    SEEDS_SR,
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
from .report_builder import (
    _bytes,
    _dims,
    _models_text,
    _num,
    _short_params,
    _sr_section,
    _table,
    _validation_rows,
    _version,
)
from .resolution import compute_target_resolution

SCHEMA = "synthetic_raw_lab.phase3_night.manifest/1"
NOTICE = (
    "Synthetic camera-grade night master reconstructed from an AI-generated source by classical interpolation "
    "and night-specific tone, colour and noise processing. It is not a camera capture or camera RAW file: "
    "shadow lift redistributes existing tones, the highlight shoulder rolls off clipped values without "
    "recovering radiance, and dither, grain and bloom are simulated."
)
NOTICE_SR = (
    "Synthetic camera-grade night master reconstructed from an AI-generated source by pretrained "
    "super-resolution and night-specific tone, colour and noise processing. Detail above the source sampling "
    "limit was synthesised by a learned model, not recovered. It is not a camera capture or camera RAW file: "
    "shadow lift redistributes existing tones, the highlight shoulder rolls off clipped values without "
    "recovering radiance, and dither, grain and bloom are simulated."
)
NIGHT_PROCESSING = "Layer 5B night processing (per-step classes in the night stage table)"
FINAL = "final_night"

# Crop categories (PRD 26.2 night crops) and the regions each one looks for.
CROP_CATEGORIES = {
    "highlight": ("clipped_highlight", "bright_light"),
    "shadow": ("near_black", "low_luminance_textured"),
    "reflection": ("specular_reflection",),
    "gradient": ("sky_atmosphere", "smooth_gradient"),
}
_PANEL_LABELS = {FINAL: "final night"}
# Crops whose panel adds a display-gain row (stops of linear gain): near-black tones are unreadable at 100%.
DISPLAY_GAIN_EV = {"shadow": 3.0}
_GUTTER = 8
_DIFF_GAIN = 8
_PANEL_BACKGROUND = 24


@dataclass
class NightRunResult:
    run_id: str
    manifest: dict
    manifest_path: Path
    report_path: Path
    masters_dir: Path
    reports_dir: Path
    regions: night.RegionMap


@dataclass(frozen=True)
class Slot:
    """One PRD 19.3 comparison image: a classical baseline or an SR slot."""

    key: str
    upscaler: night.Upscaler

    def panel_label(self, backend: str = "") -> str:
        return _label(self.key, backend)


def comparison_slots(config: night.NightConfig, sr_runner: night.SRRunner | None = None) -> list[Slot]:
    """The bicubic/Lanczos baselines and, given the Phase 2 runner, the conservative and balanced SR slots.

    Report order; without a runner (SR disabled or skipped) there are no SR slots.
    """
    slots = [Slot(m, night.BaselineUpscaler(m, m)) for m in config.experiment.baselines]
    if sr_runner is not None:
        slots += [
            Slot(f"{mode}_sr", night.PretrainedUpscaler(f"{mode}_sr", mode, sr_runner))
            for mode in night.SR_COMPARISON_MODES
        ]
    return slots


def night_input(config: night.NightConfig, sr_skipped: str = "") -> tuple[str, str]:
    """Key of the slot whose output Layer 5B processes, and why it is not SR (empty when it is).

    The configured SR mode; the Lanczos baseline when super-resolution is disabled or ``sr_skipped``
    gives the reason it cannot run.
    """
    sr = config.night.super_resolution
    if not sr.enabled:
        return "lanczos", "super-resolution disabled, no SR applied"
    if sr_skipped:
        return "lanczos", f"super-resolution skipped, no SR applied: {sr_skipped}"
    return f"{sr.mode}_sr", ""


def execution_order(slots: list[Slot], input_key: str, input_note: str = "") -> list[Slot]:
    """Slots in the order they run: the night input last, so its image is kept for Layer 5B and never rebuilt.

    The input slot is added (as a Lanczos baseline carrying ``input_note``) when it is not a comparison slot.
    """
    by_key = {slot.key: slot for slot in slots}
    if input_key in by_key and not input_note:
        last = by_key[input_key]
    else:
        last = Slot(input_key, night.BaselineUpscaler(input_key, input_key, input_note))
    return [slot for slot in slots if slot.key != input_key] + [last]


def select_night_crops(
    regions: night.RegionMap, size: int, manual: Sequence = (), categories: dict | None = None
) -> list[comparison.Crop]:
    """One crop per night category (PRD 26.2), placed where that category's regions are densest.

    The highlight crop is centred on :func:`light_centre`, so it doubles as the light-source
    halo comparison. Categories absent from the scene are skipped. Manual crops
    (``[x, y, width, height]`` in source pixels) replace the automatic choice.
    """
    height, width = regions.shape
    if manual:
        crops = []
        for index, (x, y, w, h) in enumerate(manual, start=1):
            if x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError(f"manual crop {[x, y, w, h]} exceeds the {width}x{height} source")
            crops.append(comparison.Crop(f"manual_{index}", int(x), int(y), int(w), int(h)))
        return crops
    size = min(size, width, height)
    crops = []
    for name, region_names in (categories or CROP_CATEGORIES).items():
        mask = regions.mask(*region_names)
        if not mask.any():
            continue
        if name == "highlight":
            x, y = light_centre(regions)
            x0 = int(np.clip(x - size // 2, 0, width - size))
            y0 = int(np.clip(y - size // 2, 0, height - size))
        else:
            integral = cv2.integral(mask.astype(np.float64))
            stride = max(1, size // 4)
            best = (-1.0, 0, 0)
            for y in sorted(set(range(0, height - size + 1, stride)) | {height - size}):
                for x in sorted(set(range(0, width - size + 1, stride)) | {width - size}):
                    count = (
                        integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
                    )
                    if count > best[0]:
                        best = (count, y, x)
            _, y0, x0 = best
        crops.append(comparison.Crop(name, int(x0), int(y0), size, size))
    return crops


def light_centre(regions: night.RegionMap) -> tuple[int, int]:
    """Source (x, y) of the light with the widest clipped core, e.g. a street lamp rather than a thin neon tube.

    The deepest point of the clipped mask (largest distance to its edge); bright lights when nothing
    clipped; the image centre when the scene has no lights.
    """
    for names in (("clipped_highlight",), ("clipped_highlight", "bright_light")):
        mask = regions.mask(*names).astype(np.uint8)
        if mask.any():
            depth = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
            y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
            return int(x), int(y)
    height, width = regions.shape
    return width // 2, height // 2


def run_night(
    source_path: str | Path,
    config: night.NightConfig,
    project_root: str | Path,
    *,
    region_overrides: Sequence[dict] = (),
    notebook_version: str | None = None,
    run_id: str | None = None,
    log: Callable[[str], None] = print,
) -> NightRunResult:
    """Run the Phase 3 night pipeline on one source image.

    ``region_overrides`` are manual PRD 8.3 region corrections, each
    ``{"region": name, "boxes": [[x, y, w, h], ...], "note": text}`` in source pixels.
    """
    experiment, params = config.experiment, config.night
    project_root = Path(project_root).resolve()
    inputs_root, output_root, reports_root = resolve_roots(experiment, project_root)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)

    runtime = environment.inspect_runtime(output_root)
    monitor = environment.MemoryMonitor().start()
    stages: list[StageRecord] = []
    manifest: dict = {
        "schema": SCHEMA,
        "phase": "Phase 3 (specialised notebooks): Nighttime Scenery",
        "notice": NOTICE,
        "started_at": started_at.isoformat(timespec="seconds"),
        "package": {
            "name": "synthetic_raw_lab",
            "version": __version__,
            "git": environment.git_revision(Path(__file__).parent) or environment.git_revision(project_root),
        },
        "notebook_version": notebook_version,
        "models": [],
        "random_seeds": {"random_seed": experiment.random_seed, "used_by": "simulated dither, grain and bloom"},
        "config": config.to_dict(),
        "runtime": runtime,
        "stages": [],
        "comparisons": {},
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
        (reports_dir / "input_validation.json").write_text(json.dumps(source.validation, indent=2))

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
        manifest["target_resolution"] = target.to_dict()
        manifest["memory"] = {"estimates": environment.estimate_memory(*size)}
        manifest["storage"] = environment.estimate_storage(
            *size,
            1,
            tiff=experiment.export_tiff16,
            exr=experiment.export_exr,
            exr_bytes_per_sample=2 if experiment.exr_precision == "half" else 4,
            source_pixels=source.width * source.height,
            save_intermediates=experiment.save_intermediates,
        )
        log(
            f"source {source.width}x{source.height} -> target {target.width}x{target.height} "
            f"({target.megapixels:.3f} MP)"
        )

        with stage(
            StageRecord(
                "region_segmentation",
                "Measurement (labels only, no image data)",
                {"overrides": len(region_overrides)},
                "float32 linear",
                "uint8 region labels",
            )
        ):
            regions = night.segment_regions(source.linear)
            for override in region_overrides:
                regions = regions.override(override["region"], override["boxes"], override.get("note", ""))
            manifest["regions"] = regions.to_dict()
            manifest["figures"] = {"regions": str(region_figure(regions, source.linear, figures_dir / "regions.png"))}

        if experiment.save_intermediates:
            with stage(
                StageRecord("save_working_source", PACKAGING, {"format": "OpenEXR float32"}, "float32", "float32")
            ):
                result = exporters.write_exr(
                    masters_dir / "intermediates" / f"{_stem(source)}_source_lin_rec709_f32.exr",
                    source.linear,
                    _exr_attributes(run_id, source, "working-space source (no resampling)"),
                    precision="float",
                    compression=experiment.exr_compression,
                )
                manifest["intermediates"] = [result.to_dict()]

        crops = select_night_crops(regions, experiment.crop_size, experiment.crops)
        manifest["crops"] = [c.to_dict() for c in crops]
        boxes = {c.name: c.output_box((source.width, source.height), size) for c in crops}
        tiles: dict[str, dict[str, np.ndarray]] = {c.name: {} for c in crops}
        profiles = _ProfileLines(boxes, source, size, light_centre(regions))

        with stage(StageRecord("night_metrics_source", MEASUREMENT, {}, "float32 linear", "metrics")):
            manifest["source_night_metrics"] = night.night_metrics(source.linear, regions, source.linear)
        source_encoded16, _ = cm.encode_srgb_uint16(source.linear)
        if experiment.export_review:
            report_builder.write_preview(
                source_encoded16, figures_dir / "source_preview.jpg", experiment.review_max_edge
            )

        sr_runner = None
        if params.super_resolution.enabled:
            sr_record = manifest["super_resolution"] = sr_run_record(experiment, target)
            if "skipped" in sr_record:
                log(f"super-resolution skipped: {sr_record['skipped']}")
            else:
                sr_runner = _sr_runner(experiment, target, project_root, manifest, figures_dir, stage, log)
        else:
            manifest["super_resolution"] = {"enabled": False, "modes": {}}
        input_key, input_note = night_input(config, manifest["super_resolution"].get("skipped", ""))
        slots = comparison_slots(config, sr_runner)
        for slot in slots:  # report order, whatever order the slots run in
            manifest["comparisons"][slot.key] = {}
        for slot in execution_order(slots, input_key, input_note):
            summary = manifest["comparisons"].setdefault(slot.key, {})
            learned = isinstance(slot.upscaler, night.PretrainedUpscaler)
            if learned:  # Phase 2 logs its own load, inference, projection and measurement stages
                upscaled = slot.upscaler(source.linear, *size)
            with stage(
                StageRecord(
                    f"compare_{slot.key}", MEASUREMENT, {"size": f"{size[0]}x{size[1]}"}, "float32 linear", "metrics"
                )
            ):
                if not learned:
                    upscaled = slot.upscaler(source.linear, *size)
                summary.update(upscaled.describe())
                # Phase 2 already measured the SR images; baselines are measured here.
                measured = manifest["super_resolution"]["modes"][slot.upscaler.mode] if learned else {}
                summary["working_pixel_sha256"] = measured.get("working_pixel_sha256") or ingestion.pixel_sha256(
                    upscaled.image
                )
                summary["statistics"] = measured.get("statistics") or quality_metrics.image_statistics(upscaled.image)
                summary["night_metrics"] = night.night_metrics(upscaled.image, regions, source.linear)
                _measure(summary, slot.key, upscaled.image, source, boxes, tiles, profiles)
                if experiment.export_review and slot.key == "lanczos":
                    encoded, _ = cm.encode_srgb_uint16(upscaled.image)
                    report_builder.write_preview(
                        encoded, figures_dir / "lanczos_preview.jpg", experiment.review_max_edge
                    )
                    del encoded
            log(f"{slot.key}: {summary['label']}")
            if slot.key != input_key:
                del upscaled
                gc.collect()
        image = upscaled.image  # the night input, built once in the comparison pass

        with stage(
            StageRecord("night_processing", NIGHT_PROCESSING, params.to_dict(), "float32 linear", "float32 linear")
        ) as record:
            result = night.process_night(
                image, regions, source.linear, params, seed=experiment.random_seed, out=image, upscale=upscaled
            )
            manifest["night"] = {
                "input": {"slot": input_key, **upscaled.describe()},
                "stages": result.stages,
                "fallbacks": result.fallbacks,
                "fallback_fraction": result.fallback_fraction,
                "metrics_before": result.metrics_before,
                "metrics_after": result.metrics_after,
                "comparison": result.comparison,
                "information": result.information,
            }
            record.notes = [f"{len(result.fallbacks)} failure-rule fallback blocks ({result.fallback_fraction:.3%})"]
            fallback_path = fallback_figure(
                night.fallback_map(result, *size, max_edge=experiment.review_max_edge),
                source.linear,
                result.fallbacks,
                result.fallback_fraction,
                figures_dir / "fallback_map.png",
            )
            manifest["figures"]["fallbacks"] = str(fallback_path)
            del result, upscaled  # the upscale was processed in place; image is the only full-size buffer
        log(f"night processing: {len(manifest['night']['fallbacks'])} fallback blocks")

        final = manifest["final"] = {"exports": {}}
        final["working_pixel_sha256"] = ingestion.pixel_sha256(image)
        final["statistics"] = quality_metrics.image_statistics(image)
        provenance = _provenance(run_id, source, size, manifest["night"])
        if manifest["night"]["input"]["provenance"] == night.SYNTHESIZED:
            manifest["notice"] = NOTICE_SR
            manifest["random_seeds"]["used_by"] = f"{manifest['random_seeds']['used_by']}; {SEEDS_SR}"
        label = f"{target.width}x{target.height}"
        if experiment.export_exr:
            with stage(
                StageRecord(
                    "export_exr_night",
                    PACKAGING,
                    {"precision": experiment.exr_precision, "compression": experiment.exr_compression},
                    "float32 linear",
                    f"{experiment.exr_precision} linear",
                )
            ):
                suffix = "half" if experiment.exr_precision == "half" else "f32"
                exported = exporters.write_exr(
                    masters_dir / f"{_stem(source)}_night_{label}_lin_rec709_{suffix}.exr",
                    image,
                    _exr_attributes(run_id, source, "Layer 5B night master"),
                    precision=experiment.exr_precision,
                    compression=experiment.exr_compression,
                )
                exporters.write_sidecar(exported, provenance)
                final["exports"]["exr"] = exported.to_dict()

        with stage(StageRecord("encode_srgb16_night", PACKAGING, {"transfer": "sRGB"}, "float32", "uint16 sRGB")):
            encoded, clip_stats = cm.encode_srgb_uint16(image)
            final["tiff_clipping"] = clip_stats.to_dict()
            _measure(final, FINAL, image, source, boxes, tiles, profiles, measure_consistency=False)
            del image
            gc.collect()

        if experiment.export_tiff16:
            with stage(
                StageRecord(
                    "export_tiff16_night",
                    PACKAGING,
                    {"compression": experiment.tiff_compression},
                    "uint16 sRGB",
                    "uint16 sRGB TIFF",
                )
            ):
                exported = exporters.write_tiff16(
                    masters_dir / f"{_stem(source)}_night_{label}_srgb16.tif",
                    encoded,
                    {**provenance, "software": f"synthetic_raw_lab {__version__}"},
                    compression=experiment.tiff_compression,
                )
                exporters.write_sidecar(exported, provenance)
                final["exports"]["tiff16"] = exported.to_dict()

        with stage(
            StageRecord(
                "source_consistency_night",
                MEASUREMENT,
                {"reduction": baselines.DOWNSAMPLE_DESCRIPTION, "compared": "16-bit sRGB night master vs source"},
                "uint16 sRGB",
                "metrics",
            )
        ):
            metrics, delta_e = quality_metrics.source_consistency(source.linear, _reduce_encoded(encoded, source))
            final["source_consistency"] = metrics
            report_builder.heatmap_figure(
                delta_e,
                figures_dir / "delta_e_night.png",
                "CIEDE2000, night master reduced to source vs source",
                experiment.review_max_edge,
            )
            if experiment.export_review:
                report_builder.write_preview(encoded, figures_dir / "night_preview.jpg", experiment.review_max_edge)
            del encoded, delta_e

        with stage(
            StageRecord("night_comparisons", MEASUREMENT, {"crops": len(crops)}, "uint16 sRGB", "review PNG figures")
        ):
            slot_order = [*manifest["comparisons"], FINAL]
            backend = _backend(manifest)
            panels = []
            for crop in crops:
                source_tile = cm.srgb_uint16_to_uint8(
                    source_encoded16[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
                )
                labelled = {_label(k, backend): tiles[crop.name][k] for k in slot_order if k in tiles[crop.name]}
                gain = DISPLAY_GAIN_EV.get(crop.name)
                path = exporters.write_review_image(
                    figures_dir / f"crop_{crop.name}.png",
                    crop_panel(
                        source_tile, labelled, (_label(input_key, backend), _label(FINAL)), display_gain_ev=gain
                    ),
                )
                panels.append({"crop": crop.name, "path": str(path), "box": list(boxes[crop.name])})
                if gain:
                    panels[-1]["display_gain_ev"] = gain
            manifest["crop_panels"] = panels
            if "gradient" in boxes:
                manifest["figures"]["banding"] = str(
                    banding_figure(profiles, input_key, figures_dir / "banding_diagnostics.png", backend)
                )
            if "highlight" in boxes:
                manifest["figures"]["halo"] = str(
                    halo_figure(profiles, slot_order, figures_dir / "halo_profile.png", backend)
                )

        manifest["status"] = "complete"
    except StageError as error:
        raise fail(error) from error.__cause__
    except Exception as error:
        last = stages[-1].name if stages else "start"
        raise fail(StageError(f"unlogged step after {last}", error, None)) from error

    manifest_path = _finish(manifest, stages, monitor, started, paths, log)
    return NightRunResult(
        manifest["run_id"],
        manifest,
        manifest_path,
        paths["reports"] / "report.md",
        paths["masters"],
        paths["reports"],
        regions,
    )


def _sr_runner(experiment, target, project_root: Path, manifest: dict, figures_dir: Path, stage, log) -> night.SRRunner:
    """Phase 2's per-mode SR path bound to this run: one network loaded at a time, released after its slot.

    Fails the ``super_resolution_setup`` stage with Phase 2's install hint when PyTorch or spandrel is missing.
    """
    with stage(StageRecord("super_resolution_setup", SETUP, {"backend": experiment.sr_backend}, "-", "-")):
        weights_dir = sr_setup(experiment, project_root)
    loaded: dict = {}

    def run(mode: str, linear: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
        if (width, height) != (target.width, target.height):
            raise ValueError(f"SR slots render the run target {target.width}x{target.height}, not {width}x{height}")
        try:
            image = run_sr_mode(
                mode, linear, target, experiment, weights_dir, loaded, manifest, figures_dir, stage, log
            )
        finally:
            release_sr_models(loaded, experiment, ())
            gc.collect()
        return image, manifest["super_resolution"]["modes"][mode]

    return run


def _label(key: str, backend: str = "") -> str:
    """Short label of a slot for panels, figures and tables; SR slots name their mode and backend."""
    if key.endswith("_sr"):
        return f"{key.removesuffix('_sr')} SR" + (f" ({backend})" if backend else "")
    return _PANEL_LABELS.get(key, key)


def _backend(manifest: dict) -> str:
    """The SR backend when SR slots ran in this run, else empty."""
    record = manifest.get("super_resolution", {})
    return record.get("backend", "") if record.get("modes") else ""


def _measure(summary: dict, key, image, source, boxes, tiles, profiles, *, measure_consistency: bool = True) -> None:
    """Crops, profile lines and (for comparison slots) source consistency of one full-size image."""
    for name, (x0, y0, x1, y1) in boxes.items():
        tile, _ = cm.encode_srgb_uint16(image[y0:y1, x0:x1])
        tiles[name][key] = cm.srgb_uint16_to_uint8(tile)
    profiles.add(key, image)
    if measure_consistency:
        reduced = np.empty((source.height, source.width, 3), dtype=np.float32)
        for channel in range(3):
            plane = np.clip(image[..., channel], 0, 1)[..., None]
            reduced[..., channel] = baselines.downsample_area(plane, source.width, source.height)[..., 0]
            del plane
        summary["source_consistency"], _ = quality_metrics.source_consistency(source.linear, reduced)


def _reduce_encoded(encoded: np.ndarray, source) -> np.ndarray:
    reduced = np.empty((source.height, source.width, 3), dtype=np.float32)
    for channel in range(3):
        plane = cm.srgb_uint16_to_linear(encoded[..., channel])[..., None]
        reduced[..., channel] = baselines.downsample_area(plane, source.width, source.height)[..., 0]
        del plane
    return reduced


class _ProfileLines:
    """Luma lines and banding maps kept from every full-size image for the diagnostics figures.

    The gradient line runs vertically through the centre of the gradient crop and the halo
    line horizontally through ``light`` (source x, y) across the highlight crop; both in 8-bit codes.
    """

    def __init__(self, boxes: dict, source, size: tuple[int, int], light: tuple[int, int]):
        self.boxes = boxes
        self.size = size
        self.scale = size[0] / source.width
        self.lines: dict[str, dict[str, np.ndarray]] = {}
        self.coords: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.banding: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.halo_rows: dict | None = None
        sx, sy = source.width / size[0], source.height / size[1]
        for name, axis in (("gradient", 0), ("highlight", 1)):
            if name not in boxes:
                continue
            x0, y0, x1, y1 = boxes[name]
            if axis == 0:
                xs = np.full(y1 - y0, (x0 + x1) // 2)
                ys = np.arange(y0, y1)
            else:
                xs = np.arange(x0, x1)
                row = int((light[1] + 0.5) / sy)  # output row through the centre of the light's source pixel
                ys = np.full(x1 - x0, min(max(row, y0), y1 - 1))
            src_x = np.minimum(((xs + 0.5) * sx).astype(int), source.width - 1)
            src_y = np.minimum(((ys + 0.5) * sy).astype(int), source.height - 1)
            self.lines[name] = {"source": self._luma(source.linear[src_y, src_x])}
            self.coords[name] = (ys, xs)
            if axis == 1:
                # At a non-integer scale the output row lies between two source rows, and every interpolating
                # upscale blends in the other one (e.g. the lit edge of a window just above the lamp centre),
                # so the figure also shows that row; the nearest sample alone cannot explain such plateaus.
                y = (ys[0] + 0.5) * sy - 0.5
                nearest = int(src_y[0])
                adjacent = int(np.clip(nearest + (1 if y > nearest else -1), 0, source.height - 1))
                self.lines[name]["source_adjacent"] = self._luma(source.linear[adjacent, src_x])
                self.halo_rows = {"output_row": int(ys[0]), "source_y": y, "nearest": nearest, "adjacent": adjacent}

    @staticmethod
    def _luma(pixels: np.ndarray) -> np.ndarray:
        return 255 * (cm.linear_to_srgb(np.clip(pixels, 0, 1)) @ quality_metrics.LUMA_WEIGHTS)

    def add(self, key: str, image: np.ndarray) -> None:
        for name, (ys, xs) in self.coords.items():
            self.lines[name][key] = self._luma(image[ys, xs])
        if "gradient" in self.boxes:
            # Measured with a margin, so the crop edges have the same context as the full-image metric.
            x0, y0, x1, y1 = self.boxes["gradient"]
            m = max(4, round(8 * self.scale)) + 2
            a, b = max(0, y0 - m), min(self.size[1], y1 + m)
            c, d = max(0, x0 - m), min(self.size[0], x1 + m)
            flat, ramp = night.banding_map(np.ascontiguousarray(image[a:b, c:d]), self.scale)
            self.banding[key] = (flat[y0 - a : y1 - a, x0 - c : x1 - c], ramp[y0 - a : y1 - a, x0 - c : x1 - c])


def crop_panel(
    source_tile: np.ndarray,
    tiles: dict[str, np.ndarray],
    diff: tuple[str, str],
    *,
    display_gain_ev: float | None = None,
) -> np.ndarray:
    """uint8 crop panel: source (nearest), each image, and the amplified difference of the ``diff`` pair.

    Column labels scale with the tile width, so they stay legible at 100% zoom on wide panels. With
    ``display_gain_ev`` a second row repeats every image column brightened by that many stops in linear
    light, under a banner marking it display only: masters, metrics and the difference column never see it.
    """
    names = list(tiles)
    height, width = tiles[names[0]].shape[:2]
    images = [("source (nearest)", cv2.resize(source_tile, (width, height), interpolation=cv2.INTER_NEAREST))]
    images += [(name, tiles[name]) for name in names]
    a, b = diff
    delta = np.abs(tiles[a].astype(np.int16) - tiles[b].astype(np.int16)).max(axis=2)
    heat = cv2.applyColorMap(np.clip(delta * _DIFF_GAIN, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    columns = [*images, (f"|{a} - {b}| x{_DIFF_GAIN}", cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))]
    total = len(columns) * width + (len(columns) - 1) * _GUTTER
    font = _panel_font(width)
    bands = [
        _text_band([label for label, _ in columns], width, total, font, (235, 235, 235)),
        _image_row([tile for _, tile in columns], total),
    ]
    if display_gain_ev:
        banner = (
            f"Row below: the image columns above at +{display_gain_ev:g} EV display gain "
            f"(x{2**display_gain_ev:g} in linear light) to show shadow structure. Display only, "
            "never written to masters or used by any metric."
        )
        bands.append(_text_band([banner], total, total, font, (255, 200, 60)))
        bands.append(_image_row([_display_gain(tile, display_gain_ev) for _, tile in images], total))
    return np.ascontiguousarray(np.concatenate(bands, axis=0))


def _panel_font(tile_width: int):
    from matplotlib import font_manager
    from PIL import ImageFont

    # About 28 px on the ~630 px tiles of a 50 MP run, never below 13 px on small test tiles.
    size = int(np.clip(round(tile_width / 22), 13, 28))
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def _text_band(texts: list[str], column_width: int, total: int, font, colour: tuple[int, int, int]) -> np.ndarray:
    """One header band: each text word-wrapped into its own column, clipped to the column width."""
    from PIL import Image, ImageDraw

    pad = max(3, round(font.size / 3))
    line_height = round(font.size * 1.25)
    wrapped = [_wrap(text, font, column_width - 2 * pad) for text in texts]
    band_height = max(len(lines) for lines in wrapped) * line_height + 2 * pad
    band = Image.new("RGB", (total, band_height), (_PANEL_BACKGROUND,) * 3)
    for index, lines in enumerate(wrapped):
        cell = Image.new("RGB", (column_width, band_height), (_PANEL_BACKGROUND,) * 3)
        draw = ImageDraw.Draw(cell)
        for row, line in enumerate(lines):
            draw.text((pad, pad + row * line_height), line, font=font, fill=colour)
        band.paste(cell, (index * (column_width + _GUTTER), 0))
    return np.asarray(band)


def _wrap(text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split(" "):
        trial = f"{line} {word}" if line else word
        if line and font.getlength(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def _image_row(tiles: list[np.ndarray], total: int) -> np.ndarray:
    height, width = tiles[0].shape[:2]
    row = np.full((height, total, 3), _PANEL_BACKGROUND, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        x = index * (width + _GUTTER)
        row[:, x : x + width] = tile
    return row


def _display_gain(tile: np.ndarray, ev: float) -> np.ndarray:
    """An 8-bit sRGB tile brightened by ``ev`` stops in linear light, for display only."""
    linear = cm.srgb_to_linear(tile.astype(np.float32) / 255) * np.float32(2.0**ev)
    return np.round(255 * cm.linear_to_srgb(np.clip(linear, 0, 1))).astype(np.uint8)


def _blocks(count: int) -> str:
    return f"{count} block" if count == 1 else f"{count} blocks"


def fallback_figure(
    canvas: np.ndarray, source_linear: np.ndarray, fallbacks: list[dict], fraction: float, path: Path
) -> Path:
    """The PRD 12.6 fallback blocks over the dimmed source luma, with a reason legend and the fallback share."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    height, width = canvas.shape[:2]
    reduced = cv2.resize(np.clip(source_linear, 0, 1), (width, height), interpolation=cv2.INTER_AREA)
    luma = cm.linear_to_srgb(reduced) @ quality_metrics.LUMA_WEIGHTS
    picture = np.repeat((0.08 + 0.4 * luma)[..., None], 3, axis=2)
    marked = (canvas != night.FALLBACK_BACKGROUND).any(axis=2)
    picture[marked] = canvas[marked] / 255
    counts = {reason: 0 for reason in night.FALLBACK_COLORS}
    for record in fallbacks:
        counts[record["reasons"][0]] += 1

    fig, ax = plt.subplots(figsize=(10, 10 * height / width + 1.0), dpi=150)
    ax.imshow(picture, interpolation="nearest")
    ax.set_title(
        f"Failure-rule fallback blocks (PRD 12.6): {_blocks(len(fallbacks))}, {fraction:.2%} of output pixels "
        "(over the dimmed source)",
        fontsize=11,
    )
    ax.axis("off")
    handles = [
        Patch(color=np.array(colour) / 255, label=f"{reason.replace('_', ' ')}: {_blocks(counts[reason])}")
        for reason, colour in night.FALLBACK_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def region_figure(regions: night.RegionMap, source_linear: np.ndarray, path: Path) -> Path:
    """The PRD 8.3 region labels over the source luminance, with a legend of region shares."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    overlay = regions.visualize(source_linear)
    fractions = regions.fractions()
    height, width = regions.shape
    fig, ax = plt.subplots(figsize=(10, 10 * height / width + 1.2), dpi=150)
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title("Luminance regions (PRD 8.3), segmented on the source")
    ax.axis("off")
    handles = [
        Patch(color=np.array(night.REGION_COLORS[n]) / 255, label=f"{n} {fractions[n]:.1%}") for n in night.REGION_NAMES
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def banding_figure(profiles: _ProfileLines, input_key: str, path: Path, backend: str = "") -> Path:
    """Gradient and banding diagnostics: banding maps before and after Layer 5B, and a luma profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def overlay(flat_ramp):
        flat, ramp = flat_ramp
        canvas = np.zeros((*ramp.shape, 3), dtype=np.float32)
        canvas[ramp] = 0.35
        canvas[flat] = (0.95, 0.25, 0.2)
        return canvas, (flat.sum() / ramp.sum()) if ramp.any() else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=120)
    for ax, key in zip(axes[:2], (input_key, FINAL), strict=True):
        canvas, share = overlay(profiles.banding[key])
        ax.imshow(canvas, interpolation="nearest")
        ax.set_title(f"{_label(key, backend)}: flat share of ramp pixels {share:.2f}", fontsize=10)
        ax.axis("off")
    lines = profiles.lines["gradient"]
    for key, style in (("source", "steps-mid"), (input_key, "default"), (FINAL, "default")):
        axes[2].plot(
            lines[key], drawstyle=style, linewidth=1.2, label=_label(key, backend) if key != "source" else "source"
        )
    axes[2].set_title("Luma along the crop's centre column", fontsize=10)
    axes[2].set_xlabel("output row in crop")
    axes[2].set_ylabel("8-bit code value")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Gradient and banding diagnostics (grey: ramp; red: flat steps inside a ramp)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def halo_figure(profiles: _ProfileLines, keys: list[str], path: Path, backend: str = "") -> Path:
    """Light-source halo comparison: luma through the widest clipped light for the source, every slot and the result."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = profiles.lines["highlight"]
    rows = profiles.halo_rows
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
    nearest = f"source row {rows['nearest']} (nearest, {abs(rows['source_y'] - rows['nearest']):.2f} px away)"
    adjacent = f"source row {rows['adjacent']} ({abs(rows['source_y'] - rows['adjacent']):.2f} px away)"
    ax.plot(lines["source"], drawstyle="steps-mid", color="black", linewidth=1.2, label=nearest)
    ax.plot(lines["source_adjacent"], drawstyle="steps-mid", color="0.6", linestyle=":", linewidth=1.2, label=adjacent)
    for key in keys:
        if key in lines:
            # Dashed SR curves stay visible where they overlap the baselines.
            style = "--" if key.endswith("_sr") else "-"
            ax.plot(lines[key], style, linewidth=1.8 if key == FINAL else 1.0, label=_label(key, backend))
    ax.set_title("Light-source halo comparison: luma through the widest clipped light (undershoot = dark halo)")
    ax.set_xlabel(
        f"output column in the highlight crop (output row {rows['output_row']} = source y {rows['source_y']:.2f}; "
        "upscales blend both source rows)"
    )
    ax.set_ylabel("8-bit code value")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _provenance(run_id: str, source, size: tuple[int, int], night_record: dict) -> dict:
    upscale = night_record["input"]
    sr_input = upscale["provenance"] == night.SYNTHESIZED
    model = {"model_weights_sha256": upscale["weights_sha256"], "model_license": upscale["license"]} if sr_input else {}
    return {
        "notice": NOTICE_SR if sr_input else NOTICE,
        "run_id": run_id,
        "source_name": source.original_path.name,
        "source_sha256": source.sha256,
        "source_dimensions": [source.width, source.height],
        "output_dimensions": list(size),
        "method": f"{night_record['input']['label']}, then Layer 5B night processing",
        "stage_provenance": {s["name"]: s["provenance"] for s in night_record["stages"] if s["enabled"]},
        "information": night_record["information"],
        "failure_rule_fallbacks": len(night_record["fallbacks"]),
        **model,
        "package_version": __version__,
    }


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
    exports = list(manifest.get("intermediates", [])) + list(manifest.get("final", {}).get("exports", {}).values())
    manifest["output_checksums"] = {Path(e["path"]).name: e["file_sha256"] for e in exports}
    manifest_path = paths["reports"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    write_night_report(manifest, paths["reports"] / "report.md")
    log(f"report: {paths['reports'] / 'report.md'}")
    return manifest_path


# --------------------------------------------------------------------------- report (PRD 27)


def write_night_report(manifest: dict, path: Path) -> Path:
    path.write_text(render_night_report(manifest, path.parent))
    return path


# (label, getter, digits) for the PRD 23.5 single-image metrics.
METRIC_ROWS = (
    ("Highlight clipping, any channel", lambda m: m["highlight_clipping_fraction"]["any_channel"], 5),
    ("Highlight clipping, all channels", lambda m: m["highlight_clipping_fraction"]["all_channels"], 5),
    ("Clipping within light regions", lambda m: m["highlight_clipping_fraction"]["within_light_regions"], 4),
    ("Gradient banding score", lambda m: m["banding_score"], 4),
    ("Halo fraction around bright edges (outside the 2x2 source range)", lambda m: m["halo"]["fraction"], 4),
    ("Halo mean excursion (8-bit codes)", lambda m: m["halo"]["mean_excursion_codes"], 3),
    ("Shadow chroma noise (8-bit codes)", lambda m: m["shadow_chroma_noise_codes"], 3),
    ("Black crush fraction", lambda m: m["black_crush_fraction"], 4),
)


def metrics_table(manifest: dict) -> tuple[list[str], list[tuple]]:
    """Header and rows of the before/after PRD 23.5 table: source, every comparison slot, the final result."""
    columns = [("source", manifest.get("source_night_metrics"))]
    backend = _backend(manifest)
    columns += [(_label(k, backend), s.get("night_metrics")) for k, s in manifest.get("comparisons", {}).items()]
    record = manifest.get("night")
    if record:
        columns += [
            ("night input (before)", record["metrics_before"]),
            ("final night (after)", record["metrics_after"]),
        ]
    columns = [(name, values) for name, values in columns if values]
    rows = [(label, *(_num(get(values), digits) for _, values in columns)) for label, get, digits in METRIC_ROWS]
    return ["Metric", *(name for name, _ in columns)], rows


def render_night_report(manifest: dict, report_dir: Path) -> str:
    out: list[str] = []
    add = out.append
    source = manifest.get("source", {})
    target = manifest.get("target_resolution", {})
    runtime = manifest.get("runtime", {})
    record = manifest.get("night")
    final = manifest.get("final", {})
    figures = report_dir / "figures"
    gpu = runtime.get("gpu")

    add(f"# Phase 3 nighttime report: {Path(source.get('original_path', 'unknown source')).name}\n")
    add(f"> {manifest['notice']}\n")
    add(f"Status: **{manifest.get('status')}**\n")
    if manifest.get("error"):
        add(f"Failure: `{manifest['error']['message']}`. Outputs written before the failure are kept.\n")

    add("## 1. Run summary\n")
    add(
        _table(
            ["Item", "Value"],
            [
                ("Run ID", manifest.get("run_id", "-")),
                ("Source", f"`{Path(source.get('original_path', '-')).name}`"),
                ("Source SHA-256", f"`{source.get('sha256', '-')}`"),
                ("Input dimensions", _dims(source.get("width"), source.get("height"))),
                (
                    "Output dimensions",
                    _dims(target.get("width"), target.get("height"))
                    + (f" ({target['megapixels']:.3f} MP)" if target else ""),
                ),
                ("Pipeline version", _version(manifest)),
                ("Notebook version", manifest.get("notebook_version") or "- (run outside a notebook)"),
                ("Models", _models_text(manifest)),
                ("GPU", f"{gpu['name']} ({gpu['memory_total_gb']} GB)" if gpu else "none detected (CPU only)"),
                ("Platform", f"{runtime.get('platform', '-')}, Python {runtime.get('python', '-')}"),
                ("Total runtime", f"{manifest.get('total_seconds', 0):.1f} s"),
                ("Peak resident memory", f"{manifest.get('memory', {}).get('pipeline_peak_rss_mb', '-')} MB"),
                ("Storage used", _bytes(manifest.get("storage_used_bytes"))),
            ],
        )
    )

    add("## 2. Scientific limitations\n")
    add(
        "- The source is a rendered, display-referred image; its linearised values are not scene-referred "
        "radiance. Night processing redistributes and filters existing values; it recovers no shadow "
        "measurements and no clipped highlight radiance."
    )
    add(_sr_limitation(manifest.get("super_resolution", {})))
    add("- Dither, grain and bloom are simulated effects, not recovered low-light sensor measurements.")
    add("- 16-bit and floating-point encodings add numerical precision, not dynamic range or detail.\n")

    if manifest.get("validation"):
        add("## 3. Input validation (PRD 3.4)\n")
        add(_table(["Check", "Result"], _validation_rows(manifest["validation"])))
        add("![Source histograms](figures/source_histograms.png)\n")

    if target:
        add("## 4. Target resolution (PRD 3.5)\n")
        add(
            _table(
                ["Quantity", "Value"],
                [
                    ("Requested pixels", f"{target['target_pixels']:,}"),
                    ("Achieved pixels", f"{target['pixels']:,}"),
                    ("Pixel-count error", f"{target['pixel_count_error']:+.4%}"),
                    ("Aspect-ratio error", f"{target['aspect_ratio_error']:+.5%}"),
                    ("Scale x / y", f"{target['scale_x']:.5f} / {target['scale_y']:.5f}"),
                ],
            )
        )

    if manifest.get("regions"):
        regions = manifest["regions"]
        add("## 5. Scene analysis: luminance regions (PRD 8.3)\n")
        add(
            _table(
                ["Region", "Share of source pixels"],
                [(name, f"{share:.2%}") for name, share in regions["fractions"].items()],
            )
        )
        overrides = regions.get("overrides") or []
        add(
            "Manual overrides: "
            + ("; ".join(f"{o['region']} at {o['where']} ({o['pixels']} px) {o['note']}" for o in overrides) or "none")
            + ".\n"
        )
        if (figures / "regions.png").exists():
            add("![Region masks](figures/regions.png)\n")

    add("## 6. Processing log (PRD 27.2)\n")
    stage_rows = []
    for stage in manifest.get("stages", []):
        issues = "; ".join(stage["memory_warnings"] + ([stage["error"]] if stage["error"] else []) + stage["notes"])
        stage_rows.append(
            (
                stage["name"],
                stage["classification"],
                f"{stage['seconds']:.2f}",
                f"{stage['input_precision']} -> {stage['output_precision']}",
                f"{stage['rss_peak_mb']:.0f}" if stage["rss_peak_mb"] is not None else "-",
                issues or "-",
            )
        )
    add(
        _table(
            ["Stage", "Class (PRD 5.2)", "Seconds", "Precision", "Peak RSS MB", "Warnings, errors, notes"], stage_rows
        )
    )
    if record:
        add("Layer 5B steps, with the provenance label each output carries:\n")
        add(
            _table(
                ["Step", "Enabled", "Provenance", "Class", "Parameters", "Note"],
                [
                    (
                        s["name"],
                        "yes" if s["enabled"] else "no",
                        s["provenance"],
                        s["classification"],
                        _short_params(s["parameters"]),
                        s["note"] or "-",
                    )
                    for s in record["stages"]
                ],
            )
        )
        fallbacks = record["fallbacks"]
        add(
            f"Failure rule (PRD 12.6): {_blocks(len(fallbacks))} fell back to the conservative result "
            f"({record['fallback_fraction']:.3%} of output pixels).\n"
        )
        if fallbacks:
            add(
                _table(
                    ["x", "y", "Width", "Height", "Reasons", "Regions"],
                    [
                        (f["x"], f["y"], f["width"], f["height"], ", ".join(f["reasons"]), ", ".join(f["regions"]))
                        for f in fallbacks[:50]
                    ],
                )
            )
            if len(fallbacks) > 50:
                add(f"... and {len(fallbacks) - 50} more blocks (all listed in manifest.json).\n")
        if (figures / "fallback_map.png").exists():
            add(
                "Fallback map (red halo, blue banding, purple black crush, yellow excess texture):\n\n"
                "![Fallback map](figures/fallback_map.png)\n"
            )

    add("## 7. Visual results (PRD 19.3, 27.3)\n")
    for name, caption in (
        ("source_preview.jpg", "Original source"),
        ("lanczos_preview.jpg", "Lanczos baseline at output size (review downscale)"),
        ("night_preview.jpg", "Final night master at output size (review downscale)"),
    ):
        if (figures / name).exists():
            add(f"{caption}:\n\n![{caption}](figures/{name})\n")
    slots = manifest.get("comparisons", {})
    if slots:
        add("Comparison images (identical crops in every panel):\n")
        add(
            _table(
                ["Panel label", "Image", "Provenance"],
                [
                    (_label(k, _backend(manifest)), s.get("label", "-"), s.get("provenance", "-"))
                    for k, s in slots.items()
                ]
                + ([(_label(FINAL), "Layer 5B night processing of the night input", "see section 6")] if final else []),
            )
        )
    for panel in manifest.get("crop_panels", []):
        gain = panel.get("display_gain_ev")
        add(
            f"Crop `{panel['crop']}`: original (nearest), baselines, SR slots and final at 100% output pixels, "
            f"then the amplified difference between the night input and the final"
            + (f"; the second row repeats the images at +{gain:g} EV display gain (display only)" if gain else "")
            + f":\n\n![{panel['crop']}](figures/{Path(panel['path']).name})\n"
        )
    if (figures / "banding_diagnostics.png").exists():
        add("Gradient and banding diagnostics:\n\n![Banding](figures/banding_diagnostics.png)\n")
    if (figures / "halo_profile.png").exists():
        add("Light-source halo comparison:\n\n![Halo](figures/halo_profile.png)\n")
    if (figures / "delta_e_night.png").exists():
        add(
            "Difference map, night master reduced to source resolution vs source:\n\n"
            "![delta E](figures/delta_e_night.png)\n"
        )

    sr_record = manifest.get("super_resolution", {})
    if sr_record.get("modes") or sr_record.get("skipped"):
        # Phase 2's own section: weights, device, tiling, seam check, projection and difference to Lanczos.
        add(_sr_section(sr_record).replace("## 8b.", "## 7b.", 1))
        for mode in sr_record.get("modes", {}):
            if (figures / f"sr_difference_{mode}.png").exists():
                add(
                    f"|{mode} SR - Lanczos| (synthesised detail and drift):\n\n"
                    f"![{mode} SR difference](figures/sr_difference_{mode}.png)\n"
                )

    add("## 8. Quantitative results (PRD 23.5, 27.4)\n")
    header, rows = metrics_table(manifest)
    if len(header) > 1:
        add("Night metrics before and after night processing (PRD 23.5):\n")
        add(_table(header, rows))
    if record:
        paired = record["comparison"]
        protected = paired["protected_color_difference_ciede2000"]
        add("Paired comparisons, night input vs final:\n")
        add(
            _table(
                ["Quantity", "Value"],
                [
                    ("Protected-region delta E 2000 mean (lights, reflections)", _num(protected["mean"], 3)),
                    ("Protected-region delta E 2000 p95", _num(protected["p95"], 3)),
                    ("Protected-region hue and chroma difference mean", _num(protected["chromatic_mean"], 3)),
                    ("Shadow chroma mean shift (8-bit codes)", _num(paired["shadow_chroma_mean_shift_codes"], 3)),
                ],
            )
        )
        before, after = (
            record["metrics_before"]["local_contrast_codes"],
            record["metrics_after"]["local_contrast_codes"],
        )
        add("Local contrast per region (RMS luma detail, 8-bit codes):\n")
        add(
            _table(
                ["Region", "Before", "After", "Change ratio"],
                [
                    (n, _num(before[n], 3), _num(after[n], 3), _num(paired["local_contrast_change"][n], 3))
                    for n in before
                ],
            )
        )
    consistency = [(k, s["source_consistency"]) for k, s in slots.items() if "source_consistency" in s]
    if "source_consistency" in final:
        consistency.append((FINAL, final["source_consistency"]))
    if consistency:
        add("Source consistency (reduced to source size; night processing changes tone by design):\n")
        add(
            _table(
                ["Metric", *(_label(k, _backend(manifest)) for k, _ in consistency)],
                [
                    ("PSNR (dB)", *(_num(m["psnr_db"], 2) for _, m in consistency)),
                    ("SSIM", *(_num(m["ssim"], 5) for _, m in consistency)),
                    ("delta E 2000 mean", *(_num(m["color_difference_ciede2000"]["mean"], 4) for _, m in consistency)),
                ],
            )
        )
    exports = list(manifest.get("intermediates", [])) + list(final.get("exports", {}).values())
    if exports:
        add("Export validation (PRD 23.6):\n")
        add(
            _table(
                ["File", "Kind", "Size", "Checks", "SHA-256"],
                [
                    (
                        f"`{Path(e['path']).name}`",
                        e["kind"],
                        _bytes(e["size_bytes"]),
                        f"{sum(e['checks'].values())}/{len(e['checks'])} passed",
                        f"`{e['file_sha256'][:16]}...`",
                    )
                    for e in exports
                ],
            )
        )

    add("## 9. Information provenance (PRD 27.5)\n")
    information = record["information"] if record else {}
    add(
        _table(
            ["Contribution", "In this run"],
            [
                ("Original source data (preserved)", information.get("preserved", "-")),
                ("Interpolated and estimated values", information.get("interpolated", "-")),
                ("Learned inference (synthesized)", information.get("inferred", "-")),
                ("Simulated effects (dither, grain, bloom)", information.get("simulated", "-")),
                ("File encoding", "16-bit sRGB TIFF (clips) and half or float linear EXR; precision only."),
            ],
        )
    )

    add("## 10. Conclusions (PRD 27.6)\n")
    add("\n".join(f"- {line}" for line in night_conclusions(manifest)) + "\n")

    add("## 11. Provenance record (PRD 6.2)\n")
    add(
        _table(
            ["Item", "Value"],
            [
                ("Source SHA-256", f"`{source.get('sha256', '-')}`"),
                ("Decoded pixel SHA-256", f"`{source.get('pixel_sha256', '-')}`"),
                ("Package", _version(manifest)),
                ("Models and weight hashes", _models_text(manifest, hashes=True)),
                ("Random seeds", json.dumps(manifest.get("random_seeds", {}))),
            ],
        )
    )
    add("Output checksums (SHA-256):\n")
    add(_table(["File", "SHA-256"], [(f"`{k}`", f"`{v}`") for k, v in manifest.get("output_checksums", {}).items()]))
    add("Configuration:\n")
    add("```json\n" + json.dumps(manifest.get("config", {}), indent=2) + "\n```\n")
    return "\n".join(out)


def _sr_limitation(sr_record: dict) -> str:
    if sr_record.get("modes"):
        return (
            "- The conservative and balanced SR slots run the Phase 2 pretrained networks (backend "
            f"`{sr_record['backend']}`). They predict detail above the source sampling limit from priors learned on "
            "other photographs; that detail is synthesised, not recovered, and may be wrong even when it looks "
            "plausible (PRD 10.5)."
        )
    reason = sr_record.get("skipped") or "super_resolution.enabled is false"
    return f"- No super-resolution ran ({reason}): every upscale is classical interpolation and adds no learned detail."


def _sr_conclusion(manifest: dict) -> str:
    sr_record = manifest.get("super_resolution", {})
    modes = sr_record.get("modes", {})
    night_input = (manifest.get("night") or {}).get("input", {})
    if not modes:
        reason = sr_record.get("skipped") or "super_resolution.enabled is false"
        return f"No SR comparison ran ({reason}); the night input is the Lanczos baseline."
    seams = [f"{m} {'passed' if s['seams']['passed'] else 'FAILED'}" for m, s in modes.items() if "seams" in s]
    text = f"SR comparisons ran with the {sr_record['backend']} backend ({', '.join(modes)})"
    text += f"; seam checks: {', '.join(seams)}" if seams else ""
    if night_input.get("mode"):
        text += f"; the night input is the {night_input['mode']} SR result, built once and reused"
    return text + "."


_ADVICE = {
    "halo": "raise highlights.halo_suppression_strength or lower texture.sharpening_strength",
    "banding": "keep shadows.banding_reduction on and lower texture.atmospheric_detail_strength",
    "black_crush": "lower shadows.black_point",
    "excess_texture": "lower texture.architecture_detail_strength or texture.sharpening_strength",
}


def night_conclusions(manifest: dict) -> list[str]:
    """Factual, automatically checkable statements; visual judgements are left to human review."""
    lines = []
    target = manifest.get("target_resolution")
    config = manifest.get("config", {})
    record = manifest.get("night")
    if manifest.get("status") != "complete":
        lines.append(f"The run did not complete ({manifest.get('status')}); conclusions cover finished stages only.")
    if target:
        within = abs(target["pixel_count_error"]) <= config.get("pixel_count_tolerance", 0.001)
        lines.append(
            f"Output size {target['width']}x{target['height']} ({target['pixels']:,} pixels) "
            f"{'meets' if within else 'misses'} the configured pixel-count target"
            + ("" if target["target_pixels"] >= 50_000_000 else f" (reduced {target['target_pixels'] / 1e6:g} MP run)")
            + "."
        )
    exports = manifest.get("final", {}).get("exports", {})
    if exports:
        valid = all(e["valid"] for e in exports.values())
        lines.append(
            f"Night master exports ({', '.join(exports)}) {'all passed' if valid else 'FAILED'} read-back validation."
        )
    if record:
        before, after = record["metrics_before"], record["metrics_after"]
        checks = (
            ("gradient banding score", before["banding_score"], after["banding_score"]),
            (
                "highlight clipping (any channel)",
                before["highlight_clipping_fraction"]["any_channel"],
                after["highlight_clipping_fraction"]["any_channel"],
            ),
            ("halo fraction", before["halo"]["fraction"], after["halo"]["fraction"]),
            ("shadow chroma noise", before["shadow_chroma_noise_codes"], after["shadow_chroma_noise_codes"]),
            ("black crush fraction", before["black_crush_fraction"], after["black_crush_fraction"]),
        )
        improved = [n for n, b, a in checks if b is not None and a is not None and a < b]
        worse = [n for n, b, a in checks if b is not None and a is not None and a > b]
        if improved:
            lines.append("Lower after night processing: " + ", ".join(improved) + ".")
        if worse:
            lines.append("Higher after night processing (review the crops): " + ", ".join(worse) + ".")
        lines.append(
            "The lower clipping fraction comes from the highlight shoulder rolling off clipped values; no clipped "
            "radiance was recovered."
        )
        reasons = sorted({r for f in record["fallbacks"] for r in f["reasons"]})
        if reasons:
            lines.append(
                f"The failure rule fell back in {_blocks(len(record['fallbacks']))} ({', '.join(reasons)}). "
                "Recommended: " + "; ".join(_ADVICE[r] for r in reasons) + "."
            )
        else:
            lines.append("The failure rule found no halos, banding, black crush or excess texture to fall back on.")
        lines.append(
            "Night metric criteria (banding, halos, clipping, shadow chroma noise and black crush not increased): "
            + ("met." if not worse else "not met.")
        )
    lines.append(_sr_conclusion(manifest))
    lines.append(
        "Visual acceptance (halos, banding, invented texture, light colour) needs human review of the crop "
        "panels at 100% and 200%."
    )
    return lines
