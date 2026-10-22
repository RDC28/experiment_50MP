"""Phase 5 synthetic sensor research run (PRD sections 14, 17.3, 24 E01 and E08, 30 Phase 5).

Input is a processed linear RGB master: the Phase 0/1 OpenEXR master, any
linear-light RGB EXR, or an in-memory working-space array. The run:

1. simulates a synthetic sensor (assumed camera colour transform, exposure
   scaling, CFA sampling, optional shot and read noise, ADC quantisation),
2. writes the simulated CFA data as a labelled mosaic TIFF and, when enabled,
   a research DNG that must pass structural and reader validation,
3. develops the mosaic back to RGB with each demosaic method (and with LibRaw
   from the DNG) and measures the round trip against the input (E08),
4. runs the bit-depth experiments (PRD 14.5, E01),
5. writes a manifest and a Markdown report.

Every stage is logged with its PRD 5.2 classification, duration and memory
peak. Every output says it is synthetic: the data structures, file metadata,
sidecars, manifest and report all carry the PRD 14.3 label.
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

import numpy as np

from . import (
    __version__,
    baselines,
    bit_depth_study,
    comparison,
    environment,
    exporters,
    ingestion,
    pipeline,
    quality_metrics,
    report_builder,
    synthetic_dng,
)
from . import color_management as cm
from . import sensor_simulation as ss
from .sensor_config import SensorResearchConfig

SIMULATION = "Simulation"
INTERPOLATION = pipeline.INTERPOLATION
PACKAGING = pipeline.PACKAGING
PRESERVATION = pipeline.PRESERVATION
MEASUREMENT = pipeline.MEASUREMENT

NOTICE = (
    "Synthetic sensor research output (PRD Phase 5). Every mosaic, DNG and developed image here is SIMULATED "
    "from processed RGB through an assumed camera model. None of it is camera-native RAW, sensor measurement or "
    "evidence of a physical exposure."
)
RAW_TERMINOLOGY = (  # PRD 17.3
    ("Camera-native RAW", "Actual camera sensor acquisition.", "Never produced by this experiment."),
    (
        "Synthetic mosaic DNG",
        "Simulated CFA-domain data in a DNG container.",
        "The research DNG this run writes when export_synthetic_dng is on.",
    ),
    (
        "Linear DNG",
        "DNG of processed, already demosaiced linear image data.",
        "Not produced; the TIFF/EXR masters cover that need.",
    ),
    (
        "TIFF/EXR master",
        "Professional image formats, not camera RAW.",
        "The Phase 0/1 masters this run starts from; the preferred deliverables (PRD 17.3).",
    ),
)
LIBRAW = "libraw"


# --------------------------------------------------------------------------- input


@dataclass
class LinearInput:
    name: str
    linear: np.ndarray  # working-space float32 (H, W, 3)
    pixel_sha256: str
    file_sha256: str | None
    path: str | None
    lineage: dict
    notes: list[str]


def read_linear_master(path: str | Path) -> LinearInput:
    """Read a linear-light RGB OpenEXR (such as a Phase 0/1 master) into the working space."""
    import OpenEXR

    path = Path(path)
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        header = exr.header()
        channels = exr.channels()
        missing = [name for name in "RGB" if name not in channels]
        if missing:
            raise ValueError(f"{path.name} lacks EXR channels {missing}; expected linear R, G, B")
        linear = np.empty((*channels["R"].pixels.shape, 3), dtype=np.float32)
        for index, name in enumerate("RGB"):
            linear[..., index] = channels[name].pixels  # one plane at a time keeps a single float32 copy
        chromaticities = header.get("chromaticities")
        lineage = {
            key: str(value)
            for key, value in header.items()
            if key.startswith("syntheticRawLab") or key in ("comments", "colorSpaceDescription")
        }
    notes = []
    if chromaticities is not None:
        space = cm.linear_color_space_from_chromaticities(chromaticities)
        if space.matrix is not None:
            linear = cm.to_working_space(linear, space)
            notes.append(f"converted from {space.description} to linear Rec.709")
    else:
        notes.append("no chromaticities attribute; assumed linear Rec.709 primaries")
    nonfinite = int(linear.size - np.count_nonzero(np.isfinite(linear)))
    if nonfinite:
        raise ValueError(f"{path.name} contains {nonfinite} NaN or infinite values")
    return LinearInput(
        path.name, linear, ingestion.pixel_sha256(linear), ingestion.sha256_file(path), str(path), lineage, notes
    )


def linear_input(linear: np.ndarray, name: str = "array") -> LinearInput:
    """Wrap an in-memory working-space (H, W, 3) array as a run input."""
    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) linear RGB array, got {linear.shape}")
    linear = np.ascontiguousarray(linear, dtype=np.float32)
    return LinearInput(name, linear, ingestion.pixel_sha256(linear), None, None, {}, [])


# --------------------------------------------------------------------------- run


@dataclass
class RunResult:
    run_id: str
    manifest: dict
    manifest_path: Path
    report_path: Path
    outputs_dir: Path
    reports_dir: Path


def run(
    master: str | Path | np.ndarray,
    config: SensorResearchConfig,
    project_root: str | Path,
    *,
    source_codes: np.ndarray | None = None,
    notebook_version: str | None = None,
    run_id: str | None = None,
    log: Callable[[str], None] = print,
) -> RunResult:
    """Run the Phase 5 experiments on a processed linear RGB master (EXR path or working-space array)."""
    project_root = Path(project_root).resolve()
    output_root = Path(config.output_root) if config.output_root else project_root / "outputs"
    reports_root = Path(config.reports_root) if config.reports_root else project_root / "reports"
    model = config.sensor_model()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    runtime = environment.inspect_runtime(output_root)
    monitor = environment.MemoryMonitor().start()
    stages: list[pipeline.StageRecord] = []
    manifest: dict = {
        "schema": "synthetic_raw_lab.phase5.manifest/1",
        "phase": "Phase 5 (synthetic sensor research)",
        "notice": NOTICE,
        "synthetic_raw_label": model.synthetic_raw_label,
        "started_at": started_at.isoformat(timespec="seconds"),
        "package": {
            "name": "synthetic_raw_lab",
            "version": __version__,
            "git": environment.git_revision(Path(__file__).parent) or environment.git_revision(project_root),
        },
        "notebook_version": notebook_version,
        "models": [],
        "random_seeds": {"random_seed": config.random_seed, "used_by": "shot and read noise generators"},
        "config": config.to_dict(),
        "sensor": model.to_dict(),
        "runtime": runtime,
        "exports": {},
        "roundtrip": {},
        "status": "running",
    }
    paths: dict[str, Path] = {}

    def stage(record: pipeline.StageRecord):
        return pipeline._stage(record, stages, monitor, log)

    def fail(error: pipeline.StageError) -> pipeline.StageError:
        manifest["status"] = f"failed in stage {error.stage}"
        manifest["error"] = {"stage": error.stage, "message": str(error), "traceback": traceback.format_exc()}
        error.manifest_path = _finish(manifest, stages, monitor, started, paths, log)
        return error

    try:
        with stage(pipeline.StageRecord("load_input", PRESERVATION, {}, "linear RGB", "float32 linear")) as record:
            source = read_linear_master(master) if not isinstance(master, np.ndarray) else linear_input(master)
            linear = source.linear
            height, width = linear.shape[0] // 2 * 2, linear.shape[1] // 2 * 2
            if (height, width) != linear.shape[:2]:
                source.notes.append(f"cropped to even size {width}x{height} for the 2 x 2 CFA")
                linear = np.ascontiguousarray(linear[:height, :width])
            record.notes = list(source.notes)

        run_id = run_id or pipeline.make_run_id(source.pixel_sha256, config.profile, started_at)
        outputs_dir = output_root / config.input_category / run_id
        reports_dir = reports_root / run_id
        figures_dir = reports_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        paths.update(outputs=outputs_dir, reports=reports_dir, figures=figures_dir)
        manifest.update(
            run_id=run_id,
            input={
                "name": source.name,
                "path": source.path,
                "file_sha256": source.file_sha256,
                "pixel_sha256": source.pixel_sha256,
                "width": width,
                "height": height,
                "lineage": source.lineage,
                "notes": source.notes,
                "statistics": quality_metrics.image_statistics(linear),
            },
        )
        log(f"input {source.name}: {width}x{height}")
        stem = _stem(source.name)

        with stage(
            pipeline.StageRecord(
                "camera_transform_and_cfa_sampling",
                SIMULATION,
                {"cfa": model.cfa_pattern, "exposure_scale": model.exposure_scale},
                "float32 linear RGB",
                "float32 mosaic signal",
            )
        ):
            signal = ss.sample_cfa(linear, model)
            manifest["mosaic_quantization"] = bit_depth_study.mosaic_quantization(
                signal, model.black, model.white, model.adc_bit_depth
            )

        with stage(
            pipeline.StageRecord(
                "noise_and_quantization",
                SIMULATION,
                {"noise": model.noise_description(), "adc_bits": model.adc_bit_depth, "seed": config.random_seed},
                "float32 mosaic signal",
                f"uint16 codes ({model.adc_bit_depth}-bit)",
            )
        ):
            dn, stats = ss.quantize(signal, model, config.random_seed)
            mosaic = ss.SyntheticMosaic(dn, model, config.random_seed, stats)
            manifest["mosaic"] = mosaic.provenance()
            del signal
            gc.collect()

        metadata = _file_metadata(run_id, source, mosaic)
        if config.export_mosaic_tiff:
            with stage(pipeline.StageRecord("export_mosaic_tiff", PACKAGING, {}, "uint16 codes", "16-bit TIFF")):
                result = synthetic_dng.write_mosaic_tiff(
                    outputs_dir / f"{stem}_synthetic_mosaic_{model.cfa_pattern}_{model.adc_bit_depth}bit.tif",
                    mosaic,
                    metadata,
                )
                exporters.write_sidecar(result, metadata)
                manifest["exports"]["mosaic_tiff"] = result.to_dict()

        dng_path = None
        if config.export_synthetic_dng:
            with stage(
                pipeline.StageRecord(
                    "export_synthetic_dng", PACKAGING, {"dng_version": "1.4.0.0"}, "uint16 codes", "DNG"
                )
            ):
                result = synthetic_dng.write_synthetic_dng(
                    outputs_dir / f"{stem}_SYNTHETIC_mosaic_{model.cfa_pattern}_{model.adc_bit_depth}bit.dng",
                    mosaic,
                    metadata,
                    synthetic_dng.preview_from_linear(linear),
                )
                exporters.write_sidecar(result, metadata)
                manifest["exports"]["dng"] = {
                    **result.to_dict(),
                    "exiftool": result.exiftool or {"ok": None, "output": ["exiftool not installed; check skipped"]},
                }
                dng_path = result.path

        crops = _select_crops(linear, config)
        manifest["crops"] = [{**c.to_dict(), "space": "input pixels"} for c in crops]
        crop_tiles: dict[str, dict[str, np.ndarray]] = {c.name: {} for c in crops}
        input_display = {c.name: _display_tile(linear, c) for c in crops}
        if config.review_max_edge:
            _preview(linear, figures_dir / "input_preview.jpg", config.review_max_edge)

        methods = list(config.demosaic_methods)
        if dng_path is not None and config.libraw_roundtrip:
            methods.append(LIBRAW)
        for method in methods:
            description = synthetic_dng.LIBRAW_DESCRIPTION if method == LIBRAW else ss.DEMOSAIC_DESCRIPTIONS[method]
            with stage(
                pipeline.StageRecord(
                    f"develop_{method}", INTERPOLATION, {"method": description}, "uint16 codes", "float32 linear"
                )
            ):
                if method == LIBRAW:
                    developed = synthetic_dng.develop_with_libraw(dng_path, mosaic)
                else:
                    developed = ss.develop(dn, model, method)
            with stage(
                pipeline.StageRecord(
                    f"roundtrip_metrics_{method}",
                    MEASUREMENT,
                    {"tile": config.metric_tile_size, "grid": config.metric_tile_grid},
                    "float32 linear",
                    "metrics",
                )
            ):
                metrics, delta_e = roundtrip_metrics(
                    linear, developed, config.metric_tile_size, config.metric_tile_grid
                )
                manifest["roundtrip"][method] = {
                    "description": description,
                    "classification": INTERPOLATION,
                    "information": ss.developed_information(model, description),
                    **metrics,
                }
                report_builder.heatmap_figure(
                    delta_e,
                    figures_dir / f"delta_e_{method}.png",
                    f"CIEDE2000, {method} round trip vs input ({metrics['sample']['description']})",
                    config.review_max_edge,
                )
                for crop in crops:
                    crop_tiles[crop.name][method] = _display_tile(developed, crop)
                if config.review_max_edge:
                    _preview(developed, figures_dir / f"roundtrip_{method}_preview.jpg", config.review_max_edge)
            del developed, delta_e
            gc.collect()

        with stage(
            pipeline.StageRecord(
                "crop_comparison", MEASUREMENT, {"crops": len(crops)}, "float32 linear", "uint8 review PNG"
            )
        ):
            panels = []
            for crop in crops:
                tiles = {"input": input_display[crop.name], **crop_tiles[crop.name]}
                path = exporters.write_review_image(
                    figures_dir / f"crop_{crop.name}.png",
                    comparison.crop_panel(
                        _cfa_tile(mosaic, crop), tiles, source_label=f"synthetic {model.cfa_pattern} CFA"
                    ),
                )
                panels.append({"crop": crop.name, "path": str(path), "columns": ["cfa", *tiles]})
            manifest["crop_panels"] = panels

        if config.bit_depth_study:
            with stage(
                pipeline.StageRecord(
                    "bit_depth_study",
                    MEASUREMENT,
                    {"edit_stops": config.edit_stops, "cycles": config.transform_cycles},
                    "float32 linear",
                    "metrics and sample files",
                )
            ):
                manifest["bit_depth"] = bit_depth_study.run_study(
                    linear,
                    outputs_dir / "bit_depth" if config.bit_depth_write_files else None,
                    edit_stops=config.edit_stops,
                    cycles=config.transform_cycles,
                    file_metadata={
                        "notice": "Bit-depth study sample of a processed master (not a camera file).",
                        "run_id": run_id,
                        "software": f"synthetic_raw_lab {__version__}",
                    },
                )
                if source_codes is not None:
                    manifest["bit_depth"]["expansion"] = bit_depth_study.expansion_information(source_codes)

        manifest["labels"] = label_checks(manifest)
        manifest["status"] = "complete"
    except pipeline.StageError as error:
        raise fail(error) from error.__cause__
    except Exception as error:
        last = stages[-1].name if stages else "start"
        raise fail(pipeline.StageError(f"unlogged step after {last}", error, None)) from error

    manifest_path = _finish(manifest, stages, monitor, started, paths, log)
    return RunResult(
        manifest["run_id"], manifest, manifest_path, paths["reports"] / "report.md", paths["outputs"], paths["reports"]
    )


def roundtrip_metrics(reference: np.ndarray, candidate: np.ndarray, tile: int, grid: int) -> tuple[dict, np.ndarray]:
    """E08 round-trip metrics at full resolution.

    Full frame: PSNR on sRGB-encoded values and linear error statistics, in row
    blocks. Phase 1 source-consistency metrics (SSIM, MS-SSIM, edges, CIEDE2000,
    L* shift) run on the whole image when it is small, otherwise on an evenly
    spaced grid of tiles assembled into one sample image.
    """
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch {reference.shape} vs {candidate.shape}")
    height, width = reference.shape[:2]
    encoded_sq = linear_abs = 0.0
    linear_max = 0.0
    for start in range(0, height, 256):
        ref = reference[start : start + 256]
        cand = candidate[start : start + 256]
        diff = (cm.linear_to_srgb(np.clip(cand, 0, 1)) - cm.linear_to_srgb(np.clip(ref, 0, 1))).astype(np.float64)
        encoded_sq += float(np.square(diff).sum())
        error = np.abs(cand - ref)
        linear_abs += float(error.sum(dtype=np.float64))
        linear_max = max(linear_max, float(error.max(initial=0.0)))
    mse = encoded_sq / reference.size
    ref_sample, cand_sample, description = _metric_sample(reference, candidate, tile, grid)
    consistency, delta_e = quality_metrics.source_consistency(ref_sample, cand_sample)
    return {
        "full_frame": {
            "psnr_encoded_db": float("inf") if mse == 0 else float(10 * np.log10(1 / mse)),
            "mean_abs_error_linear": linear_abs / reference.size,
            "max_abs_error_linear": linear_max,
        },
        "sample": {"description": description, "width": ref_sample.shape[1], "height": ref_sample.shape[0]},
        "consistency": consistency,
    }, delta_e


def label_checks(manifest: dict) -> dict:
    """PRD 14.3 / Phase 5 exit criterion: is every synthetic output labelled as synthetic?"""
    label = manifest.get("synthetic_raw_label", "")
    checks = {
        "label_says_synthetic": "synthetic" in label.lower(),
        "manifest_notice": "SIMULATED" in manifest.get("notice", ""),
        "mosaic_record": manifest.get("mosaic", {}).get("synthetic_raw_label") == label,
    }
    for name, export in manifest.get("exports", {}).items():
        checks[f"{name}_provenance_in_file"] = bool(export.get("checks", {}).get("provenance_roundtrip"))
        sidecar = Path(export["path"] + ".json")
        checks[f"{name}_sidecar"] = (
            sidecar.exists() and json.loads(sidecar.read_text())["provenance"].get("synthetic_raw_label") == label
        )
    if "dng" in manifest.get("exports", {}):
        checks["dng_unique_camera_model"] = bool(
            manifest["exports"]["dng"]["checks"].get("unique_camera_model_synthetic")
        )
    for method, summary in manifest.get("roundtrip", {}).items():
        checks[f"roundtrip_{method}_simulated"] = "simulated" in summary.get("information", {})
    return {"all_labelled": all(checks.values()), "checks": checks}


# --------------------------------------------------------------------------- helpers


def _file_metadata(run_id: str, source: LinearInput, mosaic: ss.SyntheticMosaic) -> dict:
    return {
        "notice": NOTICE,
        "synthetic_raw_label": mosaic.label,
        "format_class": "synthetic mosaic (PRD 17.3), not camera-native RAW",
        "run_id": run_id,
        "input_name": source.name,
        "input_pixel_sha256": source.pixel_sha256,
        "input_lineage": source.lineage,
        "stage_classification": [PRESERVATION, SIMULATION, PACKAGING],
        "information": ss.mosaic_information(mosaic.model),
        "sensor": mosaic.model.to_dict(),
        "random_seed": mosaic.seed,
        "software": f"synthetic_raw_lab {__version__}",
        "package_version": __version__,
    }


def _metric_sample(reference: np.ndarray, candidate: np.ndarray, tile: int, grid: int):
    height, width = reference.shape[:2]
    if height * width <= (tile * grid) ** 2 or height < tile or width < tile:
        return reference, candidate, "whole image"
    ys = np.linspace(0, height - tile, grid).astype(int)
    xs = np.linspace(0, width - tile, grid).astype(int)

    def assemble(image):
        return np.concatenate(
            [np.concatenate([image[y : y + tile, x : x + tile] for x in xs], axis=1) for y in ys], axis=0
        )

    return assemble(reference), assemble(candidate), f"{grid}x{grid} grid of {tile}-px full-resolution tiles"


def _select_crops(linear: np.ndarray, config: SensorResearchConfig) -> list[comparison.Crop]:
    """Pick crops on a reduced copy (bounded memory), then map them back to full input pixels."""
    height, width = linear.shape[:2]
    scale = min(1.0, 2048 / max(width, height))
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = baselines.downsample_area(linear, *small_size) if scale < 1 else linear
    display = cm.linear_to_srgb(np.clip(small, 0, 1))
    size = max(8, round(min(config.crop_size, width, height) * scale))
    chosen = comparison.select_crops(display, size, config.crop_count)
    crops = []
    for crop in chosen:
        x0, y0, _, _ = crop.output_box(small_size, (width, height))
        side = min(config.crop_size, width, height) // 2 * 2
        x0, y0 = min(x0 // 2 * 2, width - side), min(y0 // 2 * 2, height - side)
        crops.append(comparison.Crop(crop.name, x0, y0, side, side))
    return crops


def _display_tile(linear: np.ndarray, crop: comparison.Crop) -> np.ndarray:
    tile = linear[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
    return cm.srgb_uint16_to_uint8(cm.encode_srgb_uint16(tile)[0])


def _cfa_tile(mosaic: ss.SyntheticMosaic, crop: comparison.Crop) -> np.ndarray:
    """Each photosite drawn in the colour it samples (white balanced), as an sRGB tile."""
    model = mosaic.model
    rows = slice(crop.y, crop.y + crop.height)
    cols = slice(crop.x, crop.x + crop.width)
    native = ss.normalize(mosaic.dn[rows, cols], model)
    masks = ss.cfa_masks(crop.height, crop.width, model.cfa_pattern)
    balance = np.asarray(model.white_balance_multipliers, dtype=np.float32)
    rgb = np.stack([np.where(masks[c], native * balance[c], 0) for c in range(3)], axis=-1).astype(np.float32)
    return cm.srgb_uint16_to_uint8(cm.encode_srgb_uint16(rgb)[0])


def _preview(linear: np.ndarray, path: Path, max_edge: int) -> None:
    height, width = linear.shape[:2]
    scale = min(1.0, max_edge / max(width, height))
    small = baselines.downsample_area(linear, max(1, round(width * scale)), max(1, round(height * scale)))
    exporters.write_review_image(path, cm.srgb_uint16_to_uint8(cm.encode_srgb_uint16(small)[0]))


def _finish(manifest: dict, stages, monitor, started: float, paths: dict, log) -> Path | None:
    from . import sensor_report

    monitor.stop()
    manifest["stages"] = [s.to_dict() for s in stages]
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["total_seconds"] = round(time.perf_counter() - started, 3)
    manifest["memory"] = {
        "pipeline_peak_rss_mb": round(monitor.peak_rss / environment.MB, 1),
        "process_lifetime_peak_rss_mb": environment.process_peak_rss_mb(),
    }
    if "reports" not in paths:
        log("run failed before an output folder existed; nothing written")
        return None
    files = [f for d in (paths["outputs"], paths["reports"]) for f in d.rglob("*") if f.is_file()]
    manifest["storage_used_bytes"] = sum(f.stat().st_size for f in files)
    manifest["output_checksums"] = {
        Path(e["path"]).name: e["file_sha256"] for e in manifest.get("exports", {}).values()
    }
    manifest_path = paths["reports"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    sensor_report.write_report(manifest, paths["reports"] / "report.md")
    log(f"report: {paths['reports'] / 'report.md'}")
    return manifest_path


def _stem(name: str) -> str:
    stem = Path(name).stem
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem) or "input"
