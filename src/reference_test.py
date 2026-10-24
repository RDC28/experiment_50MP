"""Experiment E10, the controlled reference test (PRD 24, 23.1, Phase 6).

The two AI-generated inputs have no ground truth, so nothing measured on them can say whether added detail is
correct. This test supplies one: a real, freely licensed high-resolution photograph is degraded to the source
scale the way an image like the inputs could have been formed, reconstructed back to its own size by the same
configurations the ablation runs, and scored against the original (:func:`ablation.reference_metrics`).

Degradation models (each ends as an 8-bit sRGB image, like the inputs):

* ``area``: ideal sampling, an area average over each source pixel in linear light (a perfect sensor with
  100% fill factor and no optics), then 8-bit sRGB rounding.
* ``camera``: the forward camera model of PRD 14.2: a Gaussian optical blur, area sampling, then the Phase 5
  synthetic sensor (CFA sampling, shot and read noise, ADC quantisation, Malvar demosaic), 8-bit sRGB and JPEG.
* ``ai_render``: a proxy for a diffusion render's look: a sharp Lanczos reduction, edge-preserving smoothing
  of fine texture, mild sharpening, 8-bit sRGB and JPEG at quality 99 (the captain's inputs estimate 99.1).

The references are fetched at run time from Wikimedia Commons; the licence, author and page are read from the
Commons API, checked against the free licences allowed here, recorded in the manifest and report, and the
file is verified against the SHA-1 pinned in :data:`REFERENCES`. Nothing is redistributed by the repository.
"""

from __future__ import annotations

import gc
import html
import json
import re
import time
import traceback
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1 as sha1_hash
from pathlib import Path

import cv2
import numpy as np

from . import __version__, baselines, environment, ingestion, scene_analysis, sensor_simulation
from . import color_management as cm
from .ablation import (
    Engine,
    Scene,
    phase2_sr_runner,
    review_crops,
    run_configurations,
    scene_params,
    write_review_figures,
)
from .evaluation_config import EvaluationConfig, ReferenceParams
from .pipeline import (
    MEASUREMENT,
    PRESERVATION,
    SETUP,
    StageError,
    StageRecord,
    _stage,
    model_cache_dir,
    resolve_roots,
)

SCHEMA = "synthetic_raw_lab.phase6_reference.manifest/1"
PHASE = "Phase 6 (evaluation and optimisation): controlled reference test (E10)"
NOTICE = (
    "Reconstructions of degraded copies of real photographs, scored against the originals. They measure what the "
    "pipeline can recover at the source scale; the photographs belong to their authors under the licences recorded."
)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = f"synthetic-raw-lab/{__version__} (https://github.com/RDC28/experiment_50MP; research evaluation)"
FREE_LICENCES = ("CC0", "Public domain", "CC BY ", "CC BY-SA ", "CC-BY", "CC-BY-SA")
AI_RENDER_JPEG_QUALITY = 99


@dataclass(frozen=True)
class Reference:
    """A freely licensed photograph on Wikimedia Commons, pinned by the SHA-1 Commons reports for the file."""

    name: str
    title: str
    sha1: str
    scene: str
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "sha1": self.sha1,
            "scene": self.scene,
            "description": self.description,
        }


REFERENCES = {
    r.name: r
    for r in (
        Reference(
            "portrait_holi",
            "File:Colorful Portrait Holi.jpg",
            "8c9d1aaeec13daa7926c6109629b3c440a689b7c",
            "human",
            "Daylight Holi festival portrait with colour powder on face and clothes; camera photograph, 5472 x 3648 "
            "(the analogue of the festival portrait input)",
        ),
        Reference(
            "night_forest_path",
            "File:Dülmen, Kirchspiel, Börnste, Pfad bei Nacht -- 2021 -- 2379.jpg",
            "dda398100c46292b3ab65ee8a307e13d03dd7c18",
            "night",
            "Forest path at night; camera photograph, 6720 x 4480 (the analogue of the night forest path input)",
        ),
    )
}

Opener = Callable[[str], bytes]


def http_get(url: str) -> bytes:
    """GET with the descriptive User-Agent Wikimedia's policy asks for."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (fixed https endpoints)
        return response.read()


def _plain(text: str | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def resolve(reference: str | dict) -> Reference:
    if isinstance(reference, str):
        return REFERENCES[reference]
    return Reference(
        reference["name"], reference["title"], reference["sha1"], reference["scene"], reference.get("description", "")
    )


def licence_record(reference: Reference, opener: Opener = http_get) -> dict:
    """Read the file's URL, size, SHA-1, licence and author from the Commons API and check them."""
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": reference.title,
            "prop": "imageinfo",
            "iiprop": "url|size|sha1|mime|extmetadata",
        }
    )
    data = json.loads(opener(f"{COMMONS_API}?{query}"))
    page = next(iter(data["query"]["pages"].values()))
    if "imageinfo" not in page:
        raise ValueError(f"{reference.title} was not found on Wikimedia Commons")
    info = page["imageinfo"][0]
    meta = info.get("extmetadata", {})
    licence = _plain(meta.get("LicenseShortName", {}).get("value"))
    record = {
        "name": reference.name,
        "title": reference.title,
        "page": info.get("descriptionurl")
        or f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(reference.title)}",
        "file_url": info["url"].split("?")[0],
        "width": info["width"],
        "height": info["height"],
        "bytes": info["size"],
        "sha1": info["sha1"],
        "mime": info.get("mime"),
        "licence": licence,
        "licence_url": _plain(meta.get("LicenseUrl", {}).get("value")) or None,
        "author": _plain(meta.get("Artist", {}).get("value")) or "unknown",
        "credit": _plain(meta.get("Credit", {}).get("value")) or None,
        "attribution_required": _plain(meta.get("AttributionRequired", {}).get("value")) == "true",
        "date": _plain(meta.get("DateTimeOriginal", {}).get("value")) or None,
        "scene": reference.scene,
        "description": reference.description,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not licence.startswith(FREE_LICENCES):
        raise ValueError(f"{reference.title} is licensed {licence!r}, which is not one of the free licences allowed")
    if record["sha1"] != reference.sha1:
        raise ValueError(
            f"{reference.title} changed on Commons (SHA-1 {record['sha1']}, pinned {reference.sha1}); "
            "review the new file and update the pinned SHA-1"
        )
    return record


def fetch(reference: Reference, cache_dir: Path, opener: Opener = http_get) -> tuple[Path, dict]:
    """Download (or reuse) the reference into ``cache_dir`` and verify its SHA-1; returns the file and its record."""
    record = licence_record(reference, opener)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urllib.parse.urlparse(record["file_url"]).path).suffix.lower() or ".jpg"
    path = cache_dir / f"{reference.name}_{reference.sha1[:12]}{suffix}"
    if not path.exists() or sha1_hash(path.read_bytes()).hexdigest() != reference.sha1:
        data = opener(record["file_url"])
        digest = sha1_hash(data).hexdigest()
        if digest != reference.sha1:
            raise ValueError(f"downloaded {reference.title} has SHA-1 {digest}, expected {reference.sha1}")
        path.write_bytes(data)
    record["cached_file"] = str(path)
    return path, record


# --------------------------------------------------------------------------- degradation


def centre_crop(linear: np.ndarray, megapixels: float) -> tuple[np.ndarray, list[int]]:
    """The central part of ``linear`` with the same aspect ratio and at most ``megapixels`` (0 keeps it whole)."""
    height, width = linear.shape[:2]
    if megapixels <= 0 or width * height <= megapixels * 1e6:
        return linear, [0, 0, width, height]
    factor = (megapixels * 1e6 / (width * height)) ** 0.5
    w, h = int(width * factor), int(height * factor)
    x0, y0 = (width - w) // 2, (height - h) // 2
    return np.ascontiguousarray(linear[y0 : y0 + h, x0 : x0 + w]), [x0, y0, x0 + w, y0 + h]


def _quantise8(linear: np.ndarray) -> np.ndarray:
    """8-bit sRGB codes of a linear image: the inputs are 8-bit renders."""
    return np.round(cm.linear_to_srgb(np.clip(linear, 0, 1)) * 255).astype(np.uint8)


def _jpeg(codes: np.ndarray, quality: int) -> np.ndarray:
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(codes, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def degrade(
    reference: np.ndarray, size: tuple[int, int], model: str, params: ReferenceParams, seed: int = 42
) -> tuple[np.ndarray, dict]:
    """Make a source of ``size`` (width, height) from a linear reference; returns 8-bit sRGB codes and a record."""
    width, height = size
    scale = reference.shape[1] / width
    if model == "area":
        low = baselines.downsample_area(reference, width, height)
        return _quantise8(low), {
            "model": "area",
            "steps": ["area average over each source pixel in linear light", "8-bit sRGB"],
        }
    if model == "camera":
        sigma = params.optical_blur_sigma * scale
        blurred = (
            np.stack([cv2.GaussianBlur(reference[..., c], (0, 0), sigma) for c in range(3)], axis=-1)
            if sigma > 0
            else reference
        )
        low = baselines.downsample_area(np.ascontiguousarray(blurred, dtype=np.float32), width, height)
        del blurred
        sensor = sensor_simulation.SensorModel(shot_noise_enabled=True, read_noise_enabled=True)
        developed = sensor_simulation.develop(sensor_simulation.simulate(low, sensor, seed=seed).dn, sensor, "malvar")
        codes = _jpeg(_quantise8(developed), params.jpeg_quality)
        return codes, {
            "model": "camera",
            "steps": [
                f"Gaussian optical blur, sigma {params.optical_blur_sigma:g} source px",
                "area sampling",
                "Phase 5 synthetic sensor: RGGB CFA, shot and read noise, 14-bit ADC, Malvar demosaic",
                "8-bit sRGB",
                f"JPEG quality {params.jpeg_quality}",
            ],
        }
    if model == "ai_render":
        low = baselines.resample(reference, width, height, "lanczos")
        codes = _quantise8(low)
        if params.ai_smoothing > 0:
            smoothed = cv2.bilateralFilter(codes, 5, 12 * params.ai_smoothing + 1, 3)
            codes = np.round(codes + params.ai_smoothing * (smoothed.astype(np.float32) - codes)).astype(np.uint8)
        if params.ai_sharpening > 0:
            blur = cv2.GaussianBlur(codes.astype(np.float32), (0, 0), 1.0)
            codes = np.clip(np.round(codes + params.ai_sharpening * (codes - blur)), 0, 255).astype(np.uint8)
        codes = _jpeg(codes, AI_RENDER_JPEG_QUALITY)
        return codes, {
            "model": "ai_render",
            "steps": [
                "Lanczos-3 reduction in linear light (sharp, no optical blur)",
                f"edge-preserving smoothing of fine texture, strength {params.ai_smoothing:g}",
                f"unsharp mask, amount {params.ai_sharpening:g}, sigma 1 source px",
                "8-bit sRGB",
                f"JPEG quality {AI_RENDER_JPEG_QUALITY}",
            ],
        }
    raise ValueError(f"unknown degradation model {model!r}")


def source_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(8, round(width / scale)), max(8, round(height / scale))


# --------------------------------------------------------------------------- the run


@dataclass
class ReferenceResult:
    run_id: str
    manifest: dict
    manifest_path: Path
    report_path: Path
    reports_dir: Path


def run_reference_test(
    config: EvaluationConfig,
    project_root: str | Path,
    *,
    notebook_version: str | None = None,
    run_id: str | None = None,
    opener: Opener = http_get,
    sr_runner_factory: Callable | None = None,
    log: Callable[[str], None] = print,
) -> ReferenceResult:
    """Run E10 on every configured reference, degradation and scale; write the manifest, report and figures.

    ``sr_runner_factory(scene, record, figures_dir, stage)`` builds a scene's SR runner (tests inject a stand-in);
    by default it is the Phase 2 path, and SR configurations are skipped with the reason when it is unavailable.
    """
    from .evaluation_report import write_reference_report

    experiment = config.experiment
    params = config.reference
    project_root = Path(project_root).resolve()
    inputs_root, output_root, reports_root = resolve_roots(experiment, project_root)
    cache_dir = Path(params.cache_dir) if params.cache_dir else inputs_root / "reference_downloads"
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    run_id = run_id or f"{started_at.strftime('%Y%m%dT%H%M%SZ')}_reference_test"
    reports_dir = reports_root / run_id
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    monitor = environment.MemoryMonitor().start()
    stages: list[StageRecord] = []
    manifest: dict = {
        "schema": SCHEMA,
        "phase": PHASE,
        "notice": NOTICE,
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "package": {
            "name": "synthetic_raw_lab",
            "version": __version__,
            "git": environment.git_revision(Path(__file__).parent) or environment.git_revision(project_root),
        },
        "notebook_version": notebook_version,
        "config": config.to_dict(),
        "runtime": environment.inspect_runtime(output_root),
        "status": "running",
        "references": [],
        "cases": [],
    }

    def stage(record: StageRecord):
        return _stage(record, stages, monitor, log)

    def finish() -> Path:
        monitor.stop()
        manifest["stages"] = [s.to_dict() for s in stages]
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest["total_seconds"] = round(time.perf_counter() - started, 3)
        manifest["memory"] = {
            "pipeline_peak_rss_mb": round(monitor.peak_rss / environment.MB, 1),
            "process_lifetime_peak_rss_mb": environment.process_peak_rss_mb(),
        }
        path = reports_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        write_reference_report(manifest, reports_dir / "report.md")
        log(f"report: {reports_dir / 'report.md'}")
        return path

    try:
        for entry in params.references:
            reference = resolve(entry)
            with stage(StageRecord(f"fetch_{reference.name}", PRESERVATION, {"title": reference.title}, "-", "file")):
                path, licence = fetch(reference, cache_dir, opener)
                manifest["references"].append(licence)
                log(f"{reference.name}: {licence['licence']}, {licence['author']} ({licence['page']})")
            with stage(StageRecord(f"decode_{reference.name}", PRESERVATION, {}, "file", "float32 linear")) as record:
                image = ingestion.ingest(
                    path, inputs_root, "reference", source_color_space=experiment.source_color_space
                )
                truth, box = centre_crop(image.linear, params.crop_megapixels)
                record.notes = [f"{image.width}x{image.height}, centre crop {box}", *image.warnings]
                licence["crop_box"] = box
                del image
            for scale in params.scales:
                for model in params.degradations:
                    _run_case(
                        reference,
                        truth,
                        scale,
                        model,
                        config,
                        project_root,
                        manifest,
                        figures_dir,
                        stage,
                        log,
                        sr_runner_factory,
                    )
                    gc.collect()
            del truth
            gc.collect()
        manifest["status"] = "complete"
    except StageError as error:
        manifest["status"] = f"failed in stage {error.stage}"
        manifest["error"] = {"stage": error.stage, "message": str(error), "traceback": traceback.format_exc()}
        error.manifest_path = finish()
        raise
    except Exception as error:
        last = stages[-1].name if stages else "start"
        wrapped = StageError(f"unlogged step after {last}", error, None)
        manifest["status"] = f"failed in stage {wrapped.stage}"
        manifest["error"] = {"stage": wrapped.stage, "message": str(wrapped), "traceback": traceback.format_exc()}
        wrapped.manifest_path = finish()
        raise wrapped from error
    manifest_path = finish()
    return ReferenceResult(run_id, manifest, manifest_path, reports_dir / "report.md", reports_dir)


def _run_case(reference, truth, scale, model, config, project_root, manifest, figures_dir, stage, log, factory):
    experiment = config.experiment
    params = config.reference
    height, width = truth.shape[:2]
    size = source_size(width, height, scale)
    key = f"{reference.name}_{model}_x{scale:g}"
    with stage(
        StageRecord(f"degrade_{key}", SETUP, {"model": model, "scale": scale}, "float32 linear", "8-bit")
    ) as rec:
        codes, degradation = degrade(truth, size, model, params, seed=experiment.random_seed)
        degraded_path = figures_dir / f"{key}_source.png"
        cv2.imwrite(str(degraded_path), cv2.cvtColor(codes, cv2.COLOR_RGB2BGR))
        source = cm.srgb_to_linear(codes.astype(np.float32) / 255).astype(np.float32)
        rec.notes = degradation["steps"]
    analysis = config.universal.analysis
    with stage(StageRecord(f"faces_{key}", MEASUREMENT, {"detector": analysis.detector}, "float32", "boxes")):
        detector = analysis.detector if analysis.face_detection and reference.scene == "human" else "none"
        detections = scene_analysis.observe_faces(source, detector, model_cache_dir(experiment, project_root))
    kind = reference.scene if reference.scene in ("human", "night") else "general"
    scene = Scene(
        kind,
        source,
        (width, height),
        detections,
        scene_params(kind, config, project_root),
        config.universal.human_config(project_root),
        config.universal.face_protection,
        reference=truth,
        codes=codes,
    )
    case = {
        "key": key,
        "reference": reference.name,
        "degradation": degradation,
        "scale": scale,
        "scale_xy": [width / size[0], height / size[1]],
        "source_size": list(size),
        "output_size": [width, height],
        "faces": len(detections.faces),
        "degraded_source": str(degraded_path),
        "sr": {},
    }
    unavailable = ""
    runner = None
    if not config.universal.sr_enabled:
        unavailable = "super-resolution is disabled in the universal profile"
    else:
        try:
            if factory is not None:
                runner = factory(scene, case["sr"], figures_dir, stage)
            else:
                runner = phase2_sr_runner(config, scene.target(), project_root, case["sr"], figures_dir, stage, log)
        except Exception as error:
            unavailable = f"super-resolution unavailable: {error}"
            log(unavailable)
    engine = Engine(scene, config, sr_runner=runner, sr_unavailable=unavailable, log=log)
    crops = review_crops(scene, config.ablation.review_crop_count, max(32, config.ablation.review_crop_px // 2))
    with stage(
        StageRecord(f"reconstruct_{key}", MEASUREMENT, {"configurations": len(params.configurations)}, "-", "-")
    ):
        records, tiles = run_configurations(engine, params.configurations, crops, log=log)
    case["configurations"] = records
    case["figures"] = write_review_figures(
        scene, crops, tiles, records, figures_dir, prefix=f"{key}_", seed=experiment.random_seed
    )
    case["review_crops"] = [c.to_dict() for c in crops]
    manifest["cases"].append(case)
    engine.cache.clear()


def summary_rows(manifest: dict) -> list[dict]:
    """One row per case and configuration: accuracy against ground truth and the gain over Lanczos."""
    rows = []
    for case in manifest.get("cases", []):
        for record in case.get("configurations", []):
            if "reference" not in record:
                continue
            ref = record["reference"]
            gain = record.get("vs_lanczos", {})
            rows.append(
                {
                    "case": case["key"],
                    "reference": case["reference"],
                    "degradation": case["degradation"]["model"],
                    "scale": case["scale"],
                    "configuration": record["name"],
                    "label": record["label"],
                    "psnr_db": ref["psnr_db"],
                    "ssim": ref["ssim"],
                    "lowband_psnr_db": ref["lowband_psnr_db"],
                    "hf_nmse": ref.get("hf_nmse"),
                    "hf_correlation": ref.get("hf_correlation"),
                    "hf_energy_ratio": ref.get("hf_energy_ratio"),
                    "shadow_hf_nmse": ref.get("shadow", {}).get("hf_nmse"),
                    "face_psnr_db": _mean([f["psnr_db"] for f in ref.get("faces", [])]),
                    "face_hf_nmse": _mean([f["hf_nmse"] for f in ref.get("faces", []) if f.get("hf_nmse") is not None]),
                    "psnr_gain_db": gain.get("reference_psnr_db"),
                    "ssim_gain": gain.get("reference_ssim"),
                    "seconds": record["seconds"]["total"],
                }
            )
    return rows


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None
