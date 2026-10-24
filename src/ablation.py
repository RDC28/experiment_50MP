"""Phase 6 ablation (PRD 24 E02, E07, E09; PRD 31 matrix; PRD 28.3 optimisation sweep).

One input is processed through a list of named configurations that switch the pipeline's stages on and off one
at a time, and every configuration is measured the same way:

* **upscale**: bicubic, Lanczos, or a PRD 10.2 SR mode (Phase 2 backend; each mode runs once and is reused);
* **specialised processing**: the scene's own processing, exactly as notebooks 01 to 03 run it (human: region
  detail of :mod:`crowd_processing`; night: the notebook 02 Layer 5B profile; general: the notebook 03
  universal Layer 5B profile);
* **tone**: only the tone, highlight and colour part of that Layer 5B profile (E09 "SR plus tone mapping");
* **face handling**: the notebook 03 face masks keep detail-changing operations off faces, and a human scene
  adds the notebook 01 identity recovery (PRD 11.6);
* **grain** and **optics** (light bloom, the only optical effect PRD 13.3 allows by default): simulated, added
  through the Layer 5B engine so they are the same operations notebook 02 offers;
* **sensor**: the Phase 5 synthetic CFA, noise, quantisation and demosaic round trip (PRD 14, E08).

Every configuration records the standard measurements of notebook 03 (detail energy, source consistency,
clipping), its runtime and peak memory, and, when a ground-truth reference is supplied (experiment E10, see
:mod:`reference_test`), full-reference accuracy including how much of the detail above the source sampling
limit is real (:func:`reference_metrics`).

The ablation run (:func:`run_ablation`) processes a crop of the input at the scale a full 50 MP run of the
whole input uses, so the per-pixel behaviour matches a full run while the run takes minutes on Colab; a crop
size of 0 processes the whole input at the full target.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import (
    __version__,
    baselines,
    bit_depth_study,
    crowd_processing,
    environment,
    exporters,
    face_preservation,
    ingestion,
    quality_metrics,
    region_masks,
    sensor_simulation,
    sr_models,
    subjects,
)
from . import color_management as cm
from . import night_processing as night
from . import universal_pipeline as up
from .evaluation_config import EvaluationConfig
from .pipeline import (
    MEASUREMENT,
    PRESERVATION,
    SETUP,
    StageError,
    StageRecord,
    _stage,
    _without_histograms,
    make_run_id,
    release_sr_models,
    resolve_roots,
    run_sr_mode,
    sr_run_record,
    sr_setup,
)
from .resolution import TargetResolution, compute_target_resolution

SCHEMA = "synthetic_raw_lab.phase6_ablation.manifest/1"
PHASE = "Phase 6 (evaluation and optimisation): full-pipeline ablation"
NOTICE = (
    "Ablation outputs are reconstructions of an AI-generated source for measurement only. Interpolation estimates, "
    "pretrained super-resolution synthesises, and grain, bloom and the sensor round trip simulate; none recovers "
    "camera measurements."
)
UPSCALES = ("bicubic", "lanczos", "sr_conservative", "sr_balanced", "sr_aggressive")
HUMAN_SCENES = ("human",)
SHADOW_LUMA = 0.25  # encoded-sRGB luma below which a reference pixel counts as shadow (PRD 8.3 low luminance)


@dataclass(frozen=True)
class Configuration:
    """One pipeline configuration of the ablation; the flags say which stages run."""

    name: str
    label: str
    upscale: str
    purpose: str
    specialized: bool = False
    tone: bool = False
    face_handling: bool = False
    grain: bool = False
    optics: bool = False
    sensor: bool = False
    removes: str = ""

    @property
    def needs_faces(self) -> bool:
        return self.removes == "face handling"

    @property
    def learned(self) -> bool:
        return self.upscale.startswith("sr_")

    def stages(self) -> list[str]:
        flags = ("specialized", "tone", "face_handling", "grain", "optics", "sensor")
        return [self.upscale, *(name for name in flags if getattr(self, name))]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "purpose": self.purpose,
            "stages": self.stages(),
            "removes": self.removes or None,
        }


_FULL = {"specialized": True, "face_handling": True, "grain": True, "optics": True, "sensor": True}


def _full_minus(name: str, label: str, removes: str, **changes) -> Configuration:
    flags = {**_FULL, **changes}
    upscale = flags.pop("upscale", "sr_conservative")
    return Configuration(name, label, upscale, f"E09: the full pipeline without {removes}", removes=removes, **flags)


CONFIGURATIONS: dict[str, Configuration] = {
    c.name: c
    for c in (
        Configuration("bicubic", "Bicubic", "bicubic", "PRD 31 / E02 classical baseline"),
        Configuration(
            "lanczos", "Lanczos", "lanczos", "PRD 31 / E02 classical baseline; every gain is measured from it"
        ),
        Configuration("sr_conservative", "Conservative SR", "sr_conservative", "E02 / E09 SR only (Mode A)"),
        Configuration("sr_balanced", "Balanced SR", "sr_balanced", "E02 SR only (Mode B)"),
        Configuration("sr_aggressive", "Aggressive SR", "sr_aggressive", "E02 SR only (Mode C, experimental)"),
        Configuration(
            "specialized",
            "SR + specialised",
            "sr_conservative",
            "E09 SR plus the scene's specialised processing (with face handling)",
            specialized=True,
            face_handling=True,
        ),
        Configuration("tone", "SR + tone", "sr_conservative", "E09 SR plus tone mapping only", tone=True),
        Configuration(
            "optics", "SR + optics", "sr_conservative", "E09 SR plus optical simulation (bloom)", optics=True
        ),
        Configuration("grain", "SR + grain", "sr_conservative", "E07 SR with synthetic grain enabled", grain=True),
        Configuration("full", "Full pipeline", "sr_conservative", "E09 the full pipeline", **_FULL),
        _full_minus("full_no_sr", "Full, Lanczos input", "super-resolution", upscale="lanczos"),
        _full_minus("full_no_specialized", "Full, no specialised", "specialised processing", specialized=False),
        _full_minus("full_no_face_handling", "Full, no face handling", "face handling", face_handling=False),
        _full_minus("full_no_grain", "Full, no grain", "grain", grain=False),
        _full_minus("full_no_optics", "Full, no optics", "optical simulation", optics=False),
        _full_minus("full_no_sensor", "Full, no sensor sim.", "the sensor round trip", sensor=False),
        Configuration(
            "full_balanced",
            "Full, balanced SR",
            "sr_balanced",
            "E09 the full pipeline with the balanced SR mode instead of the conservative one",
            **_FULL,
        ),
    )
}
DEFAULT_MATRIX = tuple(CONFIGURATIONS)

# A Layer 5B parameter set that changes nothing (measured identity to float rounding), so grain or bloom can be
# switched on alone through the same engine notebook 02 uses.
NEUTRAL = {
    "shadows": {
        "lift_strength": 0.0,
        "black_point": 0.0,
        "color_preservation": 1.0,
        "banding_reduction": False,
        "dither_amplitude": 0.0,
    },
    "highlights": {"rolloff_strength": 0.0, "halo_suppression": False},
    "color": {
        "chromatic_adaptation": "off",
        "warm_cool_balance": 0.0,
        "saturation_adjustment": 0.0,
        "saturation_luminance_coupling": 0.0,
        "local_color_contrast": 0.0,
        "gamut_mapping": "clip",
    },
    "texture": {
        "architecture_detail_strength": 0.0,
        "sky_detail_strength": 0.0,
        "atmospheric_detail_strength": 0.0,
        "atmospheric_softness": 0.0,
        "sharpening_strength": 0.0,
    },
    "noise": {"denoise_source": 0.0, "synthetic_grain": "off", "grain_amount": 0.0},
    "optics": {"bloom": "off", "bloom_amount": 0.0},
    "failure_rule": {"enabled": False},
}


def neutral_params() -> night.NightParams:
    return night.NightParams.from_dict(NEUTRAL)


def tone_only(params: night.NightParams) -> night.NightParams:
    """The tone, highlight and colour sections of ``params`` (and its failure rule); no detail, noise or optics."""
    return params.replace(texture=NEUTRAL["texture"], noise=NEUTRAL["noise"], optics=NEUTRAL["optics"])


def without_simulation(params: night.NightParams) -> night.NightParams:
    """``params`` with grain and bloom off, so the ablation switches them separately."""
    return params.replace(
        noise={"synthetic_grain": "off", "grain_amount": 0.0}, optics={"bloom": "off", "bloom_amount": 0.0}
    )


# --------------------------------------------------------------------------- the scene


@dataclass
class Scene:
    """The shared inputs of every configuration: a linear source, the output size and the scene's settings."""

    kind: str  # human | night | general
    source: np.ndarray  # float32 linear (H, W, 3)
    size: tuple[int, int]  # output (width, height)
    detections: face_preservation.FaceDetections
    params: night.NightParams  # the scene's Layer 5B profile (night or universal); human scenes use universal
    human: object  # HumanConfig: region strengths, identity thresholds, face tiers
    face_protection: object
    reference: np.ndarray | None = None  # float32 linear ground truth at the output size (E10)
    regions: night.RegionMap | None = None
    masks: region_masks.RegionMasks | None = None
    codes: np.ndarray | None = None  # source integer codes, for the E01 expansion test

    def __post_init__(self) -> None:
        if self.kind not in ("human", "night", "general"):
            raise ValueError(f"scene kind must be human, night or general, got {self.kind!r}")
        if self.reference is not None and self.reference.shape[:2] != (self.size[1], self.size[0]):
            raise ValueError("the reference must have the output size")
        self.regions = night.segment_regions(self.source)
        if self.kind in HUMAN_SCENES:
            self.masks = region_masks.estimate_masks(self.source, self.detections)

    @property
    def source_size(self) -> tuple[int, int]:
        return self.source.shape[1], self.source.shape[0]

    def target(self) -> TargetResolution:
        width, height = self.size
        return TargetResolution(*self.source_size, width, height, width * height, "evaluation")


def scene_kind(category: str) -> str:
    """The ablation scene kind for a :mod:`scene_analysis` category (a night scene with faces is a night scene)."""
    return {"human": "human", "night": "night", "night_with_faces": "night"}.get(category, "general")


def scene_params(kind: str, config: EvaluationConfig, project_root: Path) -> night.NightParams:
    universal = config.universal
    if kind == "night":
        return universal.night_config(project_root).night
    return universal.processing


# --------------------------------------------------------------------------- the engine

SRRunner = Callable[[str, np.ndarray, int, int], tuple[np.ndarray, dict]]


@dataclass
class Upscaled:
    image: np.ndarray
    seconds: float
    summary: dict = field(default_factory=dict)


class Engine:
    """Runs configurations on one :class:`Scene`; each upscale runs once and is reused by every configuration.

    ``sr_runner(mode, linear, width, height) -> (image, Phase 2 mode summary)`` performs SR; by default it is the
    Phase 2 per-mode path (:func:`pipeline.run_sr_mode`), and ``None`` with SR unavailable marks SR
    configurations as skipped with the reason.
    """

    def __init__(
        self,
        scene: Scene,
        config: EvaluationConfig,
        *,
        sr_runner: SRRunner | None = None,
        sr_unavailable: str = "",
        log: Callable[[str], None] = print,
    ):
        self.scene = scene
        self.config = config
        self.sr_runner = sr_runner
        self.sr_unavailable = sr_unavailable
        self.log = log
        self.cache: dict[str, Upscaled] = {}
        self.failed: dict[str, str] = {}

    def skip_reason(self, configuration: Configuration) -> str:
        if configuration.needs_faces and not self.scene.detections.faces:
            return "no faces observed, so face handling changes nothing"
        if configuration.learned and self.sr_runner is None:
            return self.sr_unavailable or "super-resolution is unavailable"
        return ""

    def upscale(self, name: str) -> Upscaled:
        """The cached upscale; a failed one is not retried, so every configuration using it reports the cause."""
        if name in self.failed:
            raise RuntimeError(f"{name} failed earlier in this run: {self.failed[name]}")
        if name not in self.cache:
            width, height = self.scene.size
            started = time.perf_counter()
            try:
                if name in ("bicubic", "lanczos"):
                    upscaled = Upscaled(baselines.resample(self.scene.source, width, height, name), 0.0)
                else:
                    image, summary = self.sr_runner(name.removeprefix("sr_"), self.scene.source, width, height)
                    upscaled = Upscaled(image, 0.0, summary)
            except Exception as error:
                self.failed[name] = f"{type(error).__name__}: {error}"
                raise
            upscaled.seconds = time.perf_counter() - started
            self.cache[name] = upscaled
        return self.cache[name]

    def engine_params(self, configuration: Configuration) -> night.NightParams | None:
        """The Layer 5B parameters a configuration needs, or ``None`` when it needs no engine pass."""
        scene = self.scene
        params = neutral_params()
        if configuration.specialized and scene.kind not in HUMAN_SCENES:
            params = without_simulation(scene.params)
        elif configuration.tone:
            params = tone_only(scene.params)
        ablation = self.config.ablation
        if configuration.grain:
            params = params.replace(noise={"synthetic_grain": "optional", "grain_amount": ablation.grain_amount})
        if configuration.optics:
            params = params.replace(optics={"bloom": "optional_subtle", "bloom_amount": ablation.bloom_amount})
        return None if params.to_dict() == neutral_params().to_dict() else params

    def process(self, configuration: Configuration) -> tuple[np.ndarray, dict]:
        """Apply one configuration; returns the float32 linear output and what each stage did."""
        scene = self.scene
        upscaled = self.upscale(configuration.upscale)
        image = upscaled.image.copy()
        notes: list[str] = []
        applied: list[dict] = [{"stage": configuration.upscale, "seconds": upscaled.seconds}]

        def timed(name: str, function):
            nonlocal image
            started = time.perf_counter()
            image = function(image)
            applied.append({"stage": name, "seconds": time.perf_counter() - started})

        if configuration.specialized and scene.kind in HUMAN_SCENES:
            timed(
                "human_region_detail",
                lambda im: crowd_processing.enhance_regions(im, scene.masks, scene.human.regions).linear,
            )
        params = self.engine_params(configuration)
        if params is not None:
            protect = configuration.face_handling and bool(scene.detections.faces)
            timed("layer_5b", lambda im: self._layer_5b(im, params, protect, notes))
        if configuration.face_handling and scene.kind in HUMAN_SCENES and scene.detections.faces:

            def recover(im):
                recovery = face_preservation.preserve_face_identity(
                    scene.source,
                    im,
                    scene.detections,
                    thresholds=scene.human.identity,
                    tier_config=scene.human.face_tiers,
                    skin_mask=scene.masks.skin_samples,
                )
                notes.append(f"identity recovery: {len(recovery.events)} face event(s)")
                return recovery.linear

            timed("identity_recovery", recover)
        if configuration.sensor:
            timed("sensor_round_trip", self._sensor_round_trip)
        return image, {"applied": applied, "notes": notes}

    def _layer_5b(self, image: np.ndarray, params: night.NightParams, protect: bool, notes: list[str]) -> np.ndarray:
        scene = self.scene
        seed = self.config.experiment.random_seed
        masks = []
        patches = []
        if protect:
            masks = up.face_masks(scene.detections, scene.source_size, scene.size, scene.face_protection)
            safe = image.copy()
            night.process_night(safe, scene.regions, scene.source, up.face_safe(params), seed=seed, out=safe)
            patches = [safe[m["box"][1] : m["box"][3], m["box"][0] : m["box"][2]].copy() for m in masks]
            del safe
        result = night.process_night(image, scene.regions, scene.source, params, seed=seed, out=image)
        covered = np.zeros(image.shape[:2], dtype=bool)
        for mask, patch in zip(masks, patches, strict=True):
            x0, y0, x1, y1 = mask["box"]
            weight = up.mask_weight(mask["box"], mask["feather_px"], scene.size)[..., None]
            image[y0:y1, x0:x1] += weight * (patch - image[y0:y1, x0:x1])
            covered[y0:y1, x0:x1] = True
        if masks:
            notes.append(f"face masks cover {covered.mean():.0%} of the frame (no detail, grain or bloom there)")
        if params.failure_rule.enabled:
            notes.append(f"failure rule: {len(result.fallbacks)} fallback block(s) ({result.fallback_fraction:.2%})")
        notes.append("Layer 5B steps: " + ", ".join(s["name"] for s in result.stages if s["enabled"]))
        return image

    def _sensor_round_trip(self, image: np.ndarray) -> np.ndarray:
        ablation = self.config.ablation
        model = ablation.sensor_model()
        mosaic = sensor_simulation.simulate(np.clip(image, 0, None), model, seed=self.config.experiment.random_seed)
        return sensor_simulation.develop(mosaic.dn, model, ablation.demosaic)


def phase2_sr_runner(
    config: EvaluationConfig,
    target: TargetResolution,
    project_root: Path,
    record: dict,
    figures_dir: Path,
    stage,
    log,
    *,
    experiment=None,
) -> SRRunner:
    """Phase 2's per-mode SR path (:func:`pipeline.run_sr_mode`) bound to one scene; one network loaded at a time.

    ``record`` receives ``super_resolution`` (per-mode summaries) and ``models`` like a notebook manifest.
    Raises with Phase 2's install hint when PyTorch or spandrel is missing.
    """
    experiment = experiment or config.experiment
    weights_dir = sr_setup(experiment, project_root)
    record.setdefault("super_resolution", sr_run_record(experiment, target))
    record.setdefault("models", [])
    loaded: dict = {}

    def run(mode: str, linear: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
        if (width, height) != (target.width, target.height):
            raise ValueError(f"SR renders the scene target {target.width}x{target.height}, not {width}x{height}")
        try:
            image = run_sr_mode(mode, linear, target, experiment, weights_dir, loaded, record, figures_dir, stage, log)
        finally:
            release_sr_models(loaded, experiment, ())
            gc.collect()
        return image, record["super_resolution"]["modes"][mode]

    return run


# --------------------------------------------------------------------------- measurements


def encoded_luma(linear: np.ndarray) -> np.ndarray:
    return quality_metrics.luma(cm.linear_to_srgb(np.clip(linear, 0, 1))).astype(np.float32)


def low_band(y: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    """The part of a luma image the source sampling grid can hold: area reduction to the source size, then Keys
    bicubic back to the full size."""
    height, width = y.shape
    reduced = baselines.downsample_area(y[..., None], *source_size)
    return baselines.resample(np.ascontiguousarray(reduced), width, height, "bicubic")[..., 0]


def _band_metrics(hf_out: np.ndarray, hf_ref: np.ndarray) -> dict:
    ref_energy = float(np.mean(np.square(hf_ref, dtype=np.float64)))
    out_energy = float(np.mean(np.square(hf_out, dtype=np.float64)))
    if ref_energy <= 0:
        return {"hf_nmse": None, "hf_correlation": None, "hf_energy_ratio": None}
    error = float(np.mean(np.square(hf_out - hf_ref, dtype=np.float64)))
    denominator = np.sqrt(ref_energy * out_energy)
    correlation = float(np.mean(hf_out.astype(np.float64) * hf_ref) / denominator) if denominator > 0 else 0.0
    return {"hf_nmse": error / ref_energy, "hf_correlation": correlation, "hf_energy_ratio": out_energy / ref_energy}


def reference_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    source_size: tuple[int, int],
    face_boxes: list[tuple[int, int, int, int]] = (),
) -> dict:
    """Full-reference accuracy of a reconstruction against ground truth at the same size (experiment E10).

    Besides PSNR, SSIM, MS-SSIM, CIEDE2000 and edge displacement (``quality_metrics.source_consistency`` with the
    reference in place of the source), the luma is split at the source sampling limit (:func:`low_band`):

    * ``lowband_psnr_db``: agreement of what the source could hold;
    * ``hf_nmse``: error in the band above the source sampling limit, relative to that band's energy in the
      reference. 1 is what adding no detail scores; below 1 the reconstruction recovered real detail, above 1
      the detail it added is wrong more than right;
    * ``hf_correlation``: correlation of that band with the reference's (0 for no real detail);
    * ``hf_energy_ratio``: how much of the reference's above-limit energy the reconstruction has (1 is equal);
    * ``shadow``: the same band metrics on pixels whose reference luma is below 0.25 (dark regions);
    * ``faces``: PSNR, SSIM and the band metrics inside each face box (output pixels).
    """
    full, _ = quality_metrics.source_consistency(reference, candidate)
    y_ref, y_out = encoded_luma(reference), encoded_luma(candidate)
    low_ref, low_out = low_band(y_ref, source_size), low_band(y_out, source_size)
    hf_ref, hf_out = y_ref - low_ref, y_out - low_out
    shadow = low_ref < SHADOW_LUMA
    record = {
        "psnr_db": full["psnr_db"],
        "ssim": full["ssim"],
        "ms_ssim": full["ms_ssim"]["value"],
        "delta_e_mean": full["color_difference_ciede2000"]["mean"],
        "edge_displacement_mean_px": full["edge_displacement"]["mean_px"],
        "lowband_psnr_db": quality_metrics.psnr(low_ref, low_out),
        **_band_metrics(hf_out, hf_ref),
        "shadow": {
            "fraction": float(shadow.mean()),
            **(_band_metrics(hf_out[shadow], hf_ref[shadow]) if shadow.any() else {}),
        },
        "faces": [],
    }
    for x0, y0, x1, y1 in face_boxes:
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        ref_tile, out_tile = y_ref[y0:y1, x0:x1], y_out[y0:y1, x0:x1]
        record["faces"].append(
            {
                "box": [x0, y0, x1, y1],
                "psnr_db": quality_metrics.psnr(ref_tile, out_tile),
                "ssim": quality_metrics.ssim(ref_tile[..., None], out_tile[..., None]),
                **_band_metrics(hf_out[y0:y1, x0:x1], hf_ref[y0:y1, x0:x1]),
            }
        )
    return record


def face_output_boxes(scene: Scene) -> list[tuple[int, int, int, int]]:
    sx, sy = scene.size[0] / scene.source_size[0], scene.size[1] / scene.source_size[1]
    return [
        (round(f.x * sx), round(f.y * sy), round((f.x + f.width) * sx), round((f.y + f.height) * sy))
        for f in scene.detections.faces
    ]


# --------------------------------------------------------------------------- running configurations


def review_crops(scene: Scene, count: int, size: int) -> list:
    """Identical review crops for every configuration: faces first, else the centre of interest, then detail."""
    encoded = cm.linear_to_srgb(np.clip(scene.source, 0, 1))
    width, height = scene.source_size
    faces = tuple(subjects.Face(f.x, f.y, f.width, f.height) for f in scene.detections.faces)
    point, basis = (None, "")
    if not faces:
        point, basis = subjects.center_of_interest(quality_metrics.luma(encoded), (width, height))
    analysis = subjects.SubjectAnalysis(
        width, height, faces, scene.detections.method, scene.detections.description, point, basis
    )
    from . import comparison

    return comparison.select_crops(encoded, size, count, subjects=analysis)


def run_configurations(
    engine: Engine,
    names,
    crops: list,
    *,
    stage=None,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    """Run and measure ``names`` on the engine's scene; returns one record per configuration and the uint8 review
    tiles (crop name -> configuration name -> tile)."""
    scene = engine.scene
    boxes = {c.name: c.output_box(scene.source_size, scene.size) for c in crops}
    tiles: dict[str, dict[str, np.ndarray]] = {c.name: {} for c in crops}
    faces = face_output_boxes(scene)
    lanczos = None
    records = []
    for name in names:
        configuration = CONFIGURATIONS[name]
        record = {**configuration.to_dict()}
        reason = engine.skip_reason(configuration)
        if reason:
            record["skipped"] = reason
            log(f"{name}: skipped ({reason})")
            records.append(record)
            continue
        monitor = environment.MemoryMonitor().start()
        try:
            image, detail = engine.process(configuration)
            upscale = engine.cache[configuration.upscale]
            record.update(detail)
            # The upscale runs once and is shared; it is counted in every configuration's standalone runtime.
            total = sum(s["seconds"] for s in detail["applied"])
            record["seconds"] = {"upscale": upscale.seconds, "processing": total - upscale.seconds, "total": total}
            peak_device = (upscale.summary.get("model") or {}).get("peak_device_memory_mb")
            record["peak_device_mb"] = peak_device
            if upscale.summary.get("seams"):
                record["seam_gate_passed"] = bool(upscale.summary["seams"].get("passed"))
            record["measurements"] = up.measure(image, scene.source)
            if scene.reference is not None:
                record["reference"] = reference_metrics(image, scene.reference, scene.source_size, faces)
            for crop_name, (x0, y0, x1, y1) in boxes.items():
                tile, _ = cm.encode_srgb_uint16(image[y0:y1, x0:x1])
                tiles[crop_name][name] = cm.srgb_uint16_to_uint8(tile)
            del image
        except Exception as error:  # one failing configuration must not lose the others (PRD 22.3)
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
            log(f"{name}: FAILED ({record['error']})")
        finally:
            monitor.stop()
            record["peak_rss_mb"] = round(monitor.peak_rss / environment.MB, 1)
            gc.collect()
        if name == "lanczos" and "measurements" in record:
            lanczos = record
        if "measurements" in record:
            psnr = record["measurements"]["source_consistency"]["psnr_db"]
            log(f"{name}: {record['seconds']['total']:.1f} s, source PSNR {psnr:.2f} dB")
        records.append(record)
    if lanczos is not None:
        for record in records:
            if "measurements" in record:
                record["vs_lanczos"] = versus(record, lanczos)
    return records, tiles


def versus(record: dict, base: dict) -> dict:
    """Differences of a configuration's measurements from another's (``record - base``)."""
    a, b = record["measurements"], base["measurements"]
    out = {
        "detail_gain": a["detail_energy"] / b["detail_energy"] - 1 if b["detail_energy"] > 0 else None,
        "source_psnr_db": a["source_consistency"]["psnr_db"] - b["source_consistency"]["psnr_db"],
        "source_delta_e_mean": a["source_consistency"]["delta_e_mean"] - b["source_consistency"]["delta_e_mean"],
        "clipped_fraction": a["clipped_fraction"] - b["clipped_fraction"],
    }
    if "reference" in record and "reference" in base:
        ra, rb = record["reference"], base["reference"]
        out["reference_psnr_db"] = ra["psnr_db"] - rb["psnr_db"]
        out["reference_ssim"] = ra["ssim"] - rb["ssim"]
        if ra.get("hf_nmse") is not None and rb.get("hf_nmse") is not None:
            out["hf_nmse"] = ra["hf_nmse"] - rb["hf_nmse"]
    return out


def stage_contributions(records: list[dict]) -> list[dict]:
    """E09: what each stage adds to the full pipeline, from the full pipeline minus that stage.

    A positive ``reference_psnr_db`` means the stage brings the output closer to ground truth; a negative one
    means it measurably harms accuracy. ``detail_gain`` is the stage's change in detail energy.
    """
    by_name = {r["name"]: r for r in records if "measurements" in r}
    full = by_name.get("full")
    if full is None:
        return []
    rows = []
    for name, configuration in CONFIGURATIONS.items():
        if not configuration.removes or name not in by_name:
            continue
        rows.append({"stage": configuration.removes, "without": name, **versus(full, by_name[name])})
    return rows


# --------------------------------------------------------------------------- figures


def _label_band(text: str, width: int, height: int = 26) -> np.ndarray:
    band = np.full((height, width, 3), 24, np.uint8)
    scale = 0.45 if width >= 180 else 0.35
    cv2.putText(band, text[:48], (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return band


def review_sheet(rows: list[tuple[str, list[np.ndarray]]], columns: list[str], tile: int = 256) -> np.ndarray:
    """A grid of identical crops: one row per labelled entry, one column per crop, tiles resized to ``tile``
    (nearest-neighbour when enlarging, so source pixels stay visible)."""
    gutter, label_w = 6, 190
    width = label_w + len(columns) * (tile + gutter)
    header = np.full((26, width, 3), 24, np.uint8)
    for index, name in enumerate(columns):
        x = label_w + index * (tile + gutter)
        header[:, x : x + tile] = _label_band(name, tile)
    bands = [header]
    for label, tiles in rows:
        row = np.full((tile + gutter, width, 3), 24, np.uint8)
        text = _label_band(label, label_w)
        top = (tile - text.shape[0]) // 2
        row[top : top + text.shape[0], :label_w] = text
        for index, t in enumerate(tiles):
            x = label_w + index * (tile + gutter)
            interpolation = cv2.INTER_NEAREST if t.shape[0] < tile else cv2.INTER_AREA
            row[:tile, x : x + tile] = cv2.resize(t, (tile, tile), interpolation=interpolation)
        bands.append(row)
    return np.concatenate(bands, axis=0)


def write_review_figures(
    scene: Scene,
    crops: list,
    tiles: dict[str, dict[str, np.ndarray]],
    records: list[dict],
    figures_dir: Path,
    prefix: str = "",
    seed: int = 42,
) -> dict:
    """The labelled review sheet (source, reference if any, every configuration) and the blind sheet (PRD 26.3):
    the same configurations in shuffled order under letters, with the key stored separately."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    columns = [c.name for c in crops]
    source = cm.linear_to_srgb(np.clip(scene.source, 0, 1))
    rows = [
        (
            "source (nearest)",
            [np.round(source[c.y : c.y + c.height, c.x : c.x + c.width] * 255).astype(np.uint8) for c in crops],
        )
    ]
    if scene.reference is not None:
        boxes = [c.output_box(scene.source_size, scene.size) for c in crops]
        reference = [
            cm.srgb_uint16_to_uint8(cm.encode_srgb_uint16(scene.reference[y0:y1, x0:x1])[0]) for x0, y0, x1, y1 in boxes
        ]
        rows.append(("reference (truth)", reference))
    done = [r for r in records if all(r["name"] in tiles[c] for c in columns)]
    rows += [(r["label"], [tiles[c][r["name"]] for c in columns]) for r in done]
    labelled = exporters.write_review_image(figures_dir / f"{prefix}review_sheet.png", review_sheet(rows, columns))
    order = [r["name"] for r in done]
    random.Random(seed).shuffle(order)
    letters = [chr(ord("A") + i) if i < 26 else f"Z{i}" for i in range(len(order))]
    blind_rows = [
        (f"method {letter}", [tiles[c][name] for c in columns]) for letter, name in zip(letters, order, strict=True)
    ]
    blind = exporters.write_review_image(figures_dir / f"{prefix}blind_sheet.png", review_sheet(blind_rows, columns))
    key = dict(zip(letters, order, strict=True))
    (figures_dir / f"{prefix}blind_key.json").write_text(json.dumps(key, indent=2))
    return {"review_sheet": str(labelled), "blind_sheet": str(blind), "blind_key": key}


# --------------------------------------------------------------------------- the PRD 28.3 sweep and E01


def performance_sweep(
    scene: Scene,
    engine: Engine,
    config: EvaluationConfig,
    project_root: Path,
    figures_dir: Path,
    stage,
    log,
) -> list[dict]:
    """Conservative SR under each ``ablation.performance_sweep`` setting (tile size, overlap, precision,
    device): runtime, peak device memory, seam gate and how far the output moves from the default settings."""
    if engine.sr_runner is None:
        return [{"skipped": engine.sr_unavailable or "super-resolution is unavailable"}]
    default = engine.upscale("sr_conservative")
    tiling = default.summary.get("tiling") or {}
    rows = [
        {
            "setting": "default",
            "seconds": default.seconds,
            "inference_seconds": tiling.get("inference_seconds"),
            "tile_size": tiling.get("tile_size"),
            "tile_overlap": tiling.get("tile_overlap"),
            "precision": (default.summary.get("model") or {}).get("precision"),
            "peak_device_mb": (default.summary.get("model") or {}).get("peak_device_memory_mb"),
            "seam_gate_passed": (default.summary.get("seams") or {}).get("passed"),
        }
    ]
    reference = encoded_luma(default.image)
    base = config.experiment
    for entry in config.ablation.performance_sweep:
        changes = {
            "sr_tile_size": entry.get("tile_size", base.sr_tile_size),
            "sr_tile_overlap": entry.get("tile_overlap", base.sr_tile_overlap),
            "sr_precision": entry.get("precision", base.sr_precision),
            "sr_device": entry.get("device", base.sr_device),
        }
        setting = ", ".join(f"{k}={v}" for k, v in entry.items())
        row: dict = {"setting": setting}
        try:
            record: dict = {}
            runner = phase2_sr_runner(
                config,
                scene.target(),
                project_root,
                record,
                figures_dir / "sweep",
                stage,
                log,
                experiment=base.replace(**changes),
            )
            (figures_dir / "sweep").mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            image, summary = runner("conservative", scene.source, *scene.size)
            tiling = summary.get("tiling") or {}
            difference = np.abs(encoded_luma(image) - reference)
            row.update(
                seconds=time.perf_counter() - started,
                inference_seconds=tiling.get("inference_seconds"),
                tile_size=tiling.get("tile_size"),
                tile_overlap=tiling.get("tile_overlap"),
                precision=(summary.get("model") or {}).get("precision"),
                peak_device_mb=(summary.get("model") or {}).get("peak_device_memory_mb"),
                seam_gate_passed=(summary.get("seams") or {}).get("passed"),
                max_abs_difference_8bit=float(difference.max() * 255),
                psnr_vs_default_db=quality_metrics.psnr(reference, encoded_luma(image)),
            )
            del image
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            log(f"sweep {setting}: FAILED ({row['error']})")
        rows.append(row)
        gc.collect()
    return rows


def bit_depth_evidence(scene: Scene, image: np.ndarray) -> dict:
    """E01 and PRD 14.5 on this input: the source codes' expansion test and the representation study of a
    reconstruction (quantisation, headroom, shadow levels after an identical push, transform drift)."""
    record = {"study": bit_depth_study.run_study(image)}
    if scene.codes is not None and scene.codes.dtype.kind == "u":
        record["expansion"] = bit_depth_study.expansion_information(scene.codes)
    return record


# --------------------------------------------------------------------------- the ablation run


@dataclass
class AblationResult:
    run_id: str
    manifest: dict
    manifest_path: Path
    report_path: Path
    reports_dir: Path


def crop_box(width: int, height: int, size: int, centre: tuple[float, float]) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a square crop of ``size`` (clamped to the image) centred as close to ``centre`` as fits."""
    if size <= 0:
        return 0, 0, width, height
    side_x, side_y = min(size, width), min(size, height)
    x0 = int(round(min(max(centre[0] - side_x / 2, 0), width - side_x)))
    y0 = int(round(min(max(centre[1] - side_y / 2, 0), height - side_y)))
    return x0, y0, x0 + side_x, y0 + side_y


def crop_centre(source_linear: np.ndarray, detections, requested) -> tuple[tuple[float, float], str]:
    height, width = source_linear.shape[:2]
    if requested != "auto":
        return (float(requested[0]), float(requested[1])), "configured"
    if detections.faces:
        face = detections.faces[detections.primary_index or 0]
        return (face.x + face.width / 2, face.y + face.height / 2), "primary face"
    point, basis = subjects.center_of_interest(encoded_luma(source_linear))
    if point is not None:
        return (float(point[0]), float(point[1])), f"centre of interest ({basis})"
    return (width / 2, height / 2), "image centre"


def faces_in_crop(detections, box: tuple[int, int, int, int]) -> face_preservation.FaceDetections:
    """The face observations wholly inside ``box``, in crop coordinates, with the primary subject chosen again."""
    x0, y0, x1, y1 = box
    kept = tuple(
        face_preservation.FaceBox(face.x - x0, face.y - y0, face.width, face.height, face.confidence)
        for face in detections.faces
        if face.x >= x0 and face.y >= y0 and face.x + face.width <= x1 and face.y + face.height <= y1
    )
    return face_preservation.select_primary_subject(dataclasses.replace(detections, faces=kept, primary_index=None))


def run_ablation(
    source_path: str | Path,
    config: EvaluationConfig,
    project_root: str | Path,
    *,
    notebook_version: str | None = None,
    run_id: str | None = None,
    sr_runner: SRRunner | None = None,
    log: Callable[[str], None] = print,
) -> AblationResult:
    """Run the PRD 31 matrix and the E09 ablation on one input and write its manifest, report and figures."""
    from .evaluation_report import write_ablation_report

    experiment = config.experiment
    ablation = config.ablation
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
        "random_seeds": {"random_seed": experiment.random_seed, "used_by": "grain, dither and sensor noise"},
        "config": config.to_dict(),
        "runtime": environment.inspect_runtime(output_root),
        "status": "running",
        "models": [],
    }
    paths: dict[str, Path] = {}

    def stage(record: StageRecord):
        return _stage(record, stages, monitor, log)

    def finish() -> Path | None:
        monitor.stop()
        manifest["stages"] = [s.to_dict() for s in stages]
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest["total_seconds"] = round(time.perf_counter() - started, 3)
        manifest["memory"] = {
            "pipeline_peak_rss_mb": round(monitor.peak_rss / environment.MB, 1),
            "process_lifetime_peak_rss_mb": environment.process_peak_rss_mb(),
        }
        if "reports" not in paths:
            return None
        path = paths["reports"] / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        write_ablation_report(manifest, paths["reports"] / "report.md")
        log(f"report: {paths['reports'] / 'report.md'}")
        return path

    try:
        with stage(StageRecord("ingest_and_validate", PRESERVATION, {}, "source file", "float32 linear")) as record:
            source = ingestion.ingest(
                source_path, inputs_root, experiment.input_category, source_color_space=experiment.source_color_space
            )
            record.notes = list(source.warnings)
        run_id = run_id or make_run_id(source.sha256, "ablation", started_at)
        reports_dir = reports_root / run_id
        figures_dir = reports_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        paths.update(reports=reports_dir, figures=figures_dir)
        manifest.update(
            run_id=run_id,
            source={
                "original_path": str(source.original_path),
                "sha256": source.sha256,
                "width": source.width,
                "height": source.height,
                "color_space": source.color_space.to_dict(),
            },
            validation=_without_histograms(source.validation),
        )
        with stage(StageRecord("scene_analysis", MEASUREMENT, {}, "float32 linear", "category, faces")) as record:
            analysis, detections = up.analyse_scene(source.linear, config.universal, project_root)
            category = analysis["classification"]["category"]
            kind = scene_kind(category) if ablation.scene == "auto" else ablation.scene
            manifest["scene"] = {"analysis": analysis, "kind": kind, "chosen_by": ablation.scene}
            record.notes = [f"category {category}; ablation scene {kind}"]
        with stage(StageRecord("target_planning", MEASUREMENT, {"megapixels": ablation.target_megapixels}, "-", "-")):
            target = compute_target_resolution(
                source.width,
                source.height,
                ablation.target_megapixels,
                mode=experiment.resolution_mode,
                pixel_count_tolerance=experiment.pixel_count_tolerance,
                dimension_multiple=experiment.dimension_multiple,
                preserve_aspect_ratio=experiment.preserve_aspect_ratio,
                max_aspect_ratio_error=experiment.max_aspect_ratio_error,
            )
        centre, basis = crop_centre(source.linear, detections, ablation.crop_center)
        box = crop_box(source.width, source.height, ablation.crop_size, centre)
        x0, y0, x1, y1 = box
        if ablation.crop_size:
            size = (round((x1 - x0) * target.scale_x), round((y1 - y0) * target.scale_y))
        else:
            size = (target.width, target.height)
        crop_faces = faces_in_crop(detections, box)
        manifest["crop"] = {
            "box": list(box),
            "centre": [round(centre[0], 1), round(centre[1], 1)],
            "centre_basis": basis,
            "output_size": list(size),
            "scale": [target.scale_x, target.scale_y],
            "full_target": target.to_dict(),
            "faces_in_crop": len(crop_faces.faces),
            "note": "processed at the scale a full run of the whole source to the target uses"
            if ablation.crop_size
            else "the whole source at the full target",
        }
        log(f"ablation crop {box} ({basis}) -> {size[0]}x{size[1]} at x{target.scale_x:.4f}; scene {kind}")
        with stage(StageRecord("prepare_scene", SETUP, {"kind": kind}, "float32 linear", "regions, masks")):
            codes = source.codes[y0:y1, x0:x1]
            scene = Scene(
                kind,
                np.ascontiguousarray(source.linear[y0:y1, x0:x1]),
                size,
                crop_faces,
                scene_params(kind, config, project_root),
                config.universal.human_config(project_root),
                config.universal.face_protection,
                codes=codes if codes.shape[-1] == 3 else None,
            )
            del source
            gc.collect()
        sr_unavailable = ""
        if sr_runner is None and config.universal.sr_enabled:
            try:
                sr_runner = phase2_sr_runner(config, scene.target(), project_root, manifest, figures_dir, stage, log)
            except Exception as error:
                sr_unavailable = f"super-resolution unavailable: {error}"
                log(sr_unavailable)
        elif not config.universal.sr_enabled:
            sr_unavailable = "super-resolution is disabled in the universal profile"
            sr_runner = None
        engine = Engine(scene, config, sr_runner=sr_runner, sr_unavailable=sr_unavailable, log=log)
        crops = review_crops(scene, ablation.review_crop_count, ablation.review_crop_px)
        manifest["review_crops"] = [c.to_dict() for c in crops]
        with stage(StageRecord("configurations", MEASUREMENT, {"count": len(ablation.configurations)}, "-", "-")):
            records, tiles = run_configurations(engine, ablation.configurations, crops, log=log)
        manifest["configurations"] = records
        manifest["contributions"] = stage_contributions(records)
        with stage(StageRecord("review_figures", MEASUREMENT, {"crops": len(crops)}, "uint8 tiles", "PNG")):
            manifest["figures"] = write_review_figures(
                scene, crops, tiles, records, figures_dir, seed=experiment.random_seed
            )
        del tiles
        if ablation.performance_sweep:
            with stage(StageRecord("performance_sweep", MEASUREMENT, {}, "-", "-")):
                manifest["performance_sweep"] = performance_sweep(
                    scene, engine, config, project_root, figures_dir, stage, log
                )
        if ablation.bit_depth_study:
            with stage(StageRecord("bit_depth_study", MEASUREMENT, {}, "float32 linear", "metrics")):
                final = "sr_conservative" if "sr_conservative" in engine.cache else "lanczos"
                manifest["bit_depth"] = {"image": final, **bit_depth_evidence(scene, engine.upscale(final).image)}
        engine.cache.clear()
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
    return AblationResult(run_id, manifest, manifest_path, paths["reports"] / "report.md", paths["reports"])


def available_sr_message() -> str:
    """Why SR configurations will be skipped in this runtime, or an empty string when SR can run."""
    return "" if sr_models.available() else "super-resolution needs PyTorch and spandrel (pip install '.[sr]')"


def configuration_rows(names=None) -> list[dict]:
    """Every configuration with its stages, for the notebook's configuration table."""
    return [CONFIGURATIONS[n].to_dict() for n in (names or DEFAULT_MATRIX)]
