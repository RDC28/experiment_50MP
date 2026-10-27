"""PRD_v3 2K-versus-4K source-resolution experiment (``comparison_2k_vs_4k.html``).

``SourceResolutionComparator`` compares two finished runs made from the same composition at two
source resolutions. Every PRD locked setting (camera profile, output geometry, SR/HDR backends,
tone inversion, lens, sensor parameters, colour matrices, seeds, development settings) is asserted
equal as a hard gate; only source resolution may vary. It then reports the final difference map,
edge and texture agreement globally and on named crops, and the exposure-posterior, white-balance,
camera-geometry, HDR-mask and DNG-statistics comparisons. An estimated camera parameter that moves
beyond its tolerance raises ``ESTIMATOR_UNSTABLE``. Analysis reads bounded strips, never a full
50 MP float array. ``UserStudyPairSheet`` is the optional blinded pair-sheet generator; it is never
run automatically.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from synthraw.provenance import SCIENTIFIC_BOUNDARY, SYNTHETIC_CAPTURE_DISCLOSURE
from synthraw.report import _CSS, SUPPORT_MASKS, format_measured, mask_percentages
from synthraw.validation import ValidationCheck, _jsonable, _pearson, block_mean, hard_failures
from synthraw.validation_runner import _open_tiff

COMPARISON_NAME = "comparison_2k_vs_4k.html"
COMPARISON_JSON = "comparison_2k_vs_4k.json"
USER_STUDY_NAME = "user_study_pairs.html"
USER_STUDY_KEY = "user_study_key.json"
ESTIMATOR_UNSTABLE = "ESTIMATOR_UNSTABLE"
DEVELOPED_MASTER = Path("masters") / "developed_50mp_16bit.tif"
# Suggested names for user-supplied crop boxes (PRD 2K/4K table); any name is accepted.
PRD_CROP_NAMES = ("face_skin", "hair", "saree_fabric", "jewellery", "smoke", "bokeh_light")
# Config keys that identify the input rather than a locked setting.
INPUT_CONFIG_KEYS = frozenset({"INPUT_MODE", "INPUT_PATH", "DRIVE_MOUNT_PATH"})
# Sensor-profile fields that follow the (compared, not locked) exposure posterior.
EXPOSURE_DERIVED_FIELDS = frozenset(
    {"applied_exposure_time_s", "applied_exposure_scale", "applied_analogue_gain", "applied_adc_analogue_gain"}
)
GEOMETRY_KEYS = ("width", "height", "active_area", "default_crop_origin", "default_crop_size")
DNG_COLOUR_FIELDS = (
    "Make",
    "Model",
    "UniqueCameraModel",
    "ColorMatrix1",
    "ColorMatrix2",
    "CalibrationIlluminant1",
    "CalibrationIlluminant2",
    "CFAPattern",
    "CFARepeatPatternDim",
    "BlackLevel",
    "WhiteLevel",
    "NoiseProfile",
)
_ADOBE_Y = np.array([0.2973769, 0.6273491, 0.0752741])


@dataclass(frozen=True)
class EstimatorTolerances:
    """How far a hypothetical camera parameter may move between source resolutions before it is unstable."""

    exposure_stops: float = 1.0
    focal_length_rel: float = 0.10
    cct_mired: float = 25.0
    tint: float = 0.10
    as_shot_neutral_rel: float = 0.10


@dataclass(frozen=True)
class CropBox:
    """A named crop in fractions of the output frame: ``(x0, y0, x1, y1)`` with ``0 <= x0 < x1 <= 1``."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if not self.name or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("crop needs a name and finite coordinates")
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            raise ValueError(f"crop {self.name!r} must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")

    @classmethod
    def from_pixels(cls, name: str, box: Sequence[int], width: int, height: int) -> CropBox:
        """Crop from output-pixel coordinates ``(x0, y0, x1, y1)`` on a ``width`` x ``height`` frame."""
        x0, y0, x1, y1 = box
        return cls(name, x0 / width, y0 / height, x1 / width, y1 / height)

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0, x1 = int(math.floor(self.x0 * width)), int(math.ceil(self.x1 * width))
        y0, y1 = int(math.floor(self.y0 * height)), int(math.ceil(self.y1 * height))
        return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def grid_crops(rows: int = 2, cols: int = 3) -> list[CropBox]:
    """Default named crops: an even ``rows`` x ``cols`` grid over the frame."""
    if rows < 1 or cols < 1:
        raise ValueError("grid needs at least one row and column")
    return [
        CropBox(f"grid_r{r}_c{c}", c / cols, r / rows, (c + 1) / cols, (r + 1) / rows)
        for r in range(rows)
        for c in range(cols)
    ]


def parse_crops(crops: Mapping[str, Sequence[float]] | Iterable[CropBox] | None) -> list[CropBox]:
    """User boxes as ``{name: (x0, y0, x1, y1)}`` fractions or ``CropBox`` items; ``None`` gives the grid."""
    if crops is None:
        return grid_crops()
    if isinstance(crops, Mapping):
        return [CropBox(str(name), *map(float, box)) for name, box in crops.items()]
    parsed = list(crops)
    if not all(isinstance(crop, CropBox) for crop in parsed):
        raise TypeError("crops must be a mapping of name to box or CropBox items")
    return parsed


def make_downsampled_source(source: str | Path, out_path: str | Path, *, factor: int = 2) -> Path:
    """Write a Lanczos-downsampled copy of ``source`` (ICC kept) when only one resolution is supplied.

    The copy is derived from the higher-resolution input, so reports label it as such: it isolates
    the effect of source pixel count but is not an independently rendered lower-resolution source.
    """
    from PIL import Image

    if type(factor) is not int or factor < 2:
        raise ValueError("factor must be an integer >= 2")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        icc = image.info.get("icc_profile")
        if image.mode not in ("RGB", "RGBA", "I;16", "I;16B", "L"):
            image = image.convert("RGB")
        size = (max(1, round(image.width / factor)), max(1, round(image.height / factor)))
        small = image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
        options: dict[str, Any] = {"icc_profile": icc} if icc else {}
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            options["quality"] = 95
        small.save(out_path, **options)
    return out_path


def _records(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {item["field"]: item["value"] for item in data if isinstance(item, Mapping) and "field" in item}
    return dict(data)


def _same(first: Any, second: Any) -> bool:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(_same(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second, strict=True))
    if isinstance(first, bool) or isinstance(second, bool) or first is None or second is None:
        return first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-12)
    return first == second


@dataclass
class RunFacts:
    """The metadata of one run that the comparison reads (all from the run tree)."""

    root: Path
    label: str
    config: dict[str, Any]
    geometry: dict[str, Any]
    input_audit: dict[str, Any]
    camera_profile: dict[str, Any]
    sensor_profile: dict[str, Any]
    optics: dict[str, Any]
    dng: dict[str, Any]
    exposure: dict[str, Any]
    white_balance: dict[str, Any]
    camera_geometry: dict[str, Any]
    hdr: dict[str, Any]
    provenance: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path, label: str) -> RunFacts:
        root = Path(root)
        meta = root / "metadata"
        if not (meta / "config.json").is_file():
            raise FileNotFoundError(f"{root} is not a finished run folder (metadata/config.json missing)")
        groups = json.loads((meta / "metadata.json").read_text()).get("groups", {})
        dng = {item["field"]: item["value"] for item in groups.get("dng", [])}
        return cls(
            root=root,
            label=label,
            config=json.loads((meta / "config.json").read_text()),
            geometry=_records(meta / "geometry_plan.json"),
            input_audit=_records(meta / "input_audit.json"),
            camera_profile=_records(meta / "camera_profile.json"),
            sensor_profile=_records(meta / "sensor_profile.json"),
            optics=_records(meta / "optical_parameters.json"),
            dng=dng,
            exposure=_records(meta / "exposure_posterior.json"),
            white_balance=_records(meta / "white_balance.json"),
            camera_geometry=_records(meta / "camera_geometry.json"),
            hdr=_records(meta / "hdr_reconstruction.json"),
            provenance=_records(meta / "provenance.json"),
        )

    @property
    def source_size(self) -> tuple[int, int]:
        return int(self.input_audit.get("width", 0)), int(self.input_audit.get("height", 0))

    def locked(self) -> dict[str, Any]:
        """Every PRD locked setting of this run, keyed ``section.name``."""
        out: dict[str, Any] = {}
        sections = {
            "config": {k: v for k, v in self.config.items() if k not in INPUT_CONFIG_KEYS},
            "geometry": {k: self.geometry.get(k) for k in GEOMETRY_KEYS},
            "camera_profile": self.camera_profile,
            "sensor": {k: v for k, v in self.sensor_profile.items() if k not in EXPOSURE_DERIVED_FIELDS},
            "lens": self.optics,
            "dng_colour": {k: self.dng.get(k) for k in DNG_COLOUR_FIELDS},
            "backends": {
                **dict(self.provenance.get("backends", {})),
                "hdr_fallback": self.hdr.get("fallback_reason"),
                "fallbacks": list(self.provenance.get("fallbacks", [])),
            },
        }
        for section, values in sections.items():
            out.update({f"{section}.{name}": value for name, value in values.items()})
        return out


def _encoded_to_srgb8(adobe: np.ndarray) -> np.ndarray:
    """Adobe RGB (1998) encoded [0, 1] -> 8-bit sRGB for display in the report."""
    from synthraw.colour_space import rgb_to_xyz_matrix
    from synthraw.exporters import ADOBE_GAMMA
    from synthraw.validation import _srgb_encode

    linear = np.clip(adobe, 0, 1).astype(np.float64) ** ADOBE_GAMMA
    to_srgb = np.linalg.solve(rgb_to_xyz_matrix("SRGB"), rgb_to_xyz_matrix("ADOBE_RGB"))
    return np.round(_srgb_encode(linear @ to_srgb.T) * 255).astype(np.uint8)


def _png(array: np.ndarray, *, min_side: int = 0) -> bytes:
    from PIL import Image

    image = Image.fromarray(array)
    if min_side and min(image.size) < min_side:
        scale = math.ceil(min_side / min(image.size))
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _heatmap(values: np.ndarray, vmax: float) -> np.ndarray:
    from matplotlib import colormaps

    scaled = np.clip(values / max(vmax, 1e-12), 0, 1)
    return np.round(colormaps["magma"](scaled)[..., :3] * 255).astype(np.uint8)


def _analysis_view(master: np.ndarray, box: tuple[int, int, int, int] | None, max_side: int) -> np.ndarray:
    """Area-averaged [0, 1] view of a uint16 master (whole frame or a pixel box), bounded to ``max_side``."""
    if box is not None:
        x0, y0, x1, y1 = box
        master = master[y0:y1, x0:x1]
    factor = max(1, math.ceil(max(master.shape[:2]) / max_side))
    return block_mean(master, factor) / 65535.0


def image_agreement(first: np.ndarray, second: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Difference, SSIM, edge and texture agreement between two [0, 1] RGB views on one grid."""
    from scipy.ndimage import gaussian_filter, sobel
    from skimage.metrics import structural_similarity

    if first.shape != second.shape:
        raise ValueError("views must share one grid")
    diff = np.abs(first - second).mean(axis=-1)
    luma = [np.asarray(view, np.float64) @ _ADOBE_Y for view in (first, second)]
    mse = float(np.mean((first - second) ** 2))
    side = min(diff.shape)
    win = min(7, side if side % 2 else side - 1)
    ssim = float(structural_similarity(luma[0], luma[1], data_range=1.0, win_size=win)) if win >= 3 else math.nan
    edges = [np.hypot(sobel(y, axis=0), sobel(y, axis=1)) for y in luma]
    strong = [e >= np.percentile(e, 90) for e in edges]
    union = np.count_nonzero(strong[0] | strong[1])
    highpass = [y - gaussian_filter(y, 2.0) for y in luma]
    energy = [np.sqrt(gaussian_filter(h * h, 4.0)) for h in highpass]
    metrics = {
        "mean_abs_difference": float(diff.mean()),
        "p95_abs_difference": float(np.percentile(diff, 95)),
        "max_abs_difference": float(diff.max()),
        "psnr_db": math.inf if mse == 0 else float(10 * math.log10(1.0 / mse)),
        "ssim": ssim,
        "edge_correlation": _pearson(edges[0], edges[1]),
        "edge_iou_top10": float(np.count_nonzero(strong[0] & strong[1]) / union) if union else 1.0,
        "texture_correlation": _pearson(energy[0], energy[1]),
        "texture_energy_ratio": float(energy[1].mean() / energy[0].mean()) if energy[0].mean() > 0 else math.nan,
    }
    return metrics, {"difference": diff, "edges_low": edges[0], "edges_high": edges[1]}


def _stops(first: float, second: float, *, per_doubling: float = 1.0) -> float:
    if first <= 0 or second <= 0:
        return math.inf
    return abs(per_doubling * math.log2(second / first))


def _mask_disagreement(first: Path, second: Path, *, strip_rows: int = 512) -> float | None:
    if not (first.is_file() and second.is_file()):
        return None
    a, b = np.load(first, mmap_mode="r"), np.load(second, mmap_mode="r")
    if a.shape != b.shape:
        return None
    differ = sum(
        int(np.count_nonzero((np.asarray(a[i : i + strip_rows]) > 0.5) != (np.asarray(b[i : i + strip_rows]) > 0.5)))
        for i in range(0, len(a), strip_rows)
    )
    return 100.0 * differ / max(1, a.size)


def dng_statistics(root: str | Path, *, max_samples: int = 250_000) -> dict[str, dict[str, float]]:
    """Per-CFA-plane statistics of the stored mosaic, from an even stride that preserves CFA phase."""
    root = Path(root)
    mosaic = np.load(root / "raw" / "sensor_mosaic_uint16.npy", mmap_mode="r")
    profile = _records(root / "metadata" / "sensor_profile.json")
    phase = str(profile.get("applied_cfa_phase") or profile.get("cfa_pattern") or "RGGB")
    black = float(profile.get("applied_adc_black_level", profile.get("BlackLevel", 0)))
    white = float(profile.get("applied_adc_white_level", profile.get("WhiteLevel", 65535)))
    plane_pixels = mosaic.size / 4
    stride = 2 * max(1, math.ceil(math.sqrt(plane_pixels / max_samples)))
    out: dict[str, dict[str, float]] = {}
    for index, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        values = np.asarray(mosaic[dy::stride, dx::stride], dtype=np.float64).ravel()
        out[f"{index}_{phase[index]}"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p01": float(np.percentile(values, 1)),
            "p50": float(np.percentile(values, 50)),
            "p99": float(np.percentile(values, 99)),
            "black_clipped_pct": 100.0 * float(np.mean(values <= black)),
            "white_clipped_pct": 100.0 * float(np.mean(values >= white)),
        }
    return out


@dataclass
class SourceResolutionComparison:
    """Result of the 2K-versus-4K comparison: checks, flags, tables and in-memory PNGs."""

    low: RunFacts
    high: RunFacts
    checks: list[ValidationCheck]
    flags: list[str]
    low_source_derived: bool
    tolerances: EstimatorTolerances
    locked_mismatches: dict[str, tuple[Any, Any]]
    estimators: list[dict[str, Any]]
    hdr_masks: list[dict[str, Any]]
    dng_stats: dict[str, dict[str, dict[str, float]]]
    crops: list[dict[str, Any]]
    images: dict[str, bytes] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "FAILED" if hard_failures(self.checks) else "COMPLETED"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "status": self.status,
                "flags": self.flags,
                "low_run": str(self.low.root),
                "high_run": str(self.high.root),
                "low_source_size": list(self.low.source_size),
                "high_source_size": list(self.high.source_size),
                "low_source_derived": self.low_source_derived,
                "tolerances": asdict(self.tolerances),
                "locked_mismatches": {k: list(v) for k, v in self.locked_mismatches.items()},
                "estimators": self.estimators,
                "hdr_masks": self.hdr_masks,
                "dng_stats": self.dng_stats,
                "crops": [{k: v for k, v in crop.items() if k != "images"} for crop in self.crops],
                "checks": [check.to_dict() for check in self.checks],
            }
        )


class SourceResolutionComparator:
    """PRD 2K-versus-4K experiment over two finished runs; only source resolution may vary."""

    def __init__(
        self,
        tolerances: EstimatorTolerances | None = None,
        *,
        analysis_side: int = 1024,
        crop_side: int = 384,
        aspect_tolerance: float = 0.01,
    ) -> None:
        self.tolerances = tolerances or EstimatorTolerances()
        self.analysis_side = analysis_side
        self.crop_side = crop_side
        self.aspect_tolerance = aspect_tolerance

    def compare(
        self,
        first_root: str | Path,
        second_root: str | Path,
        *,
        crops: Mapping[str, Sequence[float]] | Iterable[CropBox] | None = None,
        low_source_derived: bool = False,
        log: Callable[[str], None] | None = None,
    ) -> SourceResolutionComparison:
        """Compare two runs; they are ordered by source pixel count (low = 2K-like, high = 4K-like)."""
        say = log or (lambda _message: None)
        runs = sorted(
            (RunFacts.load(first_root, "first"), RunFacts.load(second_root, "second")),
            key=lambda facts: facts.source_size[0] * facts.source_size[1],
        )
        low, high = runs
        low.label, high.label = "low-resolution source", "high-resolution source"
        say(f"comparing {low.root.name} ({low.source_size}) with {high.root.name} ({high.source_size})")
        checks = [self._pairing(low, high)]
        mismatches = self._locked(low, high)
        checks.append(
            ValidationCheck(
                "locked_settings",
                "Experiment design",
                not mismatches,
                hard=True,
                measured={"settings_compared": len(low.locked()), "mismatches": sorted(mismatches)},
                target="every PRD locked setting identical; only source resolution varies",
            )
        )
        estimators, estimator_checks = self._estimators(low, high)
        checks += estimator_checks
        flags = [ESTIMATOR_UNSTABLE] if any(check.passed is False for check in estimator_checks) else []
        result = SourceResolutionComparison(
            low=low,
            high=high,
            checks=checks,
            flags=flags,
            low_source_derived=low_source_derived,
            tolerances=self.tolerances,
            locked_mismatches=mismatches,
            estimators=estimators,
            hdr_masks=self._hdr(low, high),
            dng_stats={"low": dng_statistics(low.root), "high": dng_statistics(high.root)},
            crops=[],
        )
        if not any(check.name == "locked_settings" and check.passed for check in checks):
            say("locked settings differ; image agreement is not attributable to source resolution alone")
        self._images(result, parse_crops(crops))
        return result

    def _pairing(self, low: RunFacts, high: RunFacts) -> ValidationCheck:
        (lw, lh), (hw, hh) = low.source_size, high.source_size
        aspect_error = abs((lw / lh) / (hw / hh) - 1) if min(lw, lh, hw, hh) > 0 else math.inf
        ratio = math.sqrt((hw * hh) / (lw * lh)) if min(lw, lh) > 0 else math.inf
        return ValidationCheck(
            "source_pairing",
            "Experiment design",
            aspect_error <= self.aspect_tolerance and hw * hh > lw * lh,
            hard=True,
            measured={
                "low_source": [lw, lh],
                "high_source": [hw, hh],
                "linear_ratio": ratio,
                "aspect_error": aspect_error,
                "low_sha256": low.input_audit.get("sha256"),
                "high_sha256": high.input_audit.get("sha256"),
            },
            target=f"same composition (aspect within {self.aspect_tolerance:.0%}) at a strictly higher resolution",
        )

    @staticmethod
    def _locked(low: RunFacts, high: RunFacts) -> dict[str, tuple[Any, Any]]:
        first, second = low.locked(), high.locked()
        return {
            key: (first.get(key), second.get(key))
            for key in sorted(first.keys() | second.keys())
            if not _same(first.get(key), second.get(key))
        }

    def _estimators(self, low: RunFacts, high: RunFacts) -> tuple[list[dict[str, Any]], list[ValidationCheck]]:
        tol = self.tolerances
        rows: list[dict[str, Any]] = []

        def add(group: str, name: str, a: Any, b: Any, delta: float | None, unit: str, limit: float | None) -> None:
            rows.append(
                {
                    "group": group,
                    "field": name,
                    "low": a,
                    "high": b,
                    "delta": delta,
                    "unit": unit,
                    "tolerance": limit,
                    "stable": None if delta is None or limit is None else bool(delta <= limit),
                }
            )

        exposure = (("ExposureTime", 1.0), ("PhotographicSensitivity", 1.0), ("FNumber", 2.0))
        for name, per_doubling in exposure:
            a, b = low.exposure.get(name), high.exposure.get(name)
            delta = _stops(float(a), float(b), per_doubling=per_doubling) if a is not None and b is not None else None
            add("exposure", name, a, b, delta, "stops", tol.exposure_stops)
        for name in ("scene_ev100", "BaselineExposure", "ExposureBiasValue"):
            a, b = low.exposure.get(name), high.exposure.get(name)
            delta = abs(float(b) - float(a)) if a is not None and b is not None else None
            add("exposure", name, a, b, delta, "EV", tol.exposure_stops)
        for name in ("FocalLength", "FocalLengthIn35mmFormat", "horizontal_fov_degrees"):
            a, b = low.camera_geometry.get(name), high.camera_geometry.get(name)
            delta = abs(float(b) / float(a) - 1) if a and b is not None else None
            add("camera_geometry", name, a, b, delta, "relative", tol.focal_length_rel)
        a, b = low.white_balance.get("CCT"), high.white_balance.get("CCT")
        delta = abs(1e6 / float(a) - 1e6 / float(b)) if a and b else None
        add("white_balance", "CCT", a, b, delta, "mired", tol.cct_mired)
        a, b = low.white_balance.get("tint"), high.white_balance.get("tint")
        add(
            "white_balance",
            "tint",
            a,
            b,
            abs(float(b) - float(a)) if a is not None and b is not None else None,
            "",
            tol.tint,
        )
        a, b = low.white_balance.get("AsShotNeutral"), high.white_balance.get("AsShotNeutral")
        delta = float(np.max(np.abs(np.asarray(b) / np.asarray(a) - 1))) if a is not None and b is not None else None
        add("white_balance", "AsShotNeutral", a, b, delta, "max relative", tol.as_shot_neutral_rel)
        a, b = low.white_balance.get("LightSource"), high.white_balance.get("LightSource")
        add("white_balance", "LightSource", a, b, None, "EXIF code", None)

        checks = []
        for group, title in (
            ("exposure", "estimator_exposure"),
            ("camera_geometry", "estimator_camera_geometry"),
            ("white_balance", "estimator_white_balance"),
        ):
            group_rows = [row for row in rows if row["group"] == group and row["stable"] is not None]
            unstable = [row["field"] for row in group_rows if not row["stable"]]
            checks.append(
                ValidationCheck(
                    title,
                    "Estimator stability",
                    None if not group_rows else not unstable,
                    measured={row["field"]: row["delta"] for row in group_rows} | {"unstable": unstable},
                    target=f"within tolerance, else {ESTIMATOR_UNSTABLE}",
                )
            )
        return rows, checks

    @staticmethod
    def _hdr(low: RunFacts, high: RunFacts) -> list[dict[str, Any]]:
        rows = [
            {
                "mask": "generated_hdr_area_percentage",
                "low_pct": low.hdr.get("generated_hdr_area_percentage"),
                "high_pct": high.hdr.get("generated_hdr_area_percentage"),
                "disagreement_pct": None,
            }
        ]
        low_pct, high_pct = mask_percentages(low.root), mask_percentages(high.root)
        for name, label in SUPPORT_MASKS:
            if name in low_pct or name in high_pct:
                rows.append(
                    {
                        "mask": label,
                        "low_pct": low_pct.get(name),
                        "high_pct": high_pct.get(name),
                        "disagreement_pct": _mask_disagreement(
                            low.root / "confidence" / name, high.root / "confidence" / name
                        ),
                    }
                )
        return rows

    def _images(self, result: SourceResolutionComparison, crops: list[CropBox]) -> None:
        paths = [facts.root / DEVELOPED_MASTER for facts in (result.low, result.high)]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing; the comparison needs both developed masters")
        low, high = (_open_tiff(path) for path in paths)
        if low.shape != high.shape:
            result.checks.append(
                ValidationCheck(
                    "final_difference_map",
                    "Image agreement",
                    False,
                    hard=True,
                    measured={"low_shape": list(low.shape), "high_shape": list(high.shape)},
                    target="both developed masters on the locked output geometry",
                )
            )
            return
        views = [_analysis_view(master, None, self.analysis_side) for master in (low, high)]
        metrics, maps = image_agreement(*views)
        scale = max(float(np.percentile(maps["difference"], 99)), 1e-3)
        result.images |= {
            "low": _png(_encoded_to_srgb8(views[0]), min_side=256),
            "high": _png(_encoded_to_srgb8(views[1]), min_side=256),
            "difference": _png(_heatmap(maps["difference"], scale), min_side=256),
            "edges_low": _png(_heatmap(maps["edges_low"], float(np.percentile(maps["edges_low"], 99.5))), min_side=256),
            "edges_high": _png(
                _heatmap(maps["edges_high"], float(np.percentile(maps["edges_high"], 99.5))), min_side=256
            ),
        }
        common = {"analysed_shape": list(views[0].shape[:2]), "difference_display_max": scale}
        result.checks += [
            ValidationCheck(
                "final_difference_map",
                "Image agreement",
                None,
                measured={
                    k: metrics[k]
                    for k in ("mean_abs_difference", "p95_abs_difference", "max_abs_difference", "psnr_db", "ssim")
                }
                | common,
                target="report only: where source resolution matters",
            ),
            ValidationCheck(
                "edge_agreement",
                "Image agreement",
                None,
                measured={k: metrics[k] for k in ("edge_correlation", "edge_iou_top10")},
                target="report only: structure changes lower the correlation",
            ),
            ValidationCheck(
                "texture_agreement",
                "Image agreement",
                None,
                measured={k: metrics[k] for k in ("texture_correlation", "texture_energy_ratio")},
                target="report only: hallucination instability lowers the correlation",
            ),
        ]
        height, width = low.shape[:2]
        for crop in crops:
            box = crop.pixels(width, height)
            x0, y0, x1, y1 = box
            pair = [_analysis_view(master, box, self.crop_side) for master in (low, high)]
            crop_metrics, crop_maps = image_agreement(*pair)
            result.crops.append(
                {
                    "name": crop.name,
                    "box_fraction": [crop.x0, crop.y0, crop.x1, crop.y1],
                    "box_pixels": [x0, y0, x1, y1],
                    "analysed_shape": list(pair[0].shape[:2]),
                    "metrics": crop_metrics,
                    "images": {
                        "low": _png(_encoded_to_srgb8(pair[0]), min_side=128),
                        "high": _png(_encoded_to_srgb8(pair[1]), min_side=128),
                        "difference": _png(_heatmap(crop_maps["difference"], scale), min_side=128),
                    },
                }
            )


def _fixed(value: float | None, key: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if key.endswith("_pct") else f"{value:.1f}"


def _data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


_EXTRA_CSS = """
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.grid figure{margin:0;min-width:0}
.grid img{width:100%;image-rendering:pixelated}.banner.flag{background:#fff4e5;border:1px solid #b26a00;color:#7a4800}
"""


class ComparisonReportGenerator:
    """Write the self-contained ``comparison_2k_vs_4k.html`` and its JSON for a comparison result."""

    def generate(self, result: SourceResolutionComparison, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / COMPARISON_JSON).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        path = out / COMPARISON_NAME
        path.write_text(self.render(result), encoding="utf-8")
        return path

    def render(self, result: SourceResolutionComparison) -> str:
        esc = html.escape
        low, high = result.low, result.high
        failures = hard_failures(result.checks)
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>2K versus 4K source comparison - {esc(result.status)}</title>",
            f"<style>{_CSS}{_EXTRA_CSS}</style></head><body>",
            "<h1>2K versus 4K source-resolution comparison</h1>",
            f"<p>Runs <code>{esc(low.root.name)}</code> and <code>{esc(high.root.name)}</code> - generated "
            f"{esc(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))}</p>",
        ]
        if failures:
            parts.append(
                '<div class="banner failed">COMPARISON FAILED - '
                f"{esc(', '.join(check.name for check in failures))}. The two runs differ in more than source "
                "resolution, so their differences cannot be attributed to source pixel count.</div>"
            )
        else:
            parts.append(
                '<div class="banner passed">COMPARISON COMPLETED - every locked setting is identical; only source '
                "resolution varies.</div>"
            )
        if ESTIMATOR_UNSTABLE in result.flags:
            unstable = [row["field"] for row in result.estimators if row["stable"] is False]
            parts.append(
                f'<div class="banner flag">{ESTIMATOR_UNSTABLE} - estimated camera parameters moved beyond '
                f"tolerance with source resolution: {esc(', '.join(unstable))}.</div>"
            )
        parts.append(
            '<div class="disclosure"><strong>Synthetic-capture disclosure.</strong> '
            f"{esc(SYNTHETIC_CAPTURE_DISCLOSURE)}<br><strong>Scientific boundary.</strong> "
            f"{esc(SCIENTIFIC_BOUNDARY)}</div>"
        )
        parts += self._sources(result)
        parts += self._checks(result.checks)
        parts += self._locked(result)
        parts.append("<h2>Final 50 MP difference map</h2>")
        if "difference" in result.images:
            scale = next(c for c in result.checks if c.name == "final_difference_map").measured
            parts.append("<div class='grid'>")
            parts += [
                self._figure(result.images[key], caption)
                for key, caption in (
                    ("low", f"From the {low.source_size[0]} x {low.source_size[1]} source"),
                    ("high", f"From the {high.source_size[0]} x {high.source_size[1]} source"),
                    ("difference", f"|difference|, display range 0..{scale['difference_display_max']:.3g}"),
                )
            ]
            parts.append("</div><h3>Edge agreement (Sobel magnitude)</h3><div class='grid'>")
            parts += [
                self._figure(result.images["edges_low"], "Edges, low-resolution source"),
                self._figure(result.images["edges_high"], "Edges, high-resolution source"),
            ]
            parts.append(
                "</div><p class='info'>Views are area-averaged to the analysed shape and shown as sRGB; "
                "metrics use Adobe RGB encoded values of the developed masters.</p>"
            )
        parts += self._crops(result)
        parts += self._estimator_table(result)
        parts += self._hdr_table(result)
        parts += self._dng_table(result)
        parts.append("<h2>Tolerances (configuration)</h2><table class='kv'><tr><th>Tolerance</th><th>Value</th></tr>")
        parts += [
            f"<tr><td>{esc(k)}</td><td class='num'>{esc(str(v))}</td></tr>"
            for k, v in asdict(result.tolerances).items()
        ]
        parts.append("</table></body></html>")
        return "\n".join(parts) + "\n"

    @staticmethod
    def _figure(data: bytes, caption: str) -> str:
        caption = html.escape(caption)
        return f'<figure><img src="{_data_uri(data)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'

    @staticmethod
    def _sources(result: SourceResolutionComparison) -> list[str]:
        esc = html.escape
        parts = [
            "<h2>Sources</h2><table><tr><th></th><th>Low-resolution source</th><th>High-resolution source</th></tr>"
        ]
        rows = [
            ("Run", result.low.root.name, result.high.root.name),
            (
                "Source size",
                " x ".join(map(str, result.low.source_size)),
                " x ".join(map(str, result.high.source_size)),
            ),
            ("Source SHA-256", result.low.input_audit.get("sha256"), result.high.input_audit.get("sha256")),
            (
                "Output size",
                f"{result.low.geometry.get('width')} x {result.low.geometry.get('height')}",
                f"{result.high.geometry.get('width')} x {result.high.geometry.get('height')}",
            ),
            ("SR scale", result.low.geometry.get("scale_factor"), result.high.geometry.get("scale_factor")),
        ]
        parts += [f"<tr><th>{esc(k)}</th><td>{esc(str(a))}</td><td>{esc(str(b))}</td></tr>" for k, a, b in rows]
        parts.append("</table>")
        if result.low_source_derived:
            parts.append(
                "<p class='banner flag'>The low-resolution source is a Lanczos-downsampled copy of the "
                "high-resolution input, not an independently rendered source. It isolates source pixel count only.</p>"
            )
        return parts

    @staticmethod
    def _checks(checks: list[ValidationCheck]) -> list[str]:
        esc = html.escape
        parts = [
            "<h2>Checks</h2><table><tr><th style='width:18%'>Check</th><th style='width:20%'>Target</th>"
            "<th>Measured</th><th style='width:9%'>Result</th></tr>"
        ]
        for check in checks:
            if check.passed is None:
                css, label = "info", "report only" if "report" in check.target else "n/a"
            else:
                css, label = ("pass", "pass") if check.passed else ("fail", "FAIL" if check.hard else "unstable")
            parts.append(
                f"<tr><td>{esc(check.name)}</td><td>{esc(check.target)}</td>"
                f"<td>{esc(format_measured(check.measured))}</td><td class='{css}'>{label}</td></tr>"
            )
        parts.append("</table>")
        return parts

    @staticmethod
    def _locked(result: SourceResolutionComparison) -> list[str]:
        esc = html.escape
        compared = len(result.low.locked())
        parts = [
            f"<h2>Locked settings</h2><p>{compared} settings compared across config, output geometry, camera "
            "profile, sensor parameters and seeds, lens, DNG colour matrices and backends.</p>"
        ]
        if result.locked_mismatches:
            parts.append("<table><tr><th>Setting</th><th>Low</th><th>High</th></tr>")
            parts += [
                f"<tr><td>{esc(k)}</td><td>{esc(format_measured(a))}</td><td>{esc(format_measured(b))}</td></tr>"
                for k, (a, b) in result.locked_mismatches.items()
            ]
            parts.append("</table>")
        else:
            parts.append("<p class='pass'>All locked settings identical.</p>")
        return parts

    def _crops(self, result: SourceResolutionComparison) -> list[str]:
        esc = html.escape
        if not result.crops:
            return []
        parts = [
            "<h2>Named crops</h2><table><tr><th>Crop</th><th>Box (px)</th><th>SSIM</th><th>Mean |diff|</th>"
            "<th>Edge corr.</th><th>Texture corr.</th><th>Texture energy high/low</th></tr>"
        ]
        for crop in result.crops:
            m = crop["metrics"]
            values = (m["ssim"], m["mean_abs_difference"], m["edge_correlation"], m["texture_correlation"])
            parts.append(
                f"<tr><td>{esc(crop['name'])}</td><td>{esc(str(crop['box_pixels']))}</td>"
                + "".join(f"<td class='num'>{format_measured(v)}</td>" for v in values)
                + f"<td class='num'>{format_measured(m['texture_energy_ratio'])}</td></tr>"
            )
        parts.append("</table>")
        for crop in result.crops:
            parts.append(f"<h3>{esc(crop['name'])}</h3><div class='grid'>")
            parts += [
                self._figure(crop["images"][key], f"{crop['name']} - {label}")
                for key, label in (
                    ("low", "low-resolution source"),
                    ("high", "high-resolution source"),
                    ("difference", "|difference|"),
                )
            ]
            parts.append("</div>")
        return parts

    @staticmethod
    def _estimator_table(result: SourceResolutionComparison) -> list[str]:
        esc = html.escape
        parts = [
            "<h2>Exposure posterior, camera geometry and white balance</h2><table><tr><th>Group</th><th>Field</th>"
            "<th>Low</th><th>High</th><th>Change</th><th>Tolerance</th><th>Result</th></tr>"
        ]
        for row in result.estimators:
            stable = row["stable"]
            css, label = (
                ("info", "report only") if stable is None else (("pass", "stable") if stable else ("fail", "unstable"))
            )
            delta = "n/a" if row["delta"] is None else f"{format_measured(row['delta'])} {row['unit']}".strip()
            tolerance = "n/a" if row["tolerance"] is None else format_measured(row["tolerance"])
            parts.append(
                f"<tr><td>{esc(row['group'])}</td><td>{esc(row['field'])}</td>"
                f"<td>{esc(format_measured(row['low']))}</td><td>{esc(format_measured(row['high']))}</td>"
                f"<td class='num'>{esc(delta)}</td><td class='num'>{esc(tolerance)}</td>"
                f"<td class='{css}'>{label}</td></tr>"
            )
        parts.append("</table>")
        return parts

    @staticmethod
    def _hdr_table(result: SourceResolutionComparison) -> list[str]:
        esc = html.escape

        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{float(value):.2f}"

        parts = [
            "<h2>HDR and confidence masks</h2><table><tr><th>Mask</th><th>Low (% of frame)</th>"
            "<th>High (% of frame)</th><th>Pixels that disagree (%)</th></tr>"
        ]
        parts += [
            f"<tr><td>{esc(row['mask'])}</td><td class='num'>{pct(row['low_pct'])}</td>"
            f"<td class='num'>{pct(row['high_pct'])}</td><td class='num'>{pct(row['disagreement_pct'])}</td></tr>"
            for row in result.hdr_masks
        ]
        parts.append("</table>")
        return parts

    @staticmethod
    def _dng_table(result: SourceResolutionComparison) -> list[str]:
        esc = html.escape
        keys = ("mean", "std", "p01", "p50", "p99", "black_clipped_pct", "white_clipped_pct")
        parts = [
            "<h2>DNG statistics (raw mosaic, per CFA plane)</h2><table><tr><th>Plane</th><th>Source</th>"
            + "".join(f"<th>{esc(k)}</th>" for k in keys)
            + "</tr>"
        ]
        for plane in result.dng_stats["low"]:
            for side in ("low", "high"):
                stats = result.dng_stats[side].get(plane, {})
                parts.append(
                    f"<tr><td>{esc(plane)}</td><td>{side}</td>"
                    + "".join(f"<td class='num'>{_fixed(stats.get(k), k)}</td>" for k in keys)
                    + "</tr>"
                )
        parts.append("</table>")
        return parts


def compare_and_report(
    first_root: str | Path,
    second_root: str | Path,
    out_dir: str | Path,
    *,
    crops: Mapping[str, Sequence[float]] | Iterable[CropBox] | None = None,
    low_source_derived: bool = False,
    comparator: SourceResolutionComparator | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[SourceResolutionComparison, Path]:
    """Compare two runs and write ``comparison_2k_vs_4k.html`` (plus JSON) into ``out_dir``."""
    result = (comparator or SourceResolutionComparator()).compare(
        first_root, second_root, crops=crops, low_source_derived=low_source_derived, log=log
    )
    return result, ComparisonReportGenerator().generate(result, out_dir)


USER_STUDY_QUESTIONS = (
    "Which looks more photographic?",
    "Which better preserves the original subject?",
    "Which looks less AI-generated?",
    "Which has more believable fine detail?",
    "Which tolerates the graded exposure/WB edit better?",
)


class UserStudyPairSheet:
    """Optional PRD user-study sheet: randomised, blinded pairs of matched 100% crops. Never run automatically.

    ``pairs`` is a sequence of ``(label_a, run_a, label_b, run_b)``, e.g. 2K- versus 4K-derived runs,
    fidelity versus generative SR, or noise off versus on. Each crop is a centred 100% window of at
    most ``window`` pixels inside the named box. The left/right order is randomised per pair with
    ``seed``; the key is written to a separate JSON for the experimenter.
    """

    def __init__(self, *, seed: int = 0, window: int = 512) -> None:
        self.seed = seed
        self.window = window

    def write(
        self,
        pairs: Sequence[tuple[str, str | Path, str, str | Path]],
        out_dir: str | Path,
        crops: Mapping[str, Sequence[float]] | Iterable[CropBox] | None = None,
    ) -> Path:
        rng = np.random.default_rng(self.seed)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        boxes = parse_crops(crops)
        trials: list[tuple[str, bytes, bytes]] = []
        key: list[dict[str, Any]] = []
        for label_a, run_a, label_b, run_b in pairs:
            masters = [_open_tiff(Path(run) / DEVELOPED_MASTER) for run in (run_a, run_b)]
            if masters[0].shape != masters[1].shape:
                raise ValueError(f"{label_a} and {label_b} masters differ in shape; crops would not match")
            height, width = masters[0].shape[:2]
            for crop in boxes:
                x0, y0, x1, y1 = crop.pixels(width, height)
                w, h = min(self.window, x1 - x0), min(self.window, y1 - y0)
                cx, cy = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
                images = [
                    _png(_encoded_to_srgb8(np.asarray(m[cy : cy + h, cx : cx + w], np.float64) / 65535))
                    for m in masters
                ]
                swap = bool(rng.integers(2))
                left, right = (images[1], images[0]) if swap else (images[0], images[1])
                trial = f"pair_{len(key) + 1:03d}"
                key.append(
                    {
                        "trial": trial,
                        "crop": crop.name,
                        "box_pixels": [cx, cy, cx + w, cy + h],
                        "left": label_b if swap else label_a,
                        "right": label_a if swap else label_b,
                    }
                )
                trials.append((trial, left, right))
        order = rng.permutation(len(trials))
        (out / USER_STUDY_KEY).write_text(json.dumps({"seed": self.seed, "trials": [key[i] for i in order]}, indent=2))
        esc = html.escape
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8"><title>Blinded image pairs</title>',
            f"<style>{_CSS}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}"
            ".pair img{width:100%;image-rendering:pixelated}</style></head><body>",
            "<h1>Blinded image pairs</h1>",
            "<p>Each trial shows two matched 100% crops, Image 1 and Image 2, in random order. "
            "For each question, choose 1 or 2. Answer each question separately.</p><ol>",
            *(f"<li>{esc(q)}</li>" for q in USER_STUDY_QUESTIONS),
            "</ol>",
        ]
        for index in order:
            trial, left, right = trials[index]
            parts.append(f"<h2>{esc(trial)}</h2><div class='pair'>")
            parts += [
                f"<figure><img src='{_data_uri(data)}' alt='Image {n}'><figcaption>Image {n}</figcaption></figure>"
                for n, data in ((1, left), (2, right))
            ]
            parts.append("</div>")
        parts.append("</body></html>")
        path = out / USER_STUDY_NAME
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return path
