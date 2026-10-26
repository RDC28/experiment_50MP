"""Automated validation sequence and numerical tests for PRD_v3's Validation milestone.

Every validator returns ``ValidationCheck`` records: hard gates carry pass/fail, project QA
targets carry pass/fail against configurable thresholds, and report-only metrics carry
``passed=None``. Measured values are always reported, never tuned to pass.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from synthraw.adc import ADCSimulator
from synthraw.cfa import CFAMosaicer
from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import CFA_PATTERNS, DNGFieldRecord
from synthraw.libraw_validator import HardValidationError, LibRawValidator
from synthraw.sensor import SensorNoiseSimulator

# Linear sRGB (D65) to XYZ, matching LibRaw's rgb_xyz used by LibRawValidator.
_SRGB_TO_XYZ = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
_D65_XYZ = _SRGB_TO_XYZ @ np.ones(3)


@dataclass(frozen=True)
class ValidationThresholds:
    """PRD numerical-test targets, exposed as configuration rather than altered output."""

    roundtrip_supported_ssim: float = 0.97
    roundtrip_whole_ssim: float = 0.93
    median_delta_e00: float = 2.5
    seam_ratio: float = 1.25
    noise_profile_rel_tol: float = 0.05
    noise_fit_r2: float = 0.95
    neutral_tolerance: float = 1e-3
    wb_max_clipped_increase: float = 0.05


@dataclass
class ValidationCheck:
    """One row of the validation table: hard gate, QA target or report-only metric."""

    name: str
    domain: str
    passed: bool | None
    hard: bool = False
    measured: dict[str, Any] = field(default_factory=dict)
    target: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "passed": self.passed,
            "hard": self.hard,
            "measured": _jsonable(self.measured),
            "target": self.target,
            "detail": self.detail,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def hard_failures(checks: Iterable[ValidationCheck]) -> list[ValidationCheck]:
    """Hard gates that did not pass; any entry makes the run a FAILED validation."""
    return [check for check in checks if check.hard and check.passed is not True]


def _strips(array: np.ndarray, rows: int) -> Iterable[np.ndarray]:
    for start in range(0, array.shape[0], rows):
        yield np.asarray(array[start : start + rows])


class NumericHealthValidator:
    """PRD numeric health: zero NaN/Inf and zero out-of-range samples, in bounded strips."""

    def validate(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        ranges: Mapping[str, tuple[float | None, float | None]] | None = None,
        strip_rows: int = 128,
    ) -> ValidationCheck:
        ranges = ranges or {}
        measured: dict[str, Any] = {}
        failures = 0
        for name, array in arrays.items():
            low, high = ranges.get(name, (None, None))
            nonfinite = below = above = 0
            for strip in _strips(array, strip_rows):
                finite = np.isfinite(strip) if strip.dtype.kind == "f" else np.ones(strip.shape, bool)
                nonfinite += int(strip.size - np.count_nonzero(finite))
                if low is not None:
                    below += int(np.count_nonzero(finite & (strip < low)))
                if high is not None:
                    above += int(np.count_nonzero(finite & (strip > high)))
            measured[name] = {"nonfinite": nonfinite, "below_range": below, "above_range": above}
            failures += nonfinite + below + above
        return ValidationCheck(
            "numeric_health",
            "Numeric health",
            failures == 0,
            hard=True,
            measured={"failures": failures, "arrays": measured},
            target="zero NaN/Inf/out-of-range",
        )


class CFAValidator:
    """PRD CFA gate: one value per photosite and the declared periodic Bayer phase."""

    def validate(
        self,
        mosaic: np.ndarray,
        rgb: np.ndarray,
        phase: str,
        records: Mapping[str, DNGFieldRecord] | None = None,
        *,
        strip_rows: int = 128,
    ) -> ValidationCheck:
        """``rgb`` is the pre-CFA frame that produced ``mosaic`` (e.g. 07 -> 09)."""
        if phase not in CFA_PATTERNS:
            raise ValueError("phase must be RGGB, BGGR, GRBG, or GBRG")
        pattern = CFA_PATTERNS[phase]
        problems = []
        if mosaic.ndim != 2:
            problems.append("mosaic is not one value per photosite")
        elif rgb.shape != (*mosaic.shape, 3):
            problems.append("mosaic and pre-CFA frame shapes differ")
        if records is not None:
            if tuple(records["CFAPattern"].value) != pattern:
                problems.append("DNG CFAPattern differs from the applied phase")
            if tuple(records["CFARepeatPatternDim"].value) != (2, 2):
                problems.append("CFARepeatPatternDim is not 2x2")
        mismatches = 0
        # Each photosite must equal the pre-CFA value of its own phase channel.
        if not problems:
            for start in range(0, mosaic.shape[0], strip_rows):
                stop = min(start + strip_rows, mosaic.shape[0])
                for row in range(2):
                    first = start + (row - start) % 2
                    for column in range(2):
                        expected = np.asarray(rgb[first:stop:2, column::2, pattern[2 * row + column]])
                        actual = np.asarray(mosaic[first:stop:2, column::2])
                        mismatches += int(np.count_nonzero(actual != expected))
            if mismatches:
                problems.append(f"{mismatches} photosites do not match the {phase} phase")
        return ValidationCheck(
            "cfa_phase",
            "CFA",
            not problems,
            hard=True,
            measured={"phase": phase, "mismatched_photosites": mismatches, "shape": list(mosaic.shape)},
            target="one value/photosite; correct periodic phase",
            detail="; ".join(problems),
        )


class ADCContractValidator:
    """PRD ADC gate: stored codes obey the black/white/bit-depth contract."""

    def validate(
        self,
        codes: np.ndarray,
        *,
        black_level: float,
        white_level: int,
        adc_bits: int,
        strip_rows: int = 128,
    ) -> ValidationCheck:
        problems = []
        if codes.dtype != np.uint16:
            problems.append(f"codes dtype {codes.dtype} is not uint16")
        if not 0 <= black_level < white_level <= (1 << adc_bits) - 1:
            problems.append("levels violate 0 <= black < white <= 2**bits - 1")
        low, high, above, total = math.inf, -math.inf, 0, 0
        for strip in _strips(codes, strip_rows):
            low, high = min(low, int(strip.min())), max(high, int(strip.max()))
            above += int(np.count_nonzero(strip > white_level))
            total += strip.size
        if above:
            problems.append(f"{above} codes exceed WhiteLevel")
        return ValidationCheck(
            "adc_contract",
            "ADC",
            not problems,
            hard=True,
            measured={
                "min_code": low,
                "max_code": high,
                "codes_above_white": above,
                "black_level": black_level,
                "white_level": white_level,
                "adc_bits": adc_bits,
                "photosites": total,
            },
            target="uint16 codes within [0, WhiteLevel]; levels fit ADC_BITS",
            detail="; ".join(problems),
        )


def _matrix(records: Mapping[str, DNGFieldRecord], name: str) -> np.ndarray:
    return np.asarray(records[name].value, dtype=float).reshape(3, 3)


def libraw_camera_to_srgb(color_matrix: np.ndarray) -> np.ndarray:
    """LibRaw's rgb_cam: inverse of the row-normalised camera-from-linear-sRGB matrix."""
    camera_rgb = np.asarray(color_matrix, dtype=float).reshape(3, 3) @ _SRGB_TO_XYZ
    camera_rgb = camera_rgb / camera_rgb.sum(axis=1, keepdims=True)
    return np.linalg.inv(camera_rgb)


class ColourValidator:
    """PRD colour-profile consistency of the DNG ColorMatrix/AsShotNeutral contract."""

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self, records: Mapping[str, DNGFieldRecord], camera_samples: np.ndarray | None = None
    ) -> list[ValidationCheck]:
        """``camera_samples`` are optional camera-native RGB pixels (N x 3) to transform."""
        checks = []
        matrices = {name: _matrix(records, name) for name in ("ColorMatrix1", "ColorMatrix2")}
        dets = {name: float(np.linalg.det(m)) for name, m in matrices.items()}
        conds = {name: float(np.linalg.cond(m)) for name, m in matrices.items()}
        checks.append(
            ValidationCheck(
                "color_matrix_invertible",
                "Colour",
                all(math.isfinite(d) and abs(d) > 1e-12 for d in dets.values()),
                hard=True,
                measured={"determinant": dets, "condition_number": conds},
                target="ColorMatrix determinant != 0; condition number reported",
            )
        )
        neutral = np.asarray(records["AsShotNeutral"].value, dtype=float)
        # The camera neutral must map back to a physically plausible white chromaticity.
        xyz = np.linalg.solve(matrices["ColorMatrix2"], neutral)
        chroma = xyz[:2] / xyz.sum() if xyz.sum() > 0 else np.array([math.nan, math.nan])
        d65_camera = matrices["ColorMatrix2"] @ _D65_XYZ
        d65_spread = float(d65_camera.max() / d65_camera.min() - 1) if d65_camera.min() > 0 else math.inf
        checks.append(
            ValidationCheck(
                "white_point_near_neutral",
                "Colour",
                bool(np.all(xyz > 0) and 0.2 < chroma[0] < 0.6 and 0.2 < chroma[1] < 0.5),
                measured={
                    "as_shot_white_xy": chroma.tolist(),
                    "d65_camera_response": d65_camera.tolist(),
                    "d65_camera_channel_spread": d65_spread,
                },
                target="AsShotNeutral maps to a positive XYZ white with plausible chromaticity",
            )
        )
        # LibRaw-equivalent render: WB by 1/AsShotNeutral then rgb_cam must give equal channels.
        rgb_cam = libraw_camera_to_srgb(matrices["ColorMatrix2"])
        balanced = (neutral / neutral) * 0.5
        rendered = rgb_cam @ balanced
        deviation = float(np.ptp(rendered) / rendered.mean()) if rendered.mean() > 0 else math.inf
        checks.append(
            ValidationCheck(
                "as_shot_neutral_response",
                "Colour",
                deviation <= self.thresholds.neutral_tolerance,
                measured={"rendered_neutral": rendered.tolist(), "relative_channel_spread": deviation},
                target=f"neutral render channel spread <= {self.thresholds.neutral_tolerance}",
            )
        )
        weights = np.linspace(0, 1, 11)
        interpolated = [w * matrices["ColorMatrix1"] + (1 - w) * matrices["ColorMatrix2"] for w in weights]
        finite = [bool(np.isfinite(m).all() and abs(np.linalg.det(m)) > 1e-12) for m in interpolated]
        checks.append(
            ValidationCheck(
                "color_matrix_interpolation",
                "Colour",
                all(finite),
                hard=True,
                measured={
                    "weights": weights.tolist(),
                    "condition_numbers": [float(np.linalg.cond(m)) for m in interpolated],
                },
                target="ColorMatrix1/2 interpolation stays finite and invertible",
            )
        )
        ramp = np.outer(np.linspace(0.01, 1, 16), neutral)
        samples = ramp if camera_samples is None else np.vstack([ramp, np.asarray(camera_samples).reshape(-1, 3)])
        transformed = (samples / neutral) @ rgb_cam.T
        nan = int(np.count_nonzero(~np.isfinite(transformed)))
        ramp_negative = int(np.count_nonzero(transformed[: len(ramp)] < 0))
        negative_fraction = float(np.mean(transformed < 0)) if transformed.size else 0.0
        checks.append(
            ValidationCheck(
                "transform_finite_nonnegative",
                "Colour",
                nan == 0 and ramp_negative == 0,
                hard=True,
                measured={
                    "nonfinite": nan,
                    "neutral_ramp_negative": ramp_negative,
                    "negative_fraction_all_samples": negative_fraction,
                    "samples": int(len(samples)),
                },
                target="no NaN; neutral ramp non-negative before clipping (out-of-gamut samples reported)",
            )
        )
        return checks


@dataclass
class NoiseFit:
    """Flat-field photon-transfer fit in normalised DNG units."""

    levels: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    slope: float
    offset: float
    r2: float
    residuals: np.ndarray
    samples: dict[float, np.ndarray]


class NoiseValidator:
    """PRD flat-field test: N >= 32 realisations per level, fit Var = S*mean + O, compare NoiseProfile.

    Flats use the run's sensor model and ADC with fixed body effects disabled, because
    the DNG NoiseProfile models temporal shot/read noise only. Variance is temporal (across
    realisations, per photosite), so any residual fixed pattern would cancel anyway.
    """

    DEFAULT_LEVELS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 0.6)

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def measure(
        self,
        config: ExperimentConfig,
        *,
        full_well_electrons: float,
        conversion_gain_e_per_dn: float,
        read_noise_e: float,
        black_level: float,
        white_level: int,
        levels: Iterable[float] = DEFAULT_LEVELS,
        realisations: int = 32,
        patch: int = 32,
        phase: str = "RGGB",
    ) -> tuple[NoiseFit, tuple[float, ...] | None]:
        if realisations < 32:
            raise ValueError("PRD requires N >= 32 realisations per level")
        flat_config = replace(
            config,
            PRNU=False,
            DSNU=False,
            ROW_FPN=False,
            COLUMN_FPN=False,
            DARK_CURRENT=False,
            HOT_PIXELS=False,
            DEAD_PIXELS=False,
        )
        levels = np.asarray(tuple(levels), dtype=float)
        span = white_level - black_level
        means, variances, samples = [], [], {}
        profile = None
        for level in levels:
            expected = np.full((patch, patch, 3), level * full_well_electrons)
            stack = np.empty((realisations, patch, patch))
            for index in range(realisations):
                seeded = replace(flat_config, SENSOR_NOISE_SEED=flat_config.SENSOR_NOISE_SEED * 1000 + index + 1)
                noisy = SensorNoiseSimulator().simulate(expected, seeded, read_noise_e=read_noise_e)
                result = ADCSimulator().simulate(
                    CFAMosaicer(phase).mosaic(noisy),
                    seeded,
                    conversion_gain_e_per_dn=conversion_gain_e_per_dn,
                    read_noise_e=read_noise_e,
                    black_level=black_level,
                    white_level=white_level,
                )
                profile = result.noise_profile
                stack[index] = (result.codes.astype(float) - black_level) / span
            means.append(float(stack.mean()))
            variances.append(float(stack.var(axis=0, ddof=1).mean()))
            samples[float(level)] = stack[0].ravel()
        means_a, variances_a = np.asarray(means), np.asarray(variances)
        # Heteroscedastic WLS: Var(sample variance) scales with variance squared.
        weights = 1.0 / np.maximum(variances_a, np.finfo(float).tiny)
        design = np.column_stack([means_a, np.ones_like(means_a)]) * weights[:, None]
        (slope, offset), *_ = np.linalg.lstsq(design, variances_a * weights, rcond=None)
        predicted = slope * means_a + offset
        residuals = variances_a - predicted
        weighted_residual = (residuals * weights) ** 2
        weighted_mean = np.average(variances_a, weights=weights**2)
        total = ((variances_a - weighted_mean) * weights) ** 2
        r2 = float(1 - weighted_residual.sum() / total.sum()) if total.sum() > 0 else math.nan
        fit = NoiseFit(levels, means_a, variances_a, float(slope), float(offset), r2, residuals, samples)
        return fit, profile

    def validate(self, fit: NoiseFit, noise_profile: Iterable[float] | None) -> list[ValidationCheck]:
        tol = self.thresholds.noise_profile_rel_tol
        if noise_profile is None:
            profile_check = ValidationCheck(
                "noise_profile_agreement",
                "Noise",
                None,
                measured={"fitted_S": fit.slope, "fitted_O": fit.offset},
                detail="NoiseProfile omitted because shot noise is disabled; nothing to compare",
            )
        else:
            values = np.asarray(tuple(noise_profile), dtype=float).reshape(-1, 2)
            slope, offset = float(values[:, 0].mean()), float(values[:, 1].mean())
            slope_error = abs(fit.slope - slope) / slope
            offset_error = abs(fit.offset - offset) / offset if offset > 0 else abs(fit.offset)
            profile_check = ValidationCheck(
                "noise_profile_agreement",
                "Noise",
                bool(slope_error <= tol and offset_error <= tol),
                measured={
                    "fitted_S": fit.slope,
                    "fitted_O": fit.offset,
                    "profile_S": slope,
                    "profile_O": offset,
                    "S_relative_error": slope_error,
                    "O_relative_error": offset_error,
                },
                target=f"S and O within {tol:.0%} of NoiseProfile",
            )
        r2_check = ValidationCheck(
            "noise_fit_r2",
            "Noise",
            bool(math.isfinite(fit.r2) and fit.r2 >= self.thresholds.noise_fit_r2),
            measured={"r2": fit.r2, "levels": len(fit.levels)},
            target=f"R^2 >= {self.thresholds.noise_fit_r2}",
        )
        return [profile_check, r2_check]


def _rawpy() -> Any:
    import rawpy

    return rawpy


def _downsample(rgb: np.ndarray, max_side: int) -> np.ndarray:
    step = max(1, math.ceil(max(rgb.shape[:2]) / max_side))
    return np.ascontiguousarray(rgb[::step, ::step])


def _develop(path: str | Path, **kwargs: Any) -> np.ndarray:
    rawpy = _rawpy()
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(output_bps=16, no_auto_bright=True, user_flip=0, **kwargs)


class DemosaicValidator:
    """PRD sequence: develop with default AHD, AMAZE where supported, plus further algorithms."""

    ALGORITHMS = ("AHD", "AMAZE", "DCB", "VNG", "PPG", "LINEAR")

    def develop(
        self, dng_path: str | Path, *, algorithms: Iterable[str] = ALGORITHMS, thumbnail_side: int = 512
    ) -> tuple[list[ValidationCheck], dict[str, np.ndarray]]:
        """Return checks and downsampled linear renders; full frames are dropped immediately."""
        rawpy = _rawpy()
        status: dict[str, Any] = {}
        thumbs: dict[str, np.ndarray] = {}
        for name in algorithms:
            algorithm = rawpy.DemosaicAlgorithm[name]
            if not algorithm.isSupported:
                status[name] = {"status": "unsupported", "detail": "not compiled into this LibRaw build"}
                continue
            try:
                rgb = _develop(dng_path, demosaic_algorithm=algorithm, use_camera_wb=True, gamma=(1, 1))
            except Exception as exc:
                status[name] = {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
                continue
            finite = bool(np.isfinite(rgb).all())
            status[name] = {
                "status": "ok" if finite and rgb.ndim == 3 else "invalid",
                "shape": list(rgb.shape),
                "channel_means": (rgb.reshape(-1, 3).mean(axis=0) / 65535).tolist(),
            }
            thumbs[name] = _downsample(rgb, thumbnail_side).astype(np.float32) / 65535
            del rgb
        if "AHD" in thumbs:
            for name, thumb in thumbs.items():
                status[name]["mean_abs_diff_vs_AHD"] = float(np.abs(thumb - thumbs["AHD"]).mean())
        developed = [name for name, entry in status.items() if entry["status"] == "ok"]
        extra = [name for name in developed if name not in ("AHD", "AMAZE")]
        problems = [name for name, entry in status.items() if entry["status"] in ("failed", "invalid")]
        checks = [
            ValidationCheck(
                "demosaic_ahd",
                "Demosaic",
                "AHD" in developed,
                hard=True,
                measured={"AHD": status.get("AHD")},
                target="default AHD development succeeds",
            ),
            ValidationCheck(
                "demosaic_additional",
                "Demosaic",
                bool(extra) and not problems,
                measured={"algorithms": status, "additional_developed": extra},
                target="AMAZE where supported plus at least one more algorithm",
                detail="AMAZE " + status.get("AMAZE", {"status": "not requested"})["status"],
            ),
        ]
        return checks, thumbs


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


class EditabilityValidator:
    """PRD -4..+4 EV and cool/neutral/warm WB development stress test.

    Exposure is applied as a linear scale of the 0 EV linear LibRaw render (rawpy's
    exp_shift only spans -2..+3 EV); WB uses rawpy user_wb so LibRaw itself re-renders.
    """

    EXPOSURES = tuple(range(-4, 5))
    WB_SHIFT = 1.25

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self, dng_path: str | Path, *, thumbnail_side: int = 256
    ) -> tuple[list[ValidationCheck], dict[str, dict[str, np.ndarray]]]:
        rawpy = _rawpy()
        linear = _develop(dng_path, use_camera_wb=True, gamma=(1, 1)).astype(np.float32) / 65535
        exposure_means, exposure_thumbs, nonfinite = [], {}, 0
        for ev in self.EXPOSURES:
            rendered = _srgb_encode(linear * np.float32(2.0**ev))
            nonfinite += int(np.count_nonzero(~np.isfinite(rendered)))
            exposure_means.append(float(rendered.mean()))
            exposure_thumbs[f"{ev:+d} EV"] = _downsample(rendered, thumbnail_side)
        del linear
        steps = np.diff(exposure_means)
        monotonic = bool(np.all(steps >= -1e-6))
        # A discontinuity is a step much larger than both of its neighbours.
        jumps = [
            i for i in range(1, len(steps) - 1) if steps[i] > 3 * max(steps[i - 1], steps[i + 1]) and steps[i] > 0.05
        ]
        with rawpy.imread(str(dng_path)) as raw:
            camera_wb = np.asarray(raw.camera_whitebalance, dtype=float)
        if camera_wb[3] == 0:
            camera_wb[3] = camera_wb[1]
        shifts = {"cool": 1 / self.WB_SHIFT, "neutral": 1.0, "warm": self.WB_SHIFT}
        wb_stats, wb_thumbs, wb_nonfinite = {}, {}, 0
        for name, shift in shifts.items():
            gains = camera_wb * np.array([shift, 1, 1 / shift, 1])
            rgb = _develop(dng_path, user_wb=gains.tolist(), gamma=(1, 1)).astype(np.float32) / 65535
            wb_nonfinite += int(np.count_nonzero(~np.isfinite(rgb)))
            means = rgb.reshape(-1, 3).mean(axis=0)
            wb_stats[name] = {
                "channel_means": means.tolist(),
                "r_over_g": float(means[0] / means[1]) if means[1] > 0 else math.inf,
                "b_over_g": float(means[2] / means[1]) if means[1] > 0 else math.inf,
                "clipped_fraction": float(np.mean(rgb >= 1.0)),
            }
            wb_thumbs[name] = _downsample(_srgb_encode(rgb), thumbnail_side)
            del rgb
        order = ("cool", "neutral", "warm")
        r_ratios = [wb_stats[name]["r_over_g"] for name in order]
        b_ratios = [wb_stats[name]["b_over_g"] for name in order]
        wb_monotonic = bool(np.all(np.diff(r_ratios) > 0) and np.all(np.diff(b_ratios) < 0))
        clipped_increase = max(wb_stats[n]["clipped_fraction"] for n in order) - wb_stats["neutral"]["clipped_fraction"]
        blow_up = clipped_increase > self.thresholds.wb_max_clipped_increase
        checks = [
            ValidationCheck(
                "exposure_range",
                "Exposure",
                monotonic and not jumps and nonfinite == 0,
                measured={
                    "ev": list(self.EXPOSURES),
                    "mean_display_value": exposure_means,
                    "nonfinite": nonfinite,
                    "discontinuity_steps": jumps,
                },
                target="-4..+4 EV: monotonic, no unexpected discontinuities or NaNs",
            ),
            ValidationCheck(
                "white_balance_response",
                "WB",
                wb_monotonic and not blow_up and wb_nonfinite == 0,
                measured={"renders": wb_stats, "clipped_increase": clipped_increase, "nonfinite": wb_nonfinite},
                target="cool -> warm: R/G increases, B/G decreases, no channel blow-up",
            ),
        ]
        return checks, {"exposure": exposure_thumbs, "white_balance": wb_thumbs}


def libraw_reopen_check(
    *,
    dng_path: str | Path,
    mosaic: np.ndarray,
    records: dict[str, DNGFieldRecord],
) -> list[ValidationCheck]:
    """Independent reopen as a check row; a HardValidationError becomes a failed hard gate."""
    try:
        LibRawValidator().validate(dng_path, mosaic, records)
        return [ValidationCheck("libraw_reopen", "DNG structure", True, hard=True, target="Hard pass")]
    except HardValidationError as error:
        return [ValidationCheck("libraw_reopen", "DNG structure", False, hard=True, detail=str(error))]


def block_mean(array: np.ndarray, factor: int, *, strip_blocks: int = 64) -> np.ndarray:
    """Area-average an HW[C] array by an integer factor, reading bounded strips of rows."""
    if type(factor) is not int or factor < 1:
        raise ValueError("factor must be a positive integer")
    height, width = (array.shape[0] // factor) * factor, (array.shape[1] // factor) * factor
    if height == 0 or width == 0:
        raise ValueError("array is smaller than one block")
    rows = []
    for start in range(0, height, factor * strip_blocks):
        strip = np.asarray(array[start : min(start + factor * strip_blocks, height), :width], dtype=np.float64)
        shape = (strip.shape[0] // factor, factor, width // factor, factor) + strip.shape[2:]
        rows.append(strip.reshape(shape).mean(axis=(1, 3)))
    return np.concatenate(rows).astype(np.float32)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    from skimage.color import xyz2lab

    return xyz2lab(np.clip(xyz, 0, None) / _D65_XYZ[1], illuminant="D65", observer="2")


def _smoothstep(linear: np.ndarray) -> np.ndarray:
    values = np.clip(linear, 0, 1)
    return values * values * (3 - 2 * values)


def _colour_align(xyz: np.ndarray, target_xyz: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Global 3x3 least-squares colour alignment fitted on supported pixels."""
    matrix, *_ = np.linalg.lstsq(xyz[mask].astype(np.float64), target_xyz[mask].astype(np.float64), rcond=None)
    return xyz @ matrix, matrix.T


@dataclass
class RoundTripResult:
    """Colour-managed round-trip comparison at the analysed resolution."""

    checks: list[ValidationCheck]
    difference: np.ndarray
    reference_srgb: np.ndarray
    developed_srgb: np.ndarray
    supported: np.ndarray


class RoundTripValidator:
    """PRD round trip: developed master versus the source-derived target canvas.

    The developed master is decoded from Adobe RGB (LibRaw's 563/256 gamma) into the
    source's linear primaries. It is a linear development without a display tone curve,
    so the gated metrics follow the PRD's "after alignment/tone normalisation": a global
    3x3 colour alignment fitted on source-supported pixels against the pipeline's
    scene-linear target, then the same display rendering the pipeline inverted
    (smoothstep when ``undo_tone``) and the source transfer. Unaligned and colour-only
    (no tone) metrics are reported alongside, so the normalisation is never hidden.
    """

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self,
        reference_encoded: np.ndarray,
        developed: np.ndarray,
        supported_mask: np.ndarray,
        *,
        transfer: str = "SRGB",
        undo_tone: bool = True,
        max_side: int = 2048,
    ) -> RoundTripResult:
        from skimage.color import deltaE_ciede2000, lab2rgb
        from skimage.metrics import structural_similarity

        from synthraw.colour_space import rgb_to_xyz_matrix
        from synthraw.exporters import ADOBE_GAMMA
        from synthraw.hdr import inverse_smoothstep, inverse_transfer

        if reference_encoded.shape != developed.shape or reference_encoded.shape[:2] != supported_mask.shape:
            raise ValueError("reference, developed master and support mask must share one grid")
        if developed.dtype != np.uint16:
            raise ValueError("developed master must be 16-bit")
        factor = max(1, math.ceil(max(developed.shape[:2]) / max_side))
        reference = np.clip(block_mean(reference_encoded, factor), 0, 1)
        master = np.clip(block_mean(developed, factor) / 65535, 0, 1)
        supported = block_mean(supported_mask.astype(np.float32), factor) >= 0.999
        if min(master.shape[:2]) < 7 or np.count_nonzero(supported) < 16:
            raise ValueError("round trip needs at least 7x7 analysed pixels and 16 supported pixels")
        to_xyz = rgb_to_xyz_matrix(transfer)
        display_linear = inverse_transfer(reference, transfer).astype(np.float64)
        scene_target = inverse_smoothstep(display_linear.astype(np.float32)) if undo_tone else display_linear
        # Developed Adobe RGB -> XYZ -> source primaries, still linear and without a tone curve.
        to_source = np.linalg.solve(to_xyz, rgb_to_xyz_matrix("ADOBE_RGB"))
        master_linear = (master.astype(np.float64) ** ADOBE_GAMMA) @ to_source.T
        target_lab = _xyz_to_lab(display_linear @ to_xyz.T)

        def metrics(display: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
            lab = _xyz_to_lab(np.clip(display, 0, None) @ to_xyz.T)
            _, ssim_map = structural_similarity(
                target_lab[..., 0] / 100, lab[..., 0] / 100, data_range=1.0, full=True, win_size=7
            )
            delta = deltaE_ciede2000(target_lab, lab)
            return (
                {
                    "supported_ssim": float(ssim_map[supported].mean()),
                    "whole_ssim": float(ssim_map.mean()),
                    "median_delta_e00_supported": float(np.median(delta[supported])),
                    "p95_delta_e00_supported": float(np.percentile(delta[supported], 95)),
                },
                lab,
                delta,
            )

        unaligned, _, _ = metrics(master_linear)
        colour_only, _, _ = metrics(_colour_align(master_linear, display_linear, supported)[0])
        aligned_scene, matrix = _colour_align(master_linear, scene_target, supported)
        rendered = _smoothstep(aligned_scene) if undo_tone else np.clip(aligned_scene, 0, 1)
        aligned, aligned_lab, delta = metrics(rendered)
        thresholds = self.thresholds
        common = {
            "analysis_factor": factor,
            "analysed_shape": list(master.shape[:2]),
            "supported_fraction": float(supported.mean()),
            "transfer": transfer,
        }
        alignment = "global 3x3 colour matrix on supported pixels, then " + (
            "the smoothstep display tone the pipeline inverted" if undo_tone else "no tone curve"
        )
        checks = [
            ValidationCheck(
                "roundtrip_supported_ssim",
                "Round trip",
                aligned["supported_ssim"] >= thresholds.roundtrip_supported_ssim,
                measured={"ssim": aligned["supported_ssim"], **common},
                target=f">= {thresholds.roundtrip_supported_ssim} after alignment/tone normalisation",
                detail=alignment,
            ),
            ValidationCheck(
                "roundtrip_whole_ssim",
                "Round trip",
                aligned["whole_ssim"] >= thresholds.roundtrip_whole_ssim,
                measured={"ssim": aligned["whole_ssim"], **common},
                target=f">= {thresholds.roundtrip_whole_ssim}",
                detail=alignment,
            ),
            ValidationCheck(
                "colour_delta_e00",
                "Colour",
                aligned["median_delta_e00_supported"] <= thresholds.median_delta_e00,
                measured={
                    "median_delta_e00": aligned["median_delta_e00_supported"],
                    "p95_delta_e00": aligned["p95_delta_e00_supported"],
                    "alignment_matrix": matrix,
                    **common,
                },
                target=f"median <= {thresholds.median_delta_e00} on source-supported pixels after global alignment",
                detail=alignment,
            ),
            ValidationCheck(
                "roundtrip_unaligned",
                "Round trip",
                None,
                measured={"unaligned": unaligned, "colour_alignment_only": colour_only, **common},
                target="report only",
                detail="Colour-managed comparison without alignment, and with colour-only alignment and no tone",
            ),
        ]
        return RoundTripResult(
            checks,
            delta.astype(np.float32),
            reference.astype(np.float32),
            np.clip(lab2rgb(aligned_lab, illuminant="D65"), 0, 1).astype(np.float32),
            supported,
        )


class SeamValidator:
    """PRD SR seam gate from the stored boundary/interior residual diagnostic."""

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def validate(self, sr_report: Mapping[str, Any]) -> ValidationCheck:
        seams = sr_report.get("seams")
        if seams is None:
            return ValidationCheck(
                "sr_seam_ratio",
                "SR seams",
                None,
                measured={"tile_count": sr_report.get("tiled", {}).get("tile_count")},
                target=f"< {self.thresholds.seam_ratio}",
                detail=f"not applicable: {sr_report.get('seam_unavailable_reason') or 'no seam diagnostic'}",
            )
        ratio = float(seams["ratio"])
        return ValidationCheck(
            "sr_seam_ratio",
            "SR seams",
            ratio < self.thresholds.seam_ratio,
            measured={k: seams[k] for k in ("boundary_mean", "interior_mean", "ratio")},
            target=f"< {self.thresholds.seam_ratio}",
        )


class HDRValidator:
    """PRD HDR rows: zero negative radiance (hard) and the generated area (report only)."""

    def validate(
        self, radiance: np.ndarray, hdr_report: Mapping[str, Any], *, strip_rows: int = 128
    ) -> list[ValidationCheck]:
        negative = nonfinite = 0
        for strip in _strips(radiance, strip_rows):
            finite = np.isfinite(strip)
            nonfinite += int(strip.size - np.count_nonzero(finite))
            negative += int(np.count_nonzero(np.any(finite & (strip < 0), axis=-1)))
        return [
            ValidationCheck(
                "hdr_negative_radiance",
                "HDR",
                negative == 0 and nonfinite == 0,
                hard=True,
                measured={"negative_pixels": negative, "nonfinite_samples": nonfinite},
                target="0 pixels",
            ),
            ValidationCheck(
                "hdr_generated_area",
                "HDR",
                None,
                measured={
                    "generated_hdr_area_percentage": hdr_report.get("generated_hdr_area_percentage"),
                    "backend": hdr_report.get("backend"),
                    "gamut_clipped_pixel_percentage": hdr_report.get("gamut_mapping", {}).get(
                        "affected_pixel_percentage"
                    ),
                },
                target="report percentage, no arbitrary pass",
            ),
        ]
