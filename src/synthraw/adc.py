"""ADC quantisation and applied temporal NoiseProfile for PRD_v3's sensor milestone."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


@dataclass(frozen=True)
class ADCResult:
    """Stored mosaic and DNG R/G/B S,O pairs, before clipping bias.

    A disabled shot-noise model cannot satisfy DNG's S > 0 contract, so its
    NoiseProfile is None rather than a fabricated positive slope. Fixed body
    effects are outside this Poisson/read/quantisation model.
    """

    codes: np.ndarray
    noise_profile: tuple[float, ...] | None
    applied_parameters: dict[str, float | int | bool]


class ADCSimulator:
    """Convert CFA electrons to uint16 with PRD_v3's conversion-gain convention."""

    def simulate(
        self,
        sensor_e: np.ndarray,
        config: ExperimentConfig,
        *,
        conversion_gain_e_per_dn: float,
        read_noise_e: float,
        analogue_gain: float = 1.0,
        black_level: float = 512.0,
        white_level: int | None = None,
        strip_rows: int = 128,
        out: np.ndarray | None = None,
    ) -> ADCResult:
        """Round electrons * analogue_gain / conversion_gain + black, then clip.

        DNG signal is black-subtracted and divided by WhiteLevel - BlackLevel.
        With k = analogue_gain / conversion_gain, S = k / range and
        O = (read_noise_e * k / range)**2 + 1 / (12 * range**2), where the last term is
        the uniform rounding variance of one DN in the stored codes. Parameters describe this exact call;
        read/shot toggles match SensorNoiseSimulator. Output may be an NPY memmap.
        """
        for name, value in {
            "conversion_gain_e_per_dn": conversion_gain_e_per_dn,
            "read_noise_e": read_noise_e,
            "analogue_gain": analogue_gain,
            "black_level": black_level,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name in ("conversion_gain_e_per_dn", "analogue_gain") and value == 0):
                raise ValueError(f"{name} is outside the physical domain")
        white = (1 << config.ADC_BITS) - 1 if white_level is None else white_level
        if type(white) is not int or not black_level < white <= (1 << config.ADC_BITS) - 1:
            raise ValueError("WhiteLevel must exceed BlackLevel and fit ADC_BITS")
        if config.ADC_BITS == 16 and config.TARGET_PROFILE != "VIRTUAL_16BIT_SENSOR":
            raise ValueError("16-bit ADC requires VIRTUAL_16BIT_SENSOR")
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be a positive integer")
        if (
            not isinstance(sensor_e, np.ndarray)
            or sensor_e.ndim != 2
            or min(sensor_e.shape) == 0
            or sensor_e.dtype.kind != "f"
        ):
            raise ValueError("sensor_e must be a nonempty floating CFA array")
        if out is not None and (
            not isinstance(out, np.ndarray)
            or out.shape != sensor_e.shape
            or out.dtype != np.uint16
            or not out.flags.writeable
            or np.may_share_memory(sensor_e, out)
        ):
            raise ValueError("out must be a writable, independent uint16 array matching sensor_e")
        k = analogue_gain / conversion_gain_e_per_dn
        if not math.isfinite(k) or k <= 0:
            raise ValueError("ADC gain must be finite and representable")
        scale = k / (white - black_level)
        sigma = read_noise_e * scale if config.READ_NOISE else 0.0
        offset = sigma * sigma + 1.0 / (12.0 * (white - black_level) ** 2)
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
            raise ValueError("ADC gain and NoiseProfile must be finite and representable")
        # Validate before writing so invalid input never leaves a partial output.
        for start in range(0, sensor_e.shape[0], strip_rows):
            if not np.isfinite(sensor_e[start : start + strip_rows]).all():
                raise ValueError("sensor_e must be finite")
        codes = np.empty(sensor_e.shape, dtype=np.uint16) if out is None else out
        for start in range(0, sensor_e.shape[0], strip_rows):
            # Clip electrons first to avoid overflow on otherwise valid large inputs.
            strip = np.array(sensor_e[start : start + strip_rows], dtype=np.float64, copy=True)
            np.clip(strip, -black_level / k, (white - black_level) / k, out=strip)
            np.multiply(strip, k, out=strip)
            strip += black_level
            np.rint(strip, out=strip)
            np.clip(strip, 0, white, out=strip)
            codes[start : start + strip_rows] = strip
        return ADCResult(
            codes,
            (scale, offset) * 3 if config.SHOT_NOISE else None,
            {
                "adc_bits": config.ADC_BITS,
                "conversion_gain_e_per_dn": float(conversion_gain_e_per_dn),
                "analogue_gain": float(analogue_gain),
                "black_level": float(black_level),
                "white_level": white,
                "read_noise_e": float(read_noise_e) if config.READ_NOISE else 0.0,
                "shot_noise": config.SHOT_NOISE,
                "read_noise": config.READ_NOISE,
            },
        )
