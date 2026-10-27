"""Electron-domain forward model for the PRD_v3 sensor milestone."""

from __future__ import annotations

import math

import numpy as np

from .config import ExperimentConfig


def realisation_noise_seeds(sensor_noise_seed: int, count: int) -> tuple[int, ...]:
    """SENSOR_NOISE_SEED for each of ``count`` noise realisations of one exposure and sensor body.

    Realisation 1 keeps ``sensor_noise_seed``. Each later realisation draws an independent 63-bit seed
    from ``SeedSequence([sensor_noise_seed, index])``, so neighbouring base seeds never share
    realisations. Running with SENSOR_NOISE_SEED set to a recorded seed reproduces that realisation.
    """
    if type(sensor_noise_seed) is not int or sensor_noise_seed < 0:
        raise ValueError("sensor_noise_seed must be a nonnegative integer")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    seeds = [sensor_noise_seed]
    for index in range(2, count + 1):
        seed = int(np.random.SeedSequence([sensor_noise_seed, index]).generate_state(1, np.uint64)[0] >> 1)
        if seed in seeds:
            raise RuntimeError("Derived sensor noise seeds collided")
        seeds.append(seed)
    return tuple(seeds)


class ElectronSignalSimulator:
    """Convert pre-CFA linear RGB to expected electrons before sensor noise.

    Scene signal 1 at exposure_scale 1 corresponds to full well. HDR values
    remain above full well until the later sensor/ADC saturation stage.
    Exposure scale is a linear multiplier, not an EV value or analogue gain.
    """

    def simulate(
        self,
        scene_signal: np.ndarray,
        full_well_electrons: float,
        *,
        exposure_scale: float = 1.0,
        strip_rows: int = 128,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply max(signal, 0) * full_well * exposure_scale in bounded strips.

        An optional float64 output (including an NPY memmap) avoids a second
        frame allocation. Validate all input strips before modifying output.
        """
        for name, value in (("full_well_electrons", full_well_electrons), ("exposure_scale", exposure_scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name == "full_well_electrons" and value == 0):
                raise ValueError(f"{name} is outside the physical domain")
        scale = full_well_electrons * exposure_scale
        if not math.isfinite(scale):
            raise ValueError("electron scale must be finite")
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be a positive integer")
        if (
            not isinstance(scene_signal, np.ndarray)
            or scene_signal.ndim != 3
            or scene_signal.shape[2] != 3
            or min(scene_signal.shape) == 0
            or scene_signal.dtype.kind != "f"
        ):
            raise ValueError("scene_signal must be a nonempty floating RGB array")
        if out is not None and (
            not isinstance(out, np.ndarray)
            or out.shape != scene_signal.shape
            or out.dtype != np.float64
            or not out.flags.writeable
            or np.may_share_memory(scene_signal, out)
        ):
            raise ValueError("out must be a writable, independent float64 array matching scene_signal")
        limit = np.finfo(np.float64).max / max(scale, 1.0)
        for start in range(0, scene_signal.shape[0], strip_rows):
            strip = scene_signal[start : start + strip_rows]
            if not np.isfinite(strip).all() or np.any(strip > limit):
                raise ValueError("scene_signal must be finite and produce finite electrons")
        result = np.empty(scene_signal.shape, dtype=np.float64) if out is None else out
        for start in range(0, scene_signal.shape[0], strip_rows):
            target = result[start : start + strip_rows]
            np.maximum(scene_signal[start : start + strip_rows], 0, out=target)
            np.multiply(target, scale, out=target)
        return result


class SensorNoiseSimulator:
    """Apply PRD_v3 electron noise in bounded, reproducible row sub-streams.

    Body effects are shared by the RGB channels of each photosite. Temporal
    shot/read samples are independent per channel before CFA selection.
    """

    def simulate(
        self,
        expected_e: np.ndarray,
        config: ExperimentConfig,
        *,
        read_noise_e: float,
        exposure_time_s: float = 1.0,
        prnu_sigma: float = 0.0,
        dsnu_e: float = 0.0,
        row_fpn_e: float = 0.0,
        column_fpn_e: float = 0.0,
        dark_current_e_per_s: float = 0.0,
        hot_pixel_fraction: float = 0.0,
        dead_pixel_fraction: float = 0.0,
        hot_pixel_e_per_s: float = 1000.0,
        strip_rows: int = 128,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Use Poisson, PRNU, read, dark/DSNU, FPN, then hot/dead defects.

        Sigma and rate parameters are in electrons, except dimensionless PRNU
        and defect fractions. Dead pixels override hot pixels and yield zero.
        Negative noisy electrons are retained for the later ADC pedestal.
        """
        parameters = {
            "read_noise_e": read_noise_e,
            "exposure_time_s": exposure_time_s,
            "prnu_sigma": prnu_sigma,
            "dsnu_e": dsnu_e,
            "row_fpn_e": row_fpn_e,
            "column_fpn_e": column_fpn_e,
            "dark_current_e_per_s": dark_current_e_per_s,
            "hot_pixel_fraction": hot_pixel_fraction,
            "dead_pixel_fraction": dead_pixel_fraction,
            "hot_pixel_e_per_s": hot_pixel_e_per_s,
        }
        for name, value in parameters.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if hot_pixel_fraction > 1 or dead_pixel_fraction > 1:
            raise ValueError("defect fractions must be in [0, 1]")
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be a positive integer")
        if (
            not isinstance(expected_e, np.ndarray)
            or expected_e.ndim != 3
            or expected_e.shape[2] != 3
            or min(expected_e.shape) == 0
            or expected_e.dtype.kind != "f"
        ):
            raise ValueError("expected_e must be a nonempty floating RGB array")
        if out is not None and (
            not isinstance(out, np.ndarray)
            or out.shape != expected_e.shape
            or out.dtype != np.float64
            or not out.flags.writeable
            or np.may_share_memory(expected_e, out)
        ):
            raise ValueError("out must be a writable, independent float64 array matching expected_e")
        for start in range(0, expected_e.shape[0], strip_rows):
            strip = expected_e[start : start + strip_rows]
            if not np.isfinite(strip).all() or np.any(strip < 0) or np.any(strip > 1e18):
                raise ValueError("expected_e must be finite, nonnegative and within the Poisson domain (<= 1e18)")
        for seed in (config.SENSOR_BODY_SEED, config.SENSOR_NOISE_SEED):
            if type(seed) is not int or seed < 0:
                raise ValueError("sensor seeds must be nonnegative integers")

        def rng(seed: int, component: int, row: int = 0) -> np.random.Generator:
            return np.random.default_rng(np.random.SeedSequence([seed, component, row]))

        body = config.SENSOR_BODY_SEED
        noise = config.SENSOR_NOISE_SEED
        width = expected_e.shape[1]
        columns = rng(body, 4).normal(0, column_fpn_e, (width, 1)) if config.COLUMN_FPN else None
        result = np.empty(expected_e.shape, dtype=np.float64) if out is None else out
        for start in range(0, expected_e.shape[0], strip_rows):
            for row in range(start, min(start + strip_rows, expected_e.shape[0])):
                target = result[row]
                if config.SHOT_NOISE:
                    target[:] = rng(noise, 0, row).poisson(expected_e[row])
                else:
                    target[:] = expected_e[row]
                if config.PRNU:
                    target *= rng(body, 0, row).normal(1, prnu_sigma, (width, 1))
                if config.READ_NOISE:
                    target += rng(noise, 1, row).normal(0, read_noise_e, target.shape)
                if config.DARK_CURRENT:
                    target += dark_current_e_per_s * exposure_time_s
                if config.DSNU:
                    target += rng(body, 1, row).normal(0, dsnu_e, (width, 1))
                if config.ROW_FPN:
                    target += rng(body, 2, row).normal(0, row_fpn_e)
                if columns is not None:
                    target += columns
                if config.HOT_PIXELS:
                    hot = rng(body, 5, row).random(width) < hot_pixel_fraction
                    target[hot] += hot_pixel_e_per_s * exposure_time_s
                if config.DEAD_PIXELS:
                    dead = rng(body, 6, row).random(width) < dead_pixel_fraction
                    target[dead] = 0
        return result
