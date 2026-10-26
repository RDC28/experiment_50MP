"""Electron-domain forward model for the PRD_v3 sensor milestone."""

from __future__ import annotations

import math

import numpy as np


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
