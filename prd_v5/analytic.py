"""Bounded analytic unprocessing of explicitly sRGB-encoded input arrays."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AnalyticISP:
    """Synthetic forward-render parameters, never an inferred Sony calibration.

    Forward order: exposure * white balance, camera-to-linear-sRGB matrix,
    optional smoothstep tone, then the sRGB display transfer. Inversion reverses
    that order. Out-of-gamut camera values are floored, not silently normalized.
    HDR expansion is scaled by 1/hdr_max_radiance into sensor headroom; the
    BaselineExposure returned by baseline_exposure_ev restores 18% grey.
    """

    camera_to_srgb: tuple[tuple[float, ...], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    white_balance: tuple[float, ...] = (1.0, 1.0, 1.0)
    exposure_gain: float = 1.0
    inverse_smoothstep: bool = False
    hdr_max_radiance: float = 1.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_to_srgb, dtype=np.float64)
        wb = np.asarray(self.white_balance, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e6:
            raise ValueError("camera_to_srgb must be a finite well-conditioned 3x3 matrix")
        if wb.shape != (3,) or not np.isfinite(wb).all() or np.any(wb <= 0):
            raise ValueError("white_balance must contain three positive finite gains")
        if isinstance(self.exposure_gain, bool) or not np.isfinite(self.exposure_gain) or self.exposure_gain <= 0:
            raise ValueError("exposure_gain must be positive and finite")
        if type(self.inverse_smoothstep) is not bool:
            raise ValueError("inverse_smoothstep must be boolean")
        if (
            type(self.hdr_max_radiance) not in (int, float)
            or not np.isfinite(self.hdr_max_radiance)
            or not 1 <= self.hdr_max_radiance <= 32
        ):
            raise ValueError("hdr_max_radiance must be finite in [1, 32]")
        # Copy mutable user sequences into immutable values.
        object.__setattr__(self, "camera_to_srgb", tuple(tuple(float(v) for v in row) for row in matrix))
        object.__setattr__(self, "white_balance", tuple(float(v) for v in wb))

    @property
    def baseline_exposure_ev(self) -> float:
        """DNG BaselineExposure mapping 18% grey back after HDR headroom scaling."""
        if self.hdr_max_radiance == 1:
            return 0.0
        return float(np.log2(self.hdr_max_radiance * (1 - 0.18) + 1))

    def process(self, source: np.ndarray, destination: np.memmap, *, strip_rows: int = 64) -> dict:
        """Write float32 clean camera RGB, allocating only strip-sized intermediates.

        Input must already have an explicit sRGB colour policy and be normalized
        to [0, 1]. A decoder/ICC transform is outside this API. Disk-backed source
        arrays are supported. The caller owns both buffers and must discard a
        partial destination on failure. No noise, CFA or learned model runs here.
        """
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be a positive integer")
        if source.ndim != 3 or source.shape[-1] != 3 or min(source.shape) < 1:
            raise ValueError("source must be nonempty HxWx3")
        if source.dtype.kind != "f":
            raise ValueError("source must be normalized floating point")
        if not isinstance(destination, np.memmap) or destination.dtype != np.float32:
            raise ValueError("destination must be a float32 disk-backed memmap")
        if destination.shape != source.shape or not destination.flags.writeable:
            raise ValueError("destination must match source and be writable")
        if np.may_share_memory(source, destination):
            raise ValueError("source and destination must not overlap")
        if isinstance(source, np.memmap) and source.filename == destination.filename:
            raise ValueError("source and destination must use separate files")
        inverse = np.linalg.inv(np.asarray(self.camera_to_srgb, dtype=np.float64))
        gains = np.asarray(self.white_balance, dtype=np.float64) * self.exposure_gain
        clipped = 0
        negative = 0
        expanded = 0
        reconstruction_max = 0.0
        reconstruction_squared = 0.0
        checked = 0
        for start in range(0, source.shape[0], strip_rows):
            block = np.asarray(source[start : start + strip_rows], dtype=np.float64)
            if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
                raise ValueError("source must contain finite normalized [0, 1] samples")
            clipped += int(np.count_nonzero(np.any(block >= 1, axis=-1)))
            linear = np.where(block <= 0.04045, block / 12.92, ((block + 0.055) / 1.055) ** 2.4)
            if self.inverse_smoothstep:
                linear = 0.5 - np.sin(np.arcsin(1 - 2 * linear) / 3)
            if self.hdr_max_radiance > 1:
                # Invert y = r * (1 + 1/M) / (1 + r), with r in [0, M].
                linear = linear / (1 + 1 / self.hdr_max_radiance - linear)
                expanded += int(np.count_nonzero(np.any(linear > 1 + 1e-6, axis=-1)))
                linear = linear / self.hdr_max_radiance
            camera = (linear @ inverse.T) / gains
            negative += int(np.count_nonzero(camera < 0))
            destination[start : start + len(block)] = np.maximum(camera, 0)
            # Validate the stored float32 camera values against the declared forward
            # model; irreversibly gamut-floored pixels are counted above instead.
            # Work stays local to this strip; this is self-consistency, not truth.
            inside = np.all(camera >= 0, axis=-1)
            stored = np.asarray(destination[start : start + len(block)], dtype=np.float64)
            forward = (stored * gains) @ np.asarray(self.camera_to_srgb).T
            if self.hdr_max_radiance > 1:
                forward = np.maximum(forward, 0) * self.hdr_max_radiance
                forward = forward * (1 + 1 / self.hdr_max_radiance) / (1 + forward)
            forward = np.clip(forward, 0, 1)
            if self.inverse_smoothstep:
                forward = forward**2 * (3 - 2 * forward)
            encoded = np.where(forward <= 0.0031308, 12.92 * forward, 1.055 * forward ** (1 / 2.4) - 0.055)
            residual = (encoded - block)[inside]
            checked += residual.size
            reconstruction_max = max(reconstruction_max, float(np.max(np.abs(residual), initial=0)))
            reconstruction_squared += float(np.sum(residual**2, dtype=np.float64))
        destination.flush()
        if reconstruction_max > 1e-5:
            raise ValueError(f"Analytic forward consistency failed: max error {reconstruction_max:.3g}")
        return {
            "method": "analytic_srgb_unprocessing",
            "parameters": asdict(self),
            "profile_provenance": "USER hypothesis; identity default is virtual sRGB, not calibrated Sony",
            "source_clipped_pixels": clipped,
            "negative_camera_samples_floored": negative,
            "highlight_recovery": "not performed; clipped source values remain uncertain",
            "hdr": {
                "requested": self.hdr_max_radiance > 1,
                "effective": self.hdr_max_radiance > 1,
                "status": "experimental analytic hypothesis" if self.hdr_max_radiance > 1 else "disabled",
                "method": "normalized inverse Reinhard after inverse smoothstep, before camera transform",
                "max_linear_srgb_radiance": self.hdr_max_radiance,
                "expanded_pixels": expanded,
                "uncertainty": "All expanded radiance is a tone-curve hypothesis; endpoints remain unknown. "
                "source_clipped.npy marks resampled endpoints only, not every uncertain HDR pixel.",
                "learned_hdr": "not executed; no selected compatible checkpoint",
                "raw_exposure_offset_ev": -float(np.log2(self.hdr_max_radiance)),
                "baseline_exposure_ev": self.baseline_exposure_ev,
                "sensor_limit": "hdr_max_radiance maps to sensor saturation; DNG BaselineExposure restores 18% grey",
            },
            "validation": "analytic transform only; scene radiance accuracy not measured",
            "forward_consistency": {
                "status": "passed",
                "max_absolute_error": reconstruction_max,
                "rmse": float(np.sqrt(reconstruction_squared / max(checked, 1))),
                "checked_samples": checked,
                "tolerance": 1e-5,
                "units": "normalized encoded sRGB",
                "reference": "resampled input before analytic inversion",
                "method": "camera RGB -> exposure/WB -> matrix -> optional HDR compression -> clip -> tone -> sRGB",
                "limitations": (
                    "Internal model consistency only, not calibrated colour or recovered radiance accuracy; "
                    "gamut-floored pixels are excluded and counted in negative_camera_samples_floored"
                ),
            },
            "strip_rows": strip_rows,
        }
