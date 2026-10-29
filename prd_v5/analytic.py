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
    """

    camera_to_srgb: tuple[tuple[float, ...], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    white_balance: tuple[float, ...] = (1.0, 1.0, 1.0)
    exposure_gain: float = 1.0
    inverse_smoothstep: bool = False

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
        # Copy mutable user sequences into immutable values.
        object.__setattr__(self, "camera_to_srgb", tuple(tuple(float(v) for v in row) for row in matrix))
        object.__setattr__(self, "white_balance", tuple(float(v) for v in wb))

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
        inverse = np.linalg.inv(np.asarray(self.camera_to_srgb, dtype=np.float32))
        gains = np.asarray(self.white_balance, dtype=np.float32) * self.exposure_gain
        clipped = 0
        negative = 0
        reconstruction_max = 0.0
        reconstruction_squared = 0.0
        for start in range(0, source.shape[0], strip_rows):
            block = np.asarray(source[start : start + strip_rows], dtype=np.float32)
            if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
                raise ValueError("source must contain finite normalized [0, 1] samples")
            clipped += int(np.count_nonzero(np.any(block >= 1, axis=-1)))
            linear = np.where(block <= 0.04045, block / 12.92, ((block + 0.055) / 1.055) ** 2.4)
            if self.inverse_smoothstep:
                linear = 0.5 - np.sin(np.arcsin(1 - 2 * linear) / 3)
            camera = (linear @ inverse.T) / gains
            negative += int(np.count_nonzero(camera < 0))
            destination[start : start + len(block)] = np.maximum(camera, 0)
            # Validate the stored float32 camera values, including any irreversible
            # gamut flooring, against the declared forward model. Work stays local
            # to this strip; this is self-consistency rather than camera truth.
            stored = np.asarray(destination[start : start + len(block)], dtype=np.float64)
            forward = (stored * gains) @ np.asarray(self.camera_to_srgb).T
            forward = np.clip(forward, 0, 1)
            if self.inverse_smoothstep:
                forward = forward**2 * (3 - 2 * forward)
            encoded = np.where(forward <= 0.0031308, 12.92 * forward, 1.055 * forward ** (1 / 2.4) - 0.055)
            residual = encoded - block
            reconstruction_max = max(reconstruction_max, float(np.max(np.abs(residual))))
            reconstruction_squared += float(np.sum(residual**2, dtype=np.float64))
        destination.flush()
        return {
            "method": "analytic_srgb_unprocessing",
            "parameters": asdict(self),
            "profile_provenance": "USER hypothesis; identity default is virtual sRGB, not calibrated Sony",
            "source_clipped_pixels": clipped,
            "negative_camera_samples_floored": negative,
            "highlight_recovery": "not performed; clipped source values remain uncertain",
            "validation": "analytic transform only; scene radiance accuracy not measured",
            "forward_consistency": {
                "status": "passed" if reconstruction_max <= 1e-5 else "failed",
                "max_absolute_error": reconstruction_max,
                "rmse": float(np.sqrt(reconstruction_squared / source.size)),
                "tolerance": 1e-5,
                "units": "normalized encoded sRGB",
                "reference": "resampled input before analytic inversion",
                "method": "stored clean camera RGB -> exposure/WB -> matrix -> clip -> tone -> sRGB",
                "limitations": (
                    "Internal model consistency only, not calibrated colour or recovered radiance accuracy; "
                    "camera gamut flooring and forward display clipping can be irreversible"
                ),
            },
            "strip_rows": strip_rows,
        }
