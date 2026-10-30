"""Bounded pre-CFA synthetic CA, spatial PSF, diffraction, defocus and motion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass(frozen=True)
class OpticalEffects:
    """Explicit priors, not inferred lens calibration.

    Gaussian PSFs approximate central lobes, not Airy rings. Spatial PSF blends
    center and edge Gaussian variances. Axial CA is a channel-specific blur prior,
    not a depth reconstruction. Motion is one uniform global line trajectory.
    Defaults preserve existing analytic callers; the notebook uses a full preset.
    """

    lateral_ca_px: float = 0.0
    axial_ca_sigma_px: float = 0.0
    psf_center_sigma_px: float = 0.0
    psf_edge_sigma_px: float = 0.0
    diffraction: bool = False
    wavelength_nm: float = 550.0
    defocus_radius_px: float = 0.0
    motion_length_px: float = 0.0
    motion_angle_deg: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "diffraction":
                if type(value) is not bool:
                    raise ValueError("diffraction must be boolean")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("axial_ca_sigma_px", "psf_center_sigma_px", "psf_edge_sigma_px", "defocus_radius_px"):
            if not 0 <= getattr(self, name) <= 16:
                raise ValueError(f"{name} must be in [0,16]")
        if abs(self.lateral_ca_px) > 16 or not 0 <= self.motion_length_px <= 64:
            raise ValueError("CA or motion exceeds bounded experimental range")
        if not 380 <= self.wavelength_nm <= 780:
            raise ValueError("wavelength_nm must lie in visible range 380..780")

    def states(self):
        return {
            "lateral_ca": self.lateral_ca_px != 0,
            "axial_ca": self.axial_ca_sigma_px > 0,
            "spatial_psf": max(self.psf_center_sigma_px, self.psf_edge_sigma_px) > 0,
            "diffraction": self.diffraction,
            "defocus": self.defocus_radius_px > 0,
            "motion_blur": self.motion_length_px > 0,
        }

    def process(self, source, destination, *, aperture, pixel_pitch_um, strip_rows=64):
        """Use separate disk maps, global warp coordinates and exact blur halos."""
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be positive")
        for value in (aperture, pixel_pitch_um):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("aperture and pixel_pitch_um must be positive finite numbers")
        if source.ndim != 3 or source.shape[-1] != 3 or source.dtype.kind != "f":
            raise ValueError("Expected floating RGB radiance")
        if (
            not isinstance(destination, np.memmap)
            or destination.dtype != np.float32
            or destination.shape != source.shape
        ):
            raise ValueError("Expected matching float32 output memmap")
        if not destination.flags.writeable or np.may_share_memory(source, destination):
            raise ValueError("Output must be writable and separate")
        if isinstance(source, np.memmap) and source.filename == destination.filename:
            raise ValueError("Input and output files must be separate")
        height, width = source.shape[:2]
        if min(height, width) < 1:
            raise ValueError("Empty radiance")
        for start in range(0, height, strip_rows):
            block = source[start : start + strip_rows]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError("Radiance must be finite and nonnegative")
        diffraction_sigma = (
            1.028 * self.wavelength_nm / 1000 * aperture / pixel_pitch_um / math.sqrt(8 * math.log(2))
            if self.diffraction
            else 0.0
        )
        base_sigma = math.hypot(diffraction_sigma, self.defocus_radius_px / 2)
        operations = []
        if self.lateral_ca_px:
            operations.append("warp")
        if base_sigma or max(self.axial_ca_sigma_px, self.psf_center_sigma_px, self.psf_edge_sigma_px):
            operations.append("blur")
        if self.motion_length_px:
            operations.append("motion")
        with TemporaryDirectory(prefix="optical-effects-", dir=Path(destination.filename).parent) as directory:
            maps = []
            try:
                for index in range(min(2, max(0, len(operations) - 1))):
                    maps.append(
                        np.memmap(Path(directory) / str(index), mode="w+", dtype=np.float32, shape=source.shape)
                    )
                current = source
                for index, operation in enumerate(operations):
                    target = destination if index == len(operations) - 1 else maps[index % 2]
                    for start in range(0, height, strip_rows):
                        end = min(height, start + strip_rows)
                        y, x = np.meshgrid(np.arange(start, end), np.arange(width), indexing="ij")
                        radius2 = (x + 0.5 - width / 2) ** 2 + (y + 0.5 - height / 2) ** 2
                        radius2 /= (width * width + height * height) / 4
                        if operation == "warp":
                            scale = max(width, height) / 2
                            for channel, sign in enumerate((1, 0, -1)):
                                factor = sign * self.lateral_ca_px / scale
                                sx = x + factor * (x + 0.5 - width / 2)
                                sy = y + factor * (y + 0.5 - height / 2)
                                target[start:end, :, channel] = map_coordinates(
                                    current[..., channel], [sy, sx], order=1, mode="nearest", prefilter=False
                                )
                        elif operation == "blur":
                            for channel in range(3):
                                axial = self.axial_ca_sigma_px * (1, 0, 0.75)[channel]
                                center = math.sqrt(base_sigma**2 + axial**2 + self.psf_center_sigma_px**2)
                                edge = math.sqrt(base_sigma**2 + axial**2 + self.psf_edge_sigma_px**2)
                                halo = int(4 * max(center, edge) + 0.5)
                                low, high = max(0, start - halo), min(height, end + halo)
                                patch = current[low:high, :, channel]
                                a = gaussian_filter(patch, center, mode="reflect", truncate=4)[start - low : end - low]
                                b = gaussian_filter(patch, edge, mode="reflect", truncate=4)[start - low : end - low]
                                target[start:end, :, channel] = a * (1 - radius2) + b * radius2
                        else:
                            steps = max(2, int(math.ceil(self.motion_length_px)) + 1)
                            angle = math.radians(self.motion_angle_deg)
                            target[start:end] = 0
                            for offset in np.linspace(-0.5, 0.5, steps) * self.motion_length_px:
                                sx, sy = x + offset * math.cos(angle), y + offset * math.sin(angle)
                                for channel in range(3):
                                    target[start:end, :, channel] += (
                                        map_coordinates(
                                            current[..., channel], [sy, sx], order=1, mode="nearest", prefilter=False
                                        )
                                        / steps
                                    )
                    target.flush()
                    current = target
                if not operations:
                    for start in range(0, height, strip_rows):
                        destination[start : start + strip_rows] = source[start : start + strip_rows]
                    destination.flush()
            finally:
                for array in maps:
                    array._mmap.close()
        return {
            "parameters": asdict(self),
            "states": self.states(),
            "aperture": aperture,
            "pixel_pitch_um": pixel_pitch_um,
            "diffraction_sigma_px": diffraction_sigma,
            "order": operations,
            "status": "experimental" if operations else "zero_effect",
            "boundary": "reflect for blur; nearest extension for warp and motion",
            "limitations": (
                "Synthetic approximate PSFs; axial CA has no depth model, motion is global; no lens calibration"
            ),
        }
