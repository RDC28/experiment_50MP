"""Bounded synthetic optical attenuation in clean linear camera space."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Vignetting:
    """Centered achromatic radial prior, not a calibrated lens model.

    Radius uses pixel centers normalized by half the sensor diagonal. Thus
    corner_strength describes attenuation at the continuous sensor boundary,
    not necessarily at the outermost pixel center. Transmission is 1 - k*r**2.
    """

    corner_strength: float = 0.0

    def __post_init__(self):
        if (
            type(self.corner_strength) not in (int, float)
            or not np.isfinite(self.corner_strength)
            or not 0 <= self.corner_strength <= 1
        ):
            raise ValueError("corner_strength must be finite in [0, 1]")

    def process(self, source, destination, *, strip_rows=64, accept_experimental=False):
        """Attenuate linear RGB before CFA and sensor noise, without clipping HDR.

        Caller owns separate buffers and must discard partial output on failure.
        The API does not infer a lens or correct shading already in a source.
        """
        if self.corner_strength and accept_experimental is not True:
            raise ValueError("Explicit experimental acceptance is required")
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be a positive integer")
        if source.ndim != 3 or source.shape[-1] != 3 or min(source.shape) < 1 or source.dtype.kind != "f":
            raise ValueError("source must be nonempty floating HxWx3 linear camera RGB")
        if (
            not isinstance(destination, np.memmap)
            or destination.dtype != np.float32
            or destination.shape != source.shape
            or not destination.flags.writeable
        ):
            raise ValueError("destination must be writable float32 memmap matching source")
        if np.may_share_memory(source, destination) or (
            isinstance(source, np.memmap) and source.filename == destination.filename
        ):
            raise ValueError("source and destination must use separate nonoverlapping buffers")
        height, width = source.shape[:2]
        x2 = (np.arange(width, dtype=np.float64) + 0.5 - width / 2) ** 2
        radius_scale = (width**2 + height**2) / 4
        minimum = 1.0
        changed = 0
        for start in range(0, height, strip_rows):
            block = source[start : start + strip_rows]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError("source samples must be finite and nonnegative")
            y2 = (np.arange(start, start + len(block), dtype=np.float64) + 0.5 - height / 2) ** 2
            transmission = 1 - self.corner_strength * (y2[:, None] + x2) / radius_scale
            minimum = min(minimum, float(transmission.min()))
            stored = (block * transmission[..., None]).astype(np.float32)
            if not np.isfinite(stored).all():
                raise ValueError("attenuated samples exceed float32 storage range")
            destination[start : start + len(block)] = stored
            changed += int(np.count_nonzero(np.any(stored != block, axis=-1)))
        destination.flush()
        return {
            "method": "centered_radial_linear_camera_attenuation",
            "parameters": asdict(self),
            "status": "experimental" if self.corner_strength else "zero_effect",
            "minimum_transmission": minimum,
            "changed_pixels": changed,
            "order": "clean linear camera RGB before CFA and electron-domain noise",
            "limitations": (
                "Uncalibrated achromatic synthetic prior; no lens estimation or shading removal. "
                "No distortion, CA, PSF, diffraction, defocus or motion simulation. "
                "Full-scale and physical validation pending."
            ),
        }
