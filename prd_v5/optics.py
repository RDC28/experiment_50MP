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


@dataclass(frozen=True)
class Distortion:
    """Synthetic Brown-Conrady backward sampling map, not lens calibration.

    Output pixel centers map to source coordinates with x/y normalized by half
    the sensor diagonal. Positive k1 samples farther outward (barrel appearance).
    Bilinear interpolation uses edge extension outside the source footprint.
    Radial maps that fold before the corners are rejected.
    """

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not np.isfinite(value) or abs(value) > 0.25:
                raise ValueError(f"{name} must be finite in [-0.25, 0.25]")
        # d/dr of r * (1 + k1 r^2 + k2 r^4) with u = r^2 over the normalized radius [0, 1].
        knots = [1.0] + ([-0.3 * self.k1 / self.k2] if self.k2 > 0 else [])
        if min(1 + 3 * self.k1 * u + 5 * self.k2 * u * u for u in knots if 0 <= u <= 1) <= 0:
            raise ValueError("Radial distortion must stay monotonic out to the sensor corners")

    def process(self, source, destination, *, strip_rows=64, accept_experimental=False):
        """Warp clean RGB before CFA with row-bounded gathers and float32 storage.

        This is an explicit output-to-input map, not numerical inversion of a
        calibrated forward projection. Caller discards partial output on failure.
        No flux/Jacobian correction, antialiasing or inferred geometry is claimed.
        """
        active = any(asdict(self).values())
        if active and accept_experimental is not True:
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
        # Validate the entire domain, including pixels not visited by the warp.
        for start in range(0, height, strip_rows):
            block = source[start : start + strip_rows]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError("source samples must be finite and nonnegative")
        scale = np.hypot(width, height) / 2
        x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2) / scale
        changed = outside = 0
        for start in range(0, height, strip_rows):
            stop = min(start + strip_rows, height)
            block = source[start:stop]
            if not active:
                stored = block.astype(np.float32)
            else:
                y = (np.arange(start, stop, dtype=np.float64)[:, None] + 0.5 - height / 2) / scale
                r2 = x**2 + y**2
                radial = 1 + self.k1 * r2 + self.k2 * r2**2
                sx = (x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x**2)) * scale
                sy = (y * radial + self.p1 * (r2 + 2 * y**2) + 2 * self.p2 * x * y) * scale
                sx += width / 2 - 0.5
                sy += height / 2 - 0.5
                outside += int(np.count_nonzero((sx < 0) | (sx > width - 1) | (sy < 0) | (sy > height - 1)))
                sx = np.clip(sx, 0, width - 1)
                sy = np.clip(sy, 0, height - 1)
                x0, y0 = np.floor(sx).astype(np.intp), np.floor(sy).astype(np.intp)
                x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
                wx, wy = (sx - x0)[..., None], (sy - y0)[..., None]
                top = source[y0, x0] * (1 - wx) + source[y0, x1] * wx
                bottom = source[y1, x0] * (1 - wx) + source[y1, x1] * wx
                stored = ((1 - wy) * top + wy * bottom).astype(np.float32)
            if not np.isfinite(stored).all():
                raise ValueError("warped samples exceed float32 storage range")
            destination[start:stop] = stored
            changed += int(np.count_nonzero(np.any(stored != block, axis=-1)))
        destination.flush()
        return {
            "method": "brown_conrady_backward_bilinear_edge_extension",
            "parameters": asdict(self),
            "status": "experimental" if active else "zero_effect",
            "changed_pixels": changed,
            "outside_source_pixels": outside,
            "order": "clean linear camera RGB before CFA and electron-domain noise",
            "limitations": (
                "Uncalibrated synthetic backward map; no lens estimation or inverse projection solve. "
                "Edge extension, no antialiasing or flux correction; may compound existing distortion. "
                "Full-scale and physical validation pending."
            ),
        }
