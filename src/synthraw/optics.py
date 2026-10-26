"""Pre-CFA optical radiance operations in PRD_v3 optical forward-model order."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from synthraw.provenance import FieldRecord, Provenance


@dataclass(frozen=True)
class OpticalBlurParameters:
    """Optional Gaussian PSF approximations, never measured Sony lens calibration.

    Diffraction matches the Airy central-lobe FWHM (1.028 * wavelength *
    f-number); defocus approximates a uniform circle with matching variance.
    Airy rings and detailed lens aberrations are not represented.
    """

    lens_blur: bool = False
    diffraction: bool = False
    defocus: bool = False
    lens_sigma_px: float = 0.5
    wavelength_nm: float = 550.0
    f_number: float = 2.8
    pixel_pitch_um: float = 4.16
    defocus_radius_px: float = 1.0

    def sigma_px(self) -> float:
        """Combine independent isotropic PSF variances in sensor pixels."""
        for name in ("lens_blur", "diffraction", "defocus"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("lens_sigma_px", "wavelength_nm", "f_number", "pixel_pitch_um", "defocus_radius_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name in {"wavelength_nm", "f_number", "pixel_pitch_um"} and value == 0):
                raise ValueError(f"{name} is outside the physical domain")
        airy_sigma = (
            1.028 * (self.wavelength_nm / 1000) * self.f_number / self.pixel_pitch_um / math.sqrt(8 * math.log(2))
        )
        return math.sqrt(
            (self.lens_sigma_px**2 if self.lens_blur else 0)
            + (airy_sigma**2 if self.diffraction else 0)
            + ((self.defocus_radius_px / 2) ** 2 if self.defocus else 0)
        )


@dataclass(frozen=True)
class OpticalDistortionParameters:
    """Fabricated radial inverse warp in half-frame normalized coordinates.

    Destination (x, y) samples source (x, y) * (1 + k1*r² + k2*r⁴).
    Positive coefficients move source samples outward. Reflective sampling
    outside the source is a synthetic boundary assumption, not new scene data.
    """

    enabled: bool = False
    k1: float = 0.0
    k2: float = 0.0

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("k1", "k2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        # r² spans [0, 2]. Require an orientation-preserving radial map:
        # tangential scale = 1+k1*t+k2*t²; radial = 1+3*k1*t+5*k2*t².
        for a, b in ((self.k1, self.k2), (3 * self.k1, 5 * self.k2)):
            candidates = [0.0, 2.0]
            if b > 0 and 0 < -a / (2 * b) < 2:
                candidates.append(-a / (2 * b))
            if any(1 + a * t + b * t * t <= 0 for t in candidates):
                raise ValueError("radial coefficients must avoid foldovers within the frame")


@dataclass(frozen=True)
class OpticalVignettingParameters:
    """Synthetic radial falloff, specified as attenuation in stops at corners.

    Transmission is 2**(-corner_loss_ev * (r² / 2)**power), with r measured
    in half-frame coordinates. This is a hypothesis, not lens calibration.
    """

    enabled: bool = False
    corner_loss_ev: float = 1.0
    power: float = 1.0

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("corner_loss_ev", "power"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name == "power" and value == 0):
                raise ValueError(f"{name} is outside the physical domain")


class LensSimulator:
    """Strip-friendly optical stages in PRD order, before CFA sampling."""

    def blur_radiance(
        self,
        radiance: np.ndarray,
        parameters: OpticalBlurParameters | None = None,
        *,
        strip_rows: int = 128,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply blur/diffraction/defocus with exact Gaussian support halos.

        Input/output may be memmaps. Only a strip plus its vertical halo is
        filtered at once; caller-provided output avoids a full RAM allocation.
        Signed camera-space values and HDR headroom are preserved. Reflective
        image boundaries are an explicit synthetic boundary assumption.
        """
        parameters = parameters or OpticalBlurParameters()
        sigma = parameters.sigma_px()
        self._validate_radiance(radiance, strip_rows, out)
        result = np.empty_like(radiance) if out is None else out
        halo = int(4 * sigma + 0.5)  # scipy's default truncate=4 support radius
        for start in range(0, radiance.shape[0], strip_rows):
            stop = min(start + strip_rows, radiance.shape[0])
            lower, upper = max(0, start - halo), min(radiance.shape[0], stop + halo)
            source = radiance[lower:upper]
            filtered = gaussian_filter(source, (sigma, sigma, 0), mode="reflect") if sigma else source
            result[start:stop] = filtered[start - lower : stop - lower]
        return result

    @staticmethod
    def _validate_radiance(radiance: np.ndarray, strip_rows: int, out: np.ndarray | None) -> None:
        if not isinstance(radiance, np.ndarray) or radiance.ndim != 3 or radiance.shape[2] != 3:
            raise ValueError("radiance must be an H x W x 3 RGB array before CFA sampling")
        if min(radiance.shape) <= 0 or radiance.dtype.kind != "f":
            raise ValueError("radiance must be nonempty floating-point RGB")
        if type(strip_rows) is not int or strip_rows <= 0:
            raise ValueError("strip_rows must be a positive integer")
        if out is not None:
            if out.shape != radiance.shape or out.dtype != radiance.dtype or not out.flags.writeable:
                raise ValueError("out must be writable with the input shape and dtype")
            if np.may_share_memory(out, radiance):
                raise ValueError("out must not overlap the input radiance")
        # Validate before writing any output, with bounded temporary storage.
        for start in range(0, radiance.shape[0], strip_rows):
            if not np.isfinite(radiance[start : start + strip_rows]).all():
                raise ValueError("radiance must contain only finite values")

    def distort_radiance(
        self,
        radiance: np.ndarray,
        parameters: OpticalDistortionParameters | None = None,
        *,
        strip_rows: int = 128,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Bilinear radial warp after blur, before vignetting and CFA.

        Coordinates and sampled values occupy at most one strip. Input may
        be a memmap; supply an output memmap to bound resident allocation.
        No clipping, exposure rescaling, or channel mixing is performed.
        """
        parameters = parameters or OpticalDistortionParameters()
        parameters.validate()
        self._validate_radiance(radiance, strip_rows, out)
        result = np.empty_like(radiance) if out is None else out
        height, width = radiance.shape[:2]
        cx, cy = (width - 1) / 2, (height - 1) / 2
        sx, sy = max(cx, 1), max(cy, 1)
        for start in range(0, height, strip_rows):
            stop = min(start + strip_rows, height)
            if not parameters.enabled or (parameters.k1 == 0 and parameters.k2 == 0):
                result[start:stop] = radiance[start:stop]
                continue
            y, x = np.meshgrid(np.arange(start, stop), np.arange(width), indexing="ij")
            dx, dy = (x - cx) / sx, (y - cy) / sy
            radius2 = dx * dx + dy * dy
            scale = 1 + parameters.k1 * radius2 + parameters.k2 * radius2**2
            coordinates = [cy + dy * scale * sy, cx + dx * scale * sx]
            for channel in range(3):
                result[start:stop, :, channel] = map_coordinates(
                    radiance[:, :, channel], coordinates, order=1, mode="reflect", prefilter=False
                )
        return result

    def distortion_records(self, parameters: OpticalDistortionParameters | None = None) -> dict[str, FieldRecord]:
        """Record the enabled warp and its unmeasured coefficients."""
        parameters = parameters or OpticalDistortionParameters()
        parameters.validate()
        return {
            name: FieldRecord(name, value, Provenance.FABRICATED, "Synthetic radial inverse warp; not lens calibration")
            for name, value in vars(parameters).items()
        }

    def blur_records(self, parameters: OpticalBlurParameters | None = None) -> dict[str, FieldRecord]:
        """Describe fabricated PSF hypotheses separately from correction flags."""
        parameters = parameters or OpticalBlurParameters()
        sigma = parameters.sigma_px()
        values = dict(vars(parameters), effective_sigma_px=sigma)
        return {
            name: FieldRecord(name, value, Provenance.FABRICATED, "Synthetic Gaussian optical PSF approximation")
            for name, value in values.items()
        }

    def vignette_radiance(
        self,
        radiance: np.ndarray,
        parameters: OpticalVignettingParameters | None = None,
        *,
        strip_rows: int = 128,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Attenuate radiance after distortion and before chromatic effects/CFA.

        A scalar transmission preserves channel ratios, signed values and HDR
        headroom. Coordinates and transmission occupy at most one strip.
        Supply an output memmap to avoid a full-frame resident allocation.
        """
        parameters = parameters or OpticalVignettingParameters()
        parameters.validate()
        self._validate_radiance(radiance, strip_rows, out)
        result = np.empty_like(radiance) if out is None else out
        height, width = radiance.shape[:2]
        cx, cy = (width - 1) / 2, (height - 1) / 2
        x2 = ((np.arange(width) - cx) / max(cx, 1)) ** 2
        for start in range(0, height, strip_rows):
            stop = min(start + strip_rows, height)
            if not parameters.enabled or parameters.corner_loss_ev == 0:
                result[start:stop] = radiance[start:stop]
                continue
            y2 = ((np.arange(start, stop) - cy) / max(cy, 1)) ** 2
            radius2 = (y2[:, None] + x2[None, :]) / 2
            transmission = np.exp2(-parameters.corner_loss_ev * radius2**parameters.power)
            result[start:stop] = radiance[start:stop] * transmission[:, :, None]
        return result

    def vignetting_records(self, parameters: OpticalVignettingParameters | None = None) -> dict[str, FieldRecord]:
        """Record enabled attenuation and its fabricated radial hypothesis."""
        parameters = parameters or OpticalVignettingParameters()
        parameters.validate()
        return {
            name: FieldRecord(name, value, Provenance.FABRICATED, "Synthetic radial vignetting; not lens calibration")
            for name, value in vars(parameters).items()
        }
