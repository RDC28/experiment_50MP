"""Pre-CFA optical radiance operations in PRD_v3 optical forward-model order."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

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


class LensSimulator:
    """Strip-friendly first optical stage; call before distortion and CFA sampling."""

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
        result = np.empty_like(radiance) if out is None else out
        halo = int(4 * sigma + 0.5)  # scipy's default truncate=4 support radius
        for start in range(0, radiance.shape[0], strip_rows):
            stop = min(start + strip_rows, radiance.shape[0])
            lower, upper = max(0, start - halo), min(radiance.shape[0], stop + halo)
            source = radiance[lower:upper]
            filtered = gaussian_filter(source, (sigma, sigma, 0), mode="reflect") if sigma else source
            result[start:stop] = filtered[start - lower : stop - lower]
        return result

    def blur_records(self, parameters: OpticalBlurParameters | None = None) -> dict[str, FieldRecord]:
        """Describe fabricated PSF hypotheses separately from correction flags."""
        parameters = parameters or OpticalBlurParameters()
        sigma = parameters.sigma_px()
        values = dict(vars(parameters), effective_sigma_px=sigma)
        return {
            name: FieldRecord(name, value, Provenance.FABRICATED, "Synthetic Gaussian optical PSF approximation")
            for name, value in values.items()
        }
