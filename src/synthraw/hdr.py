"""Scene-linear reconstruction primitives for PRD_v3 inverse rendering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from synthraw.colour_space import ColourSpaceResolution

import numpy as np

Transfer = Literal["SRGB", "DISPLAY_P3", "ADOBE_RGB", "REC709"]


def inverse_transfer(encoded: np.ndarray, transfer: Transfer = "SRGB") -> np.ndarray:
    """Invert the named RGB transfer curve, retaining its native colour primaries.

    Inputs are normalized encoded values. Display P3 uses the sRGB curve;
    Adobe RGB uses its specified 563/256 exponent. This performs no gamut conversion.
    """
    values = np.asarray(encoded, dtype=np.float32)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Encoded RGB must be finite and normalized to [0, 1]")
    if transfer in ("SRGB", "DISPLAY_P3"):
        result = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    elif transfer == "ADOBE_RGB":
        result = values ** (563 / 256)
    elif transfer == "REC709":
        result = np.where(values < 0.081, values / 4.5, ((values + 0.099) / 1.099) ** (1 / 0.45))
    else:
        raise ValueError(f"Unsupported transfer: {transfer}")
    return result.astype(np.float32)


def _endpoint_weight(distance: np.ndarray) -> np.ndarray:
    """Smoothstep weight over the 5/255 endpoint band; ``distance`` is positive inside."""
    weight = np.clip(distance / (5 / 255), 0, 1)
    return weight * weight * (3 - 2 * weight)


def inverse_smoothstep(linear: np.ndarray) -> np.ndarray:
    """Undo Brooks-style smoothstep tone rendering on normalized linear RGB.

    The inverse is analytic for y = 3*x**2 - 2*x**3 on [0, 1]. It is an
    inferred rendering assumption, not recovery of the source camera's tone curve.
    """
    values = np.asarray(linear, dtype=np.float32)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Tone-rendered RGB must be finite and normalized to [0, 1]")
    return (0.5 - np.sin(np.arcsin(1 - 2 * values) / 3)).astype(np.float32)


class HDRReconstructionEngine:
    """Reconstruct scene-linear RGB in bounded strips (PRD_v3 HDR processing).

    This stage inverts transfer and optional tone rendering only. Clipped-region
    HDR expansion and its confidence masks are a separate subsequent operation.
    """

    def linearize(
        self,
        encoded: np.ndarray,
        output_path: str | Path,
        *,
        transfer: Transfer = "SRGB",
        undo_tone: bool = True,
        strip_rows: int = 64,
        colour_space: ColourSpaceResolution | None = None,
    ) -> np.memmap:
        """Write float32 .npy radiance without copying the entire input to RAM.

        Accept uint8 or normalized floating RGB, including read-only memmaps.
        The caller selects the transfer from source diagnostics; ICC resolution
        is supplied through colour_space when resolved from an input audit. That
        resolution overrides transfer and converts primaries after tone inversion.
        Reject input/output aliasing to protect source data.
        """
        if encoded.ndim != 3 or encoded.shape[2] != 3 or min(encoded.shape[:2]) == 0:
            raise ValueError("Expected a nonempty H x W x 3 RGB array")
        if encoded.dtype != np.uint8 and not np.issubdtype(encoded.dtype, np.floating):
            raise ValueError("Expected uint8 or normalized floating RGB")
        if type(strip_rows) is not int or strip_rows <= 0:
            raise ValueError("strip_rows must be a positive integer")
        if colour_space is not None:
            transfer = colour_space.transfer
        if transfer not in ("SRGB", "DISPLAY_P3", "ADOBE_RGB", "REC709"):
            raise ValueError(f"Unsupported transfer: {transfer}")
        path = Path(output_path)
        source = encoded
        while source is not None:
            filename = getattr(source, "filename", None)
            if filename is not None and Path(filename).resolve() == path.resolve():
                raise ValueError("Output must not overwrite the input memmap")
            source = getattr(source, "base", None)
        # Exclusive creation prevents accidental truncation of an existing stage.
        with path.open("xb"):
            pass
        try:
            output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=encoded.shape)
            for row in range(0, encoded.shape[0], strip_rows):
                block = encoded[row : row + strip_rows].astype(np.float32)
                if encoded.dtype == np.uint8:
                    block /= 255
                block = inverse_transfer(block, transfer)
                block = inverse_smoothstep(block) if undo_tone else block
                if colour_space is not None:
                    from synthraw.colour_space import convert_linear_primaries

                    block = convert_linear_primaries(block, colour_space)
                output[row : row + strip_rows] = block
            output.flush()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return output

    def expand(
        self,
        encoded: np.ndarray,
        linear: np.ndarray,
        output_dir: str | Path,
        *,
        highlight_ev: float = 3.0,
        shadow_ev: float = 3.0,
        backend: str = "DETERMINISTIC",
        strip_rows: int = 64,
        negative_policy: Literal["REJECT", "CLIP"] = "REJECT",
    ) -> dict:
        """Write deterministic HDR and four masks (PRD_v3 HDR fallback).

        Classification uses encoded RGB: any channel above 250/255 or below
        5/255 is uncertain. A smoothstep weight reaches one at the endpoints.
        Gain uses the all-channel weight (minimum above 250/255, maximum below
        5/255), so one saturated or empty channel never rescales a well exposed
        pixel. All channels are scaled together to retain chromaticity. Endpoint pixels
        have zero confidence: their tonal extrapolation is fabricated, not detail
        recovery. EV limits bound the gain relative to source-derived radiance.
        Native colour primaries are retained, as in ``linearize``. For wide-gamut
        working-space conversion, explicit CLIP maps negative channels to zero
        without limiting positive headroom. The returned audit records this lossy
        assumption; affected pixels have zero confidence and no source support.
        The signed scene-linear input remains unchanged.
        """
        if encoded.shape != linear.shape or encoded.ndim != 3 or encoded.shape[2] != 3:
            raise ValueError("Expected matching H x W x 3 arrays")
        if min(encoded.shape[:2]) == 0:
            raise ValueError("Expected nonempty arrays")
        if encoded.dtype != np.uint8 and not np.issubdtype(encoded.dtype, np.floating):
            raise ValueError("Expected uint8 or normalized floating RGB")
        if type(strip_rows) is not int or strip_rows <= 0:
            raise ValueError("strip_rows must be a positive integer")
        if not all(np.isfinite(ev) and 0 <= ev <= 16 for ev in (highlight_ev, shadow_ev)):
            raise ValueError("EV limits must be finite and in [0, 16]")
        if backend not in ("DETERMINISTIC", "EXPANDDIFF", "VID2HDR"):
            raise ValueError(f"Unsupported HDR backend: {backend}")
        if negative_policy not in ("REJECT", "CLIP"):
            raise ValueError(f"Unsupported negative_policy: {negative_policy}")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        created = []
        names = (
            "04_hdr_radiance",
            "source_supported_mask",
            "highlight_fabricated_mask",
            "shadow_fabricated_mask",
            "hdr_confidence",
        )
        generated = 0
        negative_channels = 0
        negative_pixels = 0
        minimum_channel = float("inf")
        try:
            for name in names:
                path = directory / f"{name}.npy"
                with path.open("xb"):
                    pass
                created.append(path)
                arrays[name] = np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=np.float32 if name in (names[0], names[-1]) else np.bool_,
                    shape=linear.shape if name == names[0] else linear.shape[:2],
                )
            for row in range(0, linear.shape[0], strip_rows):
                region = slice(row, row + strip_rows)
                rgb = encoded[region].astype(np.float32)
                if encoded.dtype == np.uint8:
                    rgb /= 255
                radiance = np.asarray(linear[region], dtype=np.float32)
                if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
                    raise ValueError("Encoded RGB must be finite and normalized to [0, 1]")
                if not np.isfinite(radiance).all() or (negative_policy == "REJECT" and np.any(radiance < 0)):
                    raise ValueError("Linear radiance must be finite and nonnegative")
                negative = radiance < 0
                mapped = np.any(negative, axis=2)
                negative_channels += int(np.count_nonzero(negative))
                negative_pixels += int(np.count_nonzero(mapped))
                minimum_channel = min(minimum_channel, float(radiance.min()))
                if negative_policy == "CLIP":
                    radiance = np.maximum(radiance, 0)
                # Any endpoint channel makes a pixel uncertain; only all-channel endpoints get gain.
                high, high_gain = (
                    _endpoint_weight(rgb.max(axis=2) - 250 / 255),
                    _endpoint_weight(rgb.min(axis=2) - 250 / 255),
                )
                low, low_gain = (
                    _endpoint_weight(5 / 255 - rgb.min(axis=2)),
                    _endpoint_weight(5 / 255 - rgb.max(axis=2)),
                )
                gain = np.exp2(high_gain * highlight_ev - low_gain * shadow_ev)
                with np.errstate(over="ignore"):
                    expanded = radiance * gain[..., None]
                if not np.isfinite(expanded).all():
                    raise ValueError("HDR expansion exceeds float32 range")
                highlight = high > 0
                shadow = low > 0
                arrays[names[0]][region] = expanded
                arrays[names[1]][region] = ~(highlight | shadow | mapped)
                arrays[names[2]][region] = highlight
                arrays[names[3]][region] = shadow
                arrays[names[4]][region] = np.where(mapped, 0, 1 - np.maximum(high, low))
                generated += np.count_nonzero(highlight | shadow)
            for array in arrays.values():
                array.flush()
        except Exception:
            arrays.clear()
            for path in created:
                path.unlink(missing_ok=True)
            raise
        return {
            "arrays": arrays,
            "backend": "DETERMINISTIC",
            "fallback_reason": None if backend == "DETERMINISTIC" else f"{backend} is a gated, unavailable backend",
            "generated_hdr_area_percentage": 100 * generated / (linear.shape[0] * linear.shape[1]),
            "gamut_mapping": {
                "policy": negative_policy,
                "negative_channel_count": negative_channels,
                "affected_pixel_count": negative_pixels,
                "affected_pixel_percentage": 100 * negative_pixels / (linear.shape[0] * linear.shape[1]),
                "minimum_input_channel": minimum_channel,
                "provenance": "I",
                "method": "Clip negative working-space channels to zero; retain positive headroom"
                if negative_policy == "CLIP"
                else "Reject negative radiance",
                "disclaimer": "Negative-channel clipping is a lossy gamut assumption, not recovery of source colour",
            },
            "provenance": {"source_supported": "I", "extrapolated": "F", "destroyed": "GENERATED"},
        }
