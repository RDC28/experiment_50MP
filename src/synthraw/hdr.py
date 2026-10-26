"""Scene-linear reconstruction primitives for PRD_v3 inverse rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    ) -> np.memmap:
        """Write float32 .npy radiance without copying the entire input to RAM.

        Accept uint8 or normalized floating RGB, including read-only memmaps.
        The caller selects the transfer from source diagnostics; ICC resolution
        is not performed here. Reject input/output aliasing to protect source data.
        """
        if encoded.ndim != 3 or encoded.shape[2] != 3 or min(encoded.shape[:2]) == 0:
            raise ValueError("Expected a nonempty H x W x 3 RGB array")
        if encoded.dtype != np.uint8 and not np.issubdtype(encoded.dtype, np.floating):
            raise ValueError("Expected uint8 or normalized floating RGB")
        if type(strip_rows) is not int or strip_rows <= 0:
            raise ValueError("strip_rows must be a positive integer")
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
                output[row : row + strip_rows] = inverse_smoothstep(block) if undo_tone else block
            output.flush()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return output
