"""Bayer photosite sampling for PRD_v3's sensor milestone."""

from __future__ import annotations

import numpy as np

from synthraw.dng_metadata import CFA_PATTERNS


class CFAMosaicer:
    """Select one RGB channel per photosite using the metadata CFA convention."""

    def __init__(self, phase: str = "RGGB") -> None:
        if not isinstance(phase, str) or phase not in CFA_PATTERNS:
            raise ValueError("phase must be RGGB, BGGR, GRBG, or GBRG")
        self.phase = phase

    def mosaic(
        self,
        rgb: np.ndarray,
        *,
        strip_rows: int = 128,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample clean or noisy RGB without clipping, scaling, or mixing channels.

        Preserve the input dtype and signed noise excursions for the later ADC.
        An independent output memmap permits bounded-memory frame processing.
        Validate all strips before writing any output, including caller buffers.
        """
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape) == 0
            or rgb.dtype.kind != "f"
        ):
            raise ValueError("rgb must be a nonempty floating RGB array")
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be a positive integer")
        if out is not None and (
            not isinstance(out, np.ndarray)
            or out.shape != rgb.shape[:2]
            or out.dtype != rgb.dtype
            or not out.flags.writeable
            or np.may_share_memory(rgb, out)
        ):
            raise ValueError("out must be writable, independent, and match the RGB spatial shape and dtype")
        for start in range(0, rgb.shape[0], strip_rows):
            if not np.isfinite(rgb[start : start + strip_rows]).all():
                raise ValueError("rgb must contain only finite values")
        result = np.empty(rgb.shape[:2], dtype=rgb.dtype) if out is None else out
        pattern = CFA_PATTERNS[self.phase]
        for start in range(0, rgb.shape[0], strip_rows):
            stop = min(start + strip_rows, rgb.shape[0])
            for row in range(2):
                first = start + (row - start) % 2
                for column in range(2):
                    result[first:stop:2, column::2] = rgb[first:stop:2, column::2, pattern[2 * row + column]]
        return result

    def split_planes(self, mosaic: np.ndarray) -> dict[str, np.ndarray]:
        """Return R/G1/G2/B strided views, with greens ordered in raster order.

        Odd spatial dimensions are supported; planes may have unequal shapes.
        Views avoid another frame allocation and share the mosaic's writability.
        """
        if not isinstance(mosaic, np.ndarray) or mosaic.ndim != 2 or min(mosaic.shape) == 0:
            raise ValueError("mosaic must be a nonempty two-dimensional array")
        planes = {}
        green = 0
        for index, channel in enumerate(CFA_PATTERNS[self.phase]):
            if channel == 1:
                green += 1
                name = f"G{green}"
            else:
                name = "R" if channel == 0 else "B"
            row, column = divmod(index, 2)
            planes[name] = mosaic[row::2, column::2]
        return planes
