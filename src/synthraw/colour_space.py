"""ICC-description assumptions and D65 working primaries (PRD_v3 input audit)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from synthraw.hdr import Transfer
from synthraw.input_audit import InputAudit
from synthraw.provenance import FieldRecord, Provenance

# Standard D65 RGB-to-XYZ matrices, with rows X/Y/Z and columns R/G/B.
_RGB_TO_XYZ = {
    "SRGB": np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    ),
    "DISPLAY_P3": np.array(
        [[0.48657095, 0.26566769, 0.19821729], [0.22897456, 0.69173852, 0.07928691], [0.0, 0.04511338, 1.04394437]]
    ),
    "ADOBE_RGB": np.array(
        [[0.5767309, 0.1855540, 0.1881852], [0.2973769, 0.6273491, 0.0752741], [0.0270343, 0.0706872, 0.9911085]]
    ),
}


@dataclass(frozen=True)
class ColourSpaceResolution:
    """Description-based interpretation, never a full arbitrary ICC transform."""

    transfer: Transfer
    matrix: tuple[tuple[float, ...], ...]
    assumption: str | None
    icc_description: str | None

    def records(self) -> list[FieldRecord]:
        """Record both the inferred interpretation and deterministic matrix."""
        return [
            FieldRecord(
                field="input_colour_space",
                value={"transfer": self.transfer, "icc_description": self.icc_description},
                provenance=Provenance.INFERRED,
                method="Interpret input audit ICC description as standard D65 RGB space",
                disclaimer=self.assumption or "Description matching does not evaluate ICC LUTs or custom curves",
            ),
            FieldRecord(
                field="working_colour_space",
                value={"primaries": "linear Rec.709/sRGB", "white_point": "D65", "matrix": self.matrix},
                provenance=Provenance.DETERMINISTIC,
                method="inverse(sRGB to XYZ D65) @ source RGB to XYZ D65; no gamut clipping",
                dependencies=["input_colour_space"],
            ),
        ]


def resolve_colour_space(audit: InputAudit) -> ColourSpaceResolution:
    """Resolve supported descriptions; missing or unknown profiles assume sRGB."""
    description = audit.icc_description if audit.icc_present else None
    name = re.sub(r"[^a-z0-9]", "", (description or "").lower())
    transfer: Transfer = "SRGB"
    assumption = None
    if "displayp3" in name:
        transfer = "DISPLAY_P3"
    elif "adobergb" in name:
        transfer = "ADOBE_RGB"
    elif "rec709" in name or "bt709" in name:
        transfer = "REC709"
    elif "srgb" not in name:
        assumption = "Missing, unreadable or unsupported ICC description; assumed sRGB transfer and primaries"
    matrix = (
        np.eye(3) if transfer in ("SRGB", "REC709") else np.linalg.solve(_RGB_TO_XYZ["SRGB"], _RGB_TO_XYZ[transfer])
    )
    return ColourSpaceResolution(
        transfer, tuple(tuple(float(v) for v in row) for row in matrix), assumption, description
    )


def convert_linear_primaries(linear: np.ndarray, resolution: ColourSpaceResolution) -> np.ndarray:
    """Convert linear RGB to working primaries, retaining negative/out-of-gamut values."""
    values = np.asarray(linear, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite linear RGB with final dimension 3")
    return (values @ np.asarray(resolution.matrix, dtype=np.float32).T).astype(np.float32)


def rgb_to_xyz_matrix(transfer: Transfer) -> np.ndarray:
    """D65 RGB-to-XYZ matrix for a named space; Rec.709 shares the sRGB primaries."""
    return _RGB_TO_XYZ["SRGB" if transfer == "REC709" else transfer].copy()
