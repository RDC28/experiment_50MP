"""Source diagnostics for PRD_v3 Input audit and Foundation milestone."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, UnidentifiedImageError


@dataclass(frozen=True)
class InputAudit:
    """File observations; clipping is measured in encoded RGB, before colour conversion."""

    width: int
    height: int
    mode: str
    bit_depth: int
    image_format: str
    icc_present: bool
    icc_description: str | None
    exif_present: bool
    sha256: str
    near_black_fractions: tuple[float, float, float]
    near_white_fractions: tuple[float, float, float]
    resolution_class: str
    aspect_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


class InputInspector:
    """Audit JPEG/PNG without inferring capture metadata or modifying the source."""

    @staticmethod
    def classify_resolution(width: int, height: int) -> str:
        """Label by long edge: below 2048 OTHER, 2048-4095 2K, >=4096 4K+."""
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        long_edge = max(width, height)
        return "4K+" if long_edge >= 4096 else "2K" if long_edge >= 2048 else "OTHER"

    def inspect(self, path: str | Path) -> InputAudit:
        """Decode a source and count RGB samples <=1 or >=254 (8-bit encoded units).

        RGB, RGBA, palette PNG and CMYK JPEG are RGB-capable. Alpha is ignored,
        never composited. Grayscale sources are rejected. ICC profiles are reported,
        not applied; these diagnostics do not claim scene-linear clipping.
        """
        data = Path(path).read_bytes()
        try:
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise ValueError("Input must be a JPEG or PNG image")
                if source.mode not in {"RGB", "RGBA", "P", "CMYK"}:
                    raise ValueError(f"Input must be RGB-capable; unsupported mode: {source.mode}")
                # Pillow reduces 16-bit RGB PNG to 8-bit. Reject rather than silently
                # report incorrect bit depth or calculate quantised diagnostics.
                if source.format == "PNG" and data[24] != 8:
                    if not (source.mode == "P" and data[24] in {1, 2, 4}):
                        raise ValueError("Only 8-bit RGB PNG samples are supported")
                source.load()
                rgb = np.asarray(source.convert("RGB"))
                profile = source.info.get("icc_profile")
                description = None
                if profile:
                    try:
                        description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(profile))).strip()
                    except (OSError, ValueError, TypeError):
                        # An invalid profile must remain visible as present in the audit.
                        description = None
                return InputAudit(
                    width=source.width,
                    height=source.height,
                    mode=source.mode,
                    bit_depth=8,
                    image_format=source.format,
                    icc_present=bool(profile),
                    icc_description=description,
                    exif_present=bool(source.info.get("exif")),
                    sha256=hashlib.sha256(data).hexdigest(),
                    near_black_fractions=tuple(float(v) for v in np.mean(rgb <= 1, axis=(0, 1))),
                    near_white_fractions=tuple(float(v) for v in np.mean(rgb >= 254, axis=(0, 1))),
                    resolution_class=self.classify_resolution(source.width, source.height),
                    aspect_ratio=source.width / source.height,
                )
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"Cannot decode JPEG/PNG input: {error}") from error
