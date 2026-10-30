"""Bounded source hashing and header inspection, without pixel reconstruction."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageCms, UnidentifiedImageError


@dataclass(frozen=True)
class SourceRecord:
    """Observed source header, with no inferred capture identity or pixel truth."""

    path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    image_format: str
    mode: str
    encoded_bits_per_channel: int
    icc_sha256: str | None
    icc_description: str | None
    colour_policy: str
    exif_present: bool
    validation: str = "Header inspected only; pixel decode and clipping diagnostics pending"

    def to_dict(self) -> dict:
        return asdict(self)


class InputInspector:
    """Inspect JPEG/PNG headers and stream file bytes in bounded chunks.

    ICC descriptions are labels, not proof of a colour space. Embedded profiles
    require a future explicit colour transform, including profiles named sRGB.
    Alpha is rejected until a compositing policy exists. No EXIF identity values
    are copied to the registry. Unprofiled RGB requires an explicit sRGB assumption.
    """

    def inspect(self, path: str | Path, *, assume_unprofiled_srgb: bool = False) -> SourceRecord:
        if type(assume_unprofiled_srgb) is not bool:
            raise ValueError("assume_unprofiled_srgb must be boolean")
        path = Path(path).expanduser().resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            stream.seek(0)
            prefix = stream.read(26)
            stream.seek(0)
            try:
                with Image.open(stream) as source:
                    if source.format not in {"JPEG", "PNG"}:
                        raise ValueError("Only JPEG and PNG sources are supported")
                    if source.mode != "RGB":
                        raise ValueError("Source must be RGB without alpha; convert explicitly before use")
                    bits = prefix[24] if source.format == "PNG" else 8
                    if bits != 8:
                        raise ValueError("16-bit PNG requires a precision-preserving decoder; not supported yet")
                    profile = source.info.get("icc_profile")
                    description = None
                    if profile:
                        try:
                            description = ImageCms.getProfileDescription(
                                ImageCms.ImageCmsProfile(BytesIO(profile))
                            ).strip()
                            policy = "requires_icc_transform"
                        except (OSError, ValueError, TypeError):
                            policy = "blocked_invalid_icc"
                    else:
                        policy = "assumed_srgb" if assume_unprofiled_srgb else "requires_colour_assumption"
                    return SourceRecord(
                        path=str(path),
                        sha256=digest.hexdigest(),
                        size_bytes=size,
                        width=source.width,
                        height=source.height,
                        image_format=source.format,
                        mode=source.mode,
                        encoded_bits_per_channel=bits,
                        icc_sha256=hashlib.sha256(profile).hexdigest() if profile else None,
                        icc_description=description,
                        colour_policy=policy,
                        exif_present=bool(source.info.get("exif")),
                    )
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                raise ValueError(f"Cannot inspect source header: {error}") from error
