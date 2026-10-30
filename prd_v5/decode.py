"""Memory-gated Pillow source decoding into normalized disk-backed sRGB."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from prd_v5.inputs import InputInspector


def decode_source(
    path: str | Path,
    destination: np.memmap,
    *,
    assume_unprofiled_srgb: bool = False,
    max_decode_bytes: int = 256 * 1024 * 1024,
    strip_rows: int = 64,
) -> dict:
    """Decode 8-bit RGB JPEG/PNG, with explicit ICC conversion to sRGB.

    Pillow may hold the entire decoded source in RAM. The gate conservatively
    budgets eight bytes per source pixel for decoder storage, not peak process
    RSS. Strip buffers are additional. No target-sized RGB heap is allocated.
    Output uses stored raster orientation; EXIF orientation is not applied.
    Caller must discard partial output after any failure and keep input stable
    throughout inspection and decoding. Conversion is 8-bit relative colorimetric
    LittleCMS, without black-point compensation, not precision-preserving HDR.
    """
    for name, value in (("max_decode_bytes", max_decode_bytes), ("strip_rows", strip_rows)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    record = InputInspector().inspect(path, assume_unprofiled_srgb=assume_unprofiled_srgb)
    if record.colour_policy not in {"assumed_srgb", "requires_icc_transform"}:
        raise ValueError(f"Cannot decode colour policy: {record.colour_policy}")
    if record.width * record.height * 8 > max_decode_bytes:
        raise ValueError("Source decoder memory budget exceeded before pixel decode")
    if (
        not isinstance(destination, np.memmap)
        or destination.dtype != np.float32
        or destination.shape != (record.height, record.width, 3)
        or not destination.flags.writeable
    ):
        raise ValueError("destination must be a writable float32 HxWx3 memmap matching source")
    if Path(destination.filename).resolve() == Path(record.path):
        raise ValueError("destination must not overwrite source")
    clipped_low = clipped_high = 0
    with Image.open(record.path) as source:
        transform = None
        if record.colour_policy == "requires_icc_transform":
            transform = ImageCms.buildTransformFromOpenProfiles(
                ImageCms.ImageCmsProfile(BytesIO(source.info["icc_profile"])),
                ImageCms.createProfile("sRGB"),
                "RGB",
                "RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            )
        source.load()
        for start in range(0, record.height, strip_rows):
            end = min(record.height, start + strip_rows)
            with source.crop((0, start, record.width, end)) as strip:
                if transform is not None:
                    ImageCms.applyTransform(strip, transform, inPlace=True)
                pixels = np.asarray(strip)
                clipped_low += int(np.count_nonzero(np.any(pixels == 0, axis=2)))
                clipped_high += int(np.count_nonzero(np.any(pixels == 255, axis=2)))
                destination[start:end] = pixels.astype(np.float32) / 255.0
    destination.flush()
    return {
        "source": record.to_dict(),
        "effective_colour_policy": "icc_to_srgb" if transform is not None else "assumed_srgb",
        "icc_intent": "relative_colorimetric" if transform is not None else None,
        "icc_black_point_compensation": False,
        "output": "normalized_encoded_srgb_float32",
        "conversion_bits_per_channel": 8,
        "orientation": "stored_raster_without_exif_rotation",
        "decoder_budget_bytes": max_decode_bytes,
        "decoder_budget_estimate_bytes": record.width * record.height * 8,
        "srgb_pixels_at_zero": clipped_low,
        "srgb_pixels_at_one": clipped_high,
        "validation": "Decoded endpoints are clipping indicators, not recovered highlight truth",
    }
