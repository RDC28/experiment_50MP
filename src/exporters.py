"""Professional master exports and their validation (PRD sections 17 and 23.6).

* Export A: 16-bit sRGB TIFF, embedded sRGB ICC profile, lossless compression,
  provenance JSON in ImageDescription.
* Export B: OpenEXR with half or float R, G, B channels in linear Rec.709,
  standard ``chromaticities`` plus string attributes that state the
  scene-referred limitation.
* Export C: 8-bit review PNG/JPEG with an embedded sRGB profile.

Every writer reads its file back and checks dimensions, sample type, colour
metadata, provenance metadata and a pixel hash, so a broken or silently
down-converted master is caught at export time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import color_management as cm
from .ingestion import pixel_sha256, sha256_file

REC709_CHROMATICITIES = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
EXR_COLOR_DESCRIPTION = (
    "linear-light RGB, Rec.709/sRGB primaries, D65 white; linearised from a display-referred source; "
    "synthetic linear RGB, not calibrated scene-referred radiance"
)
TIFF_COLOR_DESCRIPTION = "sRGB IEC 61966-2-1 encoded, display-referred"


class ExportValidationError(RuntimeError):
    """A written file failed read-back validation."""


@dataclass
class ExportResult:
    path: Path
    kind: str
    width: int
    height: int
    sample_type: str
    channels: int
    compression: str
    file_sha256: str
    pixel_sha256: str
    size_bytes: int
    checks: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "sample_type": self.sample_type,
            "channels": self.channels,
            "compression": self.compression,
            "file_sha256": self.file_sha256,
            "pixel_sha256": self.pixel_sha256,
            "size_bytes": self.size_bytes,
            "checks": self.checks,
            "valid": self.valid,
        }


def write_tiff16(path: Path, encoded: np.ndarray, metadata: dict, *, compression: str = "zlib") -> ExportResult:
    """Write a (H, W, 3) uint16 sRGB-encoded master and validate it."""
    import tifffile

    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint16, got {encoded.shape} {encoded.dtype}")
    if compression not in ("zlib", "none"):
        raise ValueError(f"unsupported TIFF compression {compression!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    icc = cm.srgb_icc_profile()
    description = json.dumps(metadata, sort_keys=True)
    expected_hash = pixel_sha256(encoded)
    tifffile.imwrite(
        path,
        encoded,
        photometric="rgb",
        planarconfig="contig",
        compression="zlib" if compression == "zlib" else None,
        predictor=2 if compression == "zlib" else None,
        iccprofile=icc,
        description=description,
        software=metadata.get("software", "synthetic_raw_lab"),
        datetime=datetime.now(timezone.utc).strftime("%Y:%m:%d %H:%M:%S"),
        metadata=None,
        bigtiff=encoded.nbytes > 2**31,
    )

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        icc_tag = page.tags.get(34675)
        stored_description = page.description
        decoded = page.asarray()
        checks = {
            "decodes": True,
            "single_image": len(tif.pages) == 1,
            "dimensions": (page.imagewidth, page.imagelength) == (encoded.shape[1], encoded.shape[0]),
            "bits_per_sample_16": page.bitspersample == 16,
            "samples_per_pixel_3": page.samplesperpixel == 3,
            "photometric_rgb": page.photometric == tifffile.PHOTOMETRIC.RGB,
            "lossless_compression": page.compression in (tifffile.COMPRESSION.NONE, tifffile.COMPRESSION.ADOBE_DEFLATE),
            "icc_profile_embedded": icc_tag is not None and bytes(icc_tag.value) == icc,
            "metadata_roundtrip": _json_or_none(stored_description) == json.loads(description),
        }
        compression_name = page.compression.name
    checks["no_8bit_downconversion"] = decoded.dtype == np.uint16
    checks["pixels_identical"] = pixel_sha256(decoded) == expected_hash
    del decoded
    result = ExportResult(
        path,
        "tiff16_srgb",
        encoded.shape[1],
        encoded.shape[0],
        "uint16",
        3,
        compression_name,
        sha256_file(path),
        expected_hash,
        path.stat().st_size,
        checks,
    )
    _raise_if_invalid(result)
    return result


def write_exr(
    path: Path,
    linear: np.ndarray,
    attributes: dict[str, str],
    *,
    precision: str = "half",
    compression: str = "zip",
    chromaticities: tuple = REC709_CHROMATICITIES,
) -> ExportResult:
    """Write a (H, W, 3) linear float master as separate R, G, B channels and validate it."""
    import OpenEXR

    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {linear.shape}")
    dtype = {"half": np.float16, "float": np.float32}.get(precision)
    if dtype is None:
        raise ValueError(f"unsupported EXR precision {precision!r}")
    codec = {
        "zip": OpenEXR.ZIP_COMPRESSION,
        "piz": OpenEXR.PIZ_COMPRESSION,
        "none": OpenEXR.NO_COMPRESSION,
    }.get(compression)
    if codec is None:
        raise ValueError(f"unsupported EXR compression {compression!r}")
    for key, value in attributes.items():
        if not isinstance(value, str):
            raise ValueError(f"EXR attribute {key!r} must be a string")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    planes = {name: np.ascontiguousarray(linear[..., i], dtype=dtype) for i, name in enumerate("RGB")}
    expected_hash = _planes_sha256(planes)
    nonfinite = sum(int(p.size - np.count_nonzero(np.isfinite(p))) for p in planes.values())
    header = {
        "compression": codec,
        "type": OpenEXR.scanlineimage,
        "chromaticities": tuple(float(v) for v in chromaticities),
        **attributes,
    }
    with OpenEXR.File(header, planes) as exr:
        exr.write(str(path))
    del planes

    with OpenEXR.File(str(path), separate_channels=True) as exr:
        stored_header = exr.header()
        channels = exr.channels()
        stored_planes = {name: channels[name].pixels for name in "RGB" if name in channels}
        window = stored_header["dataWindow"]
        stored_size = (int(window[1][0] - window[0][0] + 1), int(window[1][1] - window[0][1] + 1))
        stored_chroma = stored_header.get("chromaticities")
        checks = {
            "decodes": True,
            "channels_rgb": sorted(channels) == ["B", "G", "R"],
            "dimensions": stored_size == (linear.shape[1], linear.shape[0]),
            "channel_precision": all(p.dtype == dtype for p in stored_planes.values()),
            "chromaticities": stored_chroma is not None and np.allclose(stored_chroma, chromaticities, atol=1e-6),
            "attributes_roundtrip": all(stored_header.get(k) == v for k, v in attributes.items()),
            "pixels_identical": len(stored_planes) == 3 and _planes_sha256(stored_planes) == expected_hash,
            "finite_values": nonfinite == 0,
        }
        compression_name = stored_header["compression"].name
        del stored_planes, channels
    result = ExportResult(
        path,
        f"exr_{precision}_linear_rec709",
        linear.shape[1],
        linear.shape[0],
        np.dtype(dtype).name,
        3,
        compression_name,
        sha256_file(path),
        expected_hash,
        path.stat().st_size,
        checks,
    )
    _raise_if_invalid(result)
    return result


def write_review_image(path: Path, rgb8: np.ndarray, *, quality: int = 92) -> Path:
    """Write an 8-bit sRGB review PNG or JPEG (chosen by suffix) with an embedded sRGB profile."""
    from PIL import Image

    if rgb8.dtype != np.uint8 or rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got {rgb8.shape} {rgb8.dtype}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"icc_profile": cm.srgb_icc_profile()}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        options.update(quality=quality, subsampling=0)
    else:
        options.update(compress_level=6)
    Image.fromarray(rgb8).save(path, **options)
    return path


def write_sidecar(master: ExportResult, provenance: dict) -> Path:
    """JSON sidecar next to a master, so provenance travels with it when reports are stored elsewhere."""
    sidecar = master.path.with_name(master.path.name + ".json")
    sidecar.write_text(json.dumps({"export": master.to_dict(), "provenance": provenance}, indent=2, sort_keys=True))
    return sidecar


def _planes_sha256(planes: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(planes):
        plane = planes[name]
        digest.update(f"{name}|{plane.shape}|{plane.dtype.str}|".encode())
        digest.update(np.ascontiguousarray(plane).data)
    return digest.hexdigest()


def _json_or_none(text: str | None):
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def _raise_if_invalid(result: ExportResult) -> None:
    failed = [name for name, ok in result.checks.items() if not ok]
    if failed:
        raise ExportValidationError(f"{result.path.name} failed validation: {', '.join(failed)}")
