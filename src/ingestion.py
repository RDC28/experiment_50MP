"""Input loading and validation (PRD sections 3.3, 3.4 and 7.1).

``ingest`` copies the source into a read-only project input store, decodes the
stored copy without channel swapping or implicit tone changes, validates it and
converts it to the linear working space. The original file is never modified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import color_management as cm

FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WebP",
    ".avif": "AVIF",
    ".exr": "OpenEXR",
}
MANDATORY_FORMATS = ("PNG", "JPEG", "TIFF")

CLIPPING_WARNING_FRACTION = 0.005
BLOCKINESS_WARNING_RATIO = 1.15

_EXIF_TAGS = {271: "Make", 272: "Model", 274: "Orientation", 305: "Software", 306: "DateTime"}
_EXIF_COLOR_SPACE = 0xA001
_IJG_LUMINANCE_TABLE_SUM = 3688  # sum of the IJG standard luminance quantisation table
# Pillow ``info`` keys that describe encoding rather than embedded text metadata.
_STRUCTURAL_INFO_KEYS = {
    "icc_profile",
    "exif",
    "dpi",
    "gamma",
    "srgb",
    "transparency",
    "jfif",
    "jfif_version",
    "jfif_unit",
    "jfif_density",
    "progressive",
    "progression",
    "adobe",
    "adobe_transform",
    "chromaticity",
    "loop",
    "duration",
    "background",
    "xmp",
}


class IngestError(ValueError):
    """The source cannot be used: unsupported, corrupt, or not interpretable."""


@dataclass
class SourceImage:
    original_path: Path
    stored_path: Path
    sha256: str
    pixel_sha256: str
    codes: np.ndarray  # decoded code values, RGB order, alpha removed, (H, W, 1|3)
    linear: np.ndarray  # working-space linear RGB float32 (H, W, 3)
    color_space: cm.SourceColorSpace
    validation: dict
    warnings: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.linear.shape[1]

    @property
    def height(self) -> int:
        return self.linear.shape[0]

    def normalized(self) -> np.ndarray:
        """Code values scaled to [0, 1] float32 in the source encoding."""
        return _normalize(self.codes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    """Hash of decoded pixel data including its shape and dtype."""
    digest = hashlib.sha256(f"{array.shape}|{array.dtype.str}|".encode())
    digest.update(np.ascontiguousarray(array).data)
    return digest.hexdigest()


def ingest(
    source: str | Path,
    inputs_root: str | Path,
    category: str,
    *,
    source_color_space: str = "auto",
) -> SourceImage:
    source = Path(source)
    if not source.is_file():
        raise IngestError(f"source file not found: {source}")
    image_format = FORMATS.get(source.suffix.lower())
    if image_format is None:
        raise IngestError(f"unsupported source format {source.suffix!r}; supported: {sorted(set(FORMATS.values()))}")
    if source.stat().st_size == 0:
        raise IngestError(f"source file is empty: {source}")

    file_hash = sha256_file(source)
    store_dir = Path(inputs_root) / category
    stored = _store_read_only(source, store_dir, file_hash)
    decoded = _decode(stored, image_format)

    warnings: list[str] = []
    codes = decoded.pixels
    if codes.ndim == 2:
        codes = codes[:, :, None]
    height, width, channels = codes.shape
    if width == 0 or height == 0:
        raise IngestError("source has zero width or height")
    if channels not in (1, 2, 3, 4):
        raise IngestError(f"unsupported channel count {channels}")

    alpha = None
    if channels in (2, 4):
        alpha, codes = codes[..., -1], np.ascontiguousarray(codes[..., :-1])
    if np.issubdtype(codes.dtype, np.floating):
        nonfinite = int(codes.size - np.count_nonzero(np.isfinite(codes)))
        if nonfinite:
            raise IngestError(f"source contains {nonfinite} NaN or infinite values")
    elif codes.dtype not in (np.uint8, np.uint16):
        raise IngestError(f"unsupported sample type {codes.dtype}")

    try:
        color_space = cm.resolve_source_color_space(
            source_color_space, decoded.icc_profile, is_linear_container=image_format == "OpenEXR"
        )
    except cm.UnsupportedColorProfile as error:
        raise IngestError(str(error)) from error
    if decoded.exr_chromaticities is not None and source_color_space == "auto":
        color_space = cm.linear_color_space_from_chromaticities(decoded.exr_chromaticities)

    normalized = _normalize(codes)
    linear = cm.to_working_space(normalized, color_space)
    pixels_hash = pixel_sha256(codes)

    validation = {
        "file": {
            "original_name": source.name,
            "format": image_format,
            "mandatory_format": image_format in MANDATORY_FORMATS,
            "size_bytes": source.stat().st_size,
            "sha256": file_hash,
            "stored_copy": str(stored),
            "stored_read_only": not (stored.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)),
            "decoder": decoded.decoder,
        },
        "dimensions": {
            "width": width,
            "height": height,
            "megapixels": width * height / 1e6,
            "aspect_ratio": width / height,
            "aspect_ratio_reduced": _reduced_ratio(width, height),
        },
        "samples": {
            "dtype": str(codes.dtype),
            "bit_depth": codes.dtype.itemsize * 8,
            "effective_bit_depth": _effective_bit_depth(codes),
            "channels_in_file": channels,
            "color_channels": codes.shape[2],
        },
        "alpha": _alpha_report(alpha),
        "color": {
            "icc_profile_present": decoded.icc_profile is not None,
            "icc_profile_description": cm.icc_description(decoded.icc_profile),
            "icc_profile_bytes": len(decoded.icc_profile) if decoded.icc_profile else 0,
            "resolved": color_space.to_dict(),
            "working_space": cm.WORKING_SPACE_DESCRIPTION,
            "working_signal": cm.SIGNAL_LINEAR_DISPLAY
            if color_space.signal == cm.SIGNAL_ENCODED_DISPLAY
            else color_space.signal,
        },
        "metadata": {
            "exif_present": bool(decoded.exif),
            "exif": decoded.exif,
            "text_keys": decoded.text_keys,
        },
        "clipping": _clipping_report(codes),
        "histograms": _histograms(normalized),
        "compression": {
            "jpeg_quality_estimate": decoded.jpeg_quality,
            "blockiness_ratio": _blockiness(normalized),
        },
        "integrity": {"structure_verified": decoded.verified, "fully_decoded": True},
        "pixel_sha256": pixels_hash,
        "duplicates": _find_duplicates(Path(inputs_root), stored, file_hash, pixels_hash),
    }

    if image_format not in MANDATORY_FORMATS:
        warnings.append(f"{image_format} is an optional input format; results depend on the local decoder")
    if color_space.origin == "untagged_default":
        warnings.append("no embedded ICC profile; assumed sRGB (override with source_color_space)")
    if codes.shape[2] == 1:
        warnings.append("greyscale source; the single channel is replicated to RGB")
    if alpha is not None:
        report = validation["alpha"]
        if report["opaque"]:
            warnings.append("opaque alpha channel discarded")
        else:
            warnings.append(
                f"alpha channel is not opaque (min {report['min']:.4f}); it is discarded, colour channels are "
                "used as stored"
            )
    if validation["samples"]["effective_bit_depth"] < validation["samples"]["bit_depth"] and codes.dtype == np.uint16:
        warnings.append("16-bit container holds 8-bit data; the extra bits carry no source information")
    orientation = decoded.exif.get("Orientation")
    if orientation not in (None, 1):
        warnings.append(f"EXIF orientation {orientation} is ignored; pixels are processed as stored")
    for name, channel in validation["clipping"]["channels"].items():
        if channel["at_max_fraction"] > CLIPPING_WARNING_FRACTION:
            warnings.append(f"possible highlight clipping in {name}: {channel['at_max_fraction']:.2%} at maximum")
        if channel["at_min_fraction"] > CLIPPING_WARNING_FRACTION:
            warnings.append(f"possible shadow clipping in {name}: {channel['at_min_fraction']:.2%} at minimum")
    if validation["compression"]["blockiness_ratio"] > BLOCKINESS_WARNING_RATIO:
        warnings.append(
            f"8x8 block-boundary discontinuities {validation['compression']['blockiness_ratio']:.2f}x the "
            "interior level suggest JPEG compression artefacts"
        )
    for duplicate in validation["duplicates"]:
        warnings.append(f"duplicate of previously ingested input {duplicate['stored_copy']} ({duplicate['match']})")
    megapixels = width * height / 1e6
    if not 4.0 <= megapixels <= 20.0:
        warnings.append(f"source is {megapixels:.2f} MP; the experiment expects an approximately 4K input")
    validation["warnings"] = warnings

    _write_sidecar(stored, file_hash, pixels_hash, source)
    return SourceImage(source, stored, file_hash, pixels_hash, codes, linear, color_space, validation, warnings)


# --------------------------------------------------------------------------- storage


def _store_read_only(source: Path, store_dir: Path, file_hash: str) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    stored = store_dir / f"{file_hash[:12]}_{source.name}"
    if stored.exists():
        if sha256_file(stored) != file_hash:
            raise IngestError(f"stored input {stored} exists with different content")
    else:
        shutil.copy2(source, stored)
    stored.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return stored


def _write_sidecar(stored: Path, file_hash: str, pixels_hash: str, source: Path) -> None:
    sidecar = stored.with_name(stored.name + ".json")
    if sidecar.exists():
        return
    sidecar.write_text(
        json.dumps(
            {
                "stored_copy": stored.name,
                "original_path": str(source),
                "sha256": file_hash,
                "pixel_sha256": pixels_hash,
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    sidecar.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _find_duplicates(inputs_root: Path, stored: Path, file_hash: str, pixels_hash: str) -> list[dict]:
    matches = []
    for sidecar in sorted(inputs_root.glob("*/*.json")):
        if sidecar.name == stored.name + ".json":
            continue
        try:
            record = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("sha256") == file_hash:
            match = "identical file"
        elif record.get("pixel_sha256") == pixels_hash:
            match = "identical pixels"
        else:
            continue
        matches.append({"stored_copy": str(sidecar.with_name(record.get("stored_copy", ""))), "match": match})
    return matches


# --------------------------------------------------------------------------- decoding


@dataclass
class _Decoded:
    pixels: np.ndarray
    decoder: str
    icc_profile: bytes | None = None
    exif: dict = field(default_factory=dict)
    text_keys: list[str] = field(default_factory=list)
    jpeg_quality: float | None = None
    verified: bool = False
    exr_chromaticities: tuple | None = None


def _decode(path: Path, image_format: str) -> _Decoded:
    try:
        if image_format == "TIFF":
            return _decode_tiff(path)
        if image_format == "OpenEXR":
            return _decode_exr(path)
        return _decode_raster(path, image_format)
    except IngestError:
        raise
    except Exception as error:  # decoders raise many unrelated types for corrupt data
        raise IngestError(f"{path.name} is corrupt or unreadable: {type(error).__name__}: {error}") from error


def _decode_raster(path: Path, image_format: str) -> _Decoded:
    import cv2
    from PIL import Image

    with Image.open(path) as image:
        image.verify()  # structure and checksum validation (PNG CRCs, etc.)
    with Image.open(path) as image:
        if image.mode in ("CMYK", "YCbCr", "LAB", "HSV"):
            raise IngestError(f"{image.mode} sources are not supported; convert to RGB first")
        image.load()  # full decode: raises on truncated data
        size = image.size
        icc = image.info.get("icc_profile") or None
        exif = _exif_summary(image.getexif())
        text_keys = sorted(k for k in image.info if k not in _STRUCTURAL_INFO_KEYS)
        quality = _jpeg_quality(image) if image_format == "JPEG" else None
        fallback = np.asarray(image) if image.mode in ("RGB", "RGBA", "L", "LA") else None

    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    decoder = "opencv"
    if pixels is None:
        if fallback is None:
            raise IngestError(f"no decoder could read {path.name}")
        pixels, decoder = fallback, "pillow"
    elif pixels.ndim == 3:
        code = cv2.COLOR_BGRA2RGBA if pixels.shape[2] == 4 else cv2.COLOR_BGR2RGB
        pixels = cv2.cvtColor(pixels, code)
    if (pixels.shape[1], pixels.shape[0]) != size:
        raise IngestError(f"decoders disagree on dimensions for {path.name}: {size} vs {pixels.shape[1::-1]}")
    return _Decoded(pixels, decoder, icc, exif, text_keys, quality, verified=True)


def _decode_tiff(path: Path) -> _Decoded:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        photometric = page.photometric.name if page.photometric is not None else "UNKNOWN"
        if photometric not in ("RGB", "MINISBLACK"):
            raise IngestError(f"TIFF photometric interpretation {photometric} is not supported")
        pixels = page.asarray()
        icc_tag = page.tags.get(34675)
        icc = bytes(icc_tag.value) if icc_tag is not None else None
        exif = {}
        exif_tag = page.tags.get("ExifTag")
        if exif_tag is not None and isinstance(exif_tag.value, dict):
            exif = {k: _plain(v) for k, v in exif_tag.value.items() if k in ("ColorSpace", "DateTimeOriginal")}
        for code, name in _EXIF_TAGS.items():
            tag = page.tags.get(code)
            if tag is not None:
                exif[name] = _plain(tag.value)
        text_keys = ["ImageDescription"] if page.tags.get(270) is not None else []
    if pixels.ndim == 3 and pixels.shape[0] in (3, 4) and pixels.shape[2] not in (3, 4):
        pixels = np.moveaxis(pixels, 0, -1)  # planar configuration
    return _Decoded(np.ascontiguousarray(pixels), "tifffile", icc, exif, text_keys, verified=True)


def _decode_exr(path: Path) -> _Decoded:
    import OpenEXR

    with OpenEXR.File(str(path), separate_channels=True) as exr:
        header = exr.header()
        channels = exr.channels()  # pixel data is only valid while the file is open
        names = [n for n in ("R", "G", "B") if n in channels]
        if len(names) != 3:
            if "Y" in channels:
                names = ["Y"]
            else:
                raise IngestError(f"EXR has no R, G, B or Y channels (found {sorted(channels)})")
        if "A" in channels:
            names.append("A")
        pixels = np.stack([np.array(channels[n].pixels, dtype=np.float32) for n in names], axis=-1)
    chroma = header.get("chromaticities")
    text_keys = sorted(k for k, v in header.items() if isinstance(v, str))
    chromaticities = tuple(chroma) if chroma is not None else None
    return _Decoded(pixels, "openexr", None, {}, text_keys, verified=True, exr_chromaticities=chromaticities)


def _exif_summary(exif) -> dict:
    summary = {name: _plain(exif.get(code)) for code, name in _EXIF_TAGS.items() if exif.get(code) is not None}
    try:
        sub = exif.get_ifd(0x8769)
    except Exception:
        sub = {}
    if sub.get(_EXIF_COLOR_SPACE) is not None:
        summary["ColorSpace"] = {1: "sRGB", 0xFFFF: "Uncalibrated"}.get(sub[_EXIF_COLOR_SPACE], sub[_EXIF_COLOR_SPACE])
    return summary


def _plain(value):
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace").strip("\0 ")
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _jpeg_quality(image) -> float | None:
    """IJG-style quality estimate from the luminance quantisation table (a heuristic)."""
    tables = getattr(image, "quantization", None)
    if not tables or 0 not in tables:
        return None
    scale = 100.0 * sum(tables[0]) / _IJG_LUMINANCE_TABLE_SUM
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return round(min(max(quality, 1.0), 100.0), 1)


# --------------------------------------------------------------------------- measurements


def _normalize(codes: np.ndarray) -> np.ndarray:
    if codes.dtype == np.uint8:
        return codes.astype(np.float32) / np.float32(255)
    if codes.dtype == np.uint16:
        return codes.astype(np.float32) / np.float32(65535)
    return codes.astype(np.float32, copy=True)


def _code_max(codes: np.ndarray) -> float:
    return {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(codes.dtype, 1.0)


def _effective_bit_depth(codes: np.ndarray) -> int:
    if codes.dtype == np.uint16 and not np.any(codes % 257):
        return 8
    return codes.dtype.itemsize * 8


def _alpha_report(alpha: np.ndarray | None) -> dict:
    if alpha is None:
        return {"present": False, "opaque": True, "min": None}
    top = _code_max(alpha)
    minimum = float(alpha.min()) / top
    return {"present": True, "opaque": minimum >= 1.0, "min": minimum}


def _channel_names(codes: np.ndarray) -> list[str]:
    return ["Y"] if codes.shape[2] == 1 else ["R", "G", "B"]


def _clipping_report(codes: np.ndarray) -> dict:
    top = _code_max(codes)
    pixels = codes.shape[0] * codes.shape[1]
    channels = {}
    for index, name in enumerate(_channel_names(codes)):
        plane = codes[..., index]
        channels[name] = {
            "at_min_fraction": float(np.count_nonzero(plane <= 0)) / pixels,
            "at_max_fraction": float(np.count_nonzero(plane >= top)) / pixels,
        }
    any_max = np.any(codes >= top, axis=2)
    return {"channels": channels, "any_channel_at_max_fraction": float(np.count_nonzero(any_max)) / pixels}


def _histograms(normalized: np.ndarray, bins: int = 256) -> dict:
    def hist(plane: np.ndarray) -> list[int]:
        index = np.clip(np.rint(plane * (bins - 1)), 0, bins - 1).astype(np.int64)
        return np.bincount(index.ravel(), minlength=bins).tolist()

    histograms = {name: hist(normalized[..., i]) for i, name in enumerate(_channel_names(normalized))}
    histograms["luma"] = hist(_luma(normalized))
    histograms["bins"] = bins
    histograms["domain"] = "encoded code values (Rec.709 luma weights for the luma histogram)"
    return histograms


def _luma(normalized: np.ndarray) -> np.ndarray:
    if normalized.shape[2] == 1:
        return normalized[..., 0]
    return normalized @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _blockiness(normalized: np.ndarray) -> float:
    """Ratio of mean luma steps across 8x8 block boundaries to steps inside blocks (about 1 when unblocked)."""
    luma = _luma(normalized)
    ratios = []
    for axis in (1, 0):
        steps = np.abs(np.diff(luma, axis=axis)).mean(axis=1 - axis)
        if steps.size < 16:
            continue
        boundary = np.zeros(steps.size, dtype=bool)
        boundary[7::8] = True
        inside = steps[~boundary].mean()
        if inside > 0:
            ratios.append(steps[boundary].mean() / inside)
    return float(np.mean(ratios)) if ratios else 1.0


def _reduced_ratio(width: int, height: int) -> str:
    ratio = Fraction(width, height)
    if ratio.denominator <= 64:
        return f"{ratio.numerator}:{ratio.denominator}"
    return f"{width / height:.4f}:1"
