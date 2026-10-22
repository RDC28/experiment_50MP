"""Research DNG export of a synthetic mosaic and its validation (PRD sections 14.6, 17.2 and 17.3).

A *synthetic mosaic DNG* holds simulated CFA-domain data. It is not camera-native
RAW. The file follows the DNG 1.4 layout real DNGs use:

* IFD 0: a small 8-bit sRGB preview (NewSubfileType 1) carrying the DNG
  identification, colour and provenance tags (DNGVersion, UniqueCameraModel,
  ColorMatrix1, CalibrationIlluminant1 = D65, AsShotNeutral, Orientation,
  ImageDescription with the provenance JSON),
* SubIFD 0: the full-resolution CFA image (NewSubfileType 0,
  PhotometricInterpretation 32803, 16-bit, uncompressed, CFARepeatPatternDim,
  CFAPattern, CFAPlaneColor, CFALayout, BlackLevel, WhiteLevel).

The writer never renames a TIFF: it writes DNG tags and then validates the file
three ways before returning it:

1. structure: every required tag is re-read with tifffile and compared with the
   model, and the stored mosaic must be bit-identical;
2. reader compatibility: LibRaw (through rawpy), a RAW-processing library used
   by many RAW applications, must open the file, report the same CFA pattern,
   black and white levels, white balance and colour matrix, and return the
   identical mosaic;
3. ExifTool ``-validate``, when ``exiftool`` is installed, must report no
   errors or warnings.

A file that fails any check raises :class:`exporters.ExportValidationError`. A
manifest sidecar always identifies the DNG as simulated.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .exporters import ExportResult, ExportValidationError, _json_or_none
from .ingestion import pixel_sha256, sha256_file
from .sensor_simulation import SyntheticMosaic

DNG_VERSION = (1, 4, 0, 0)
DNG_BACKWARD_VERSION = (1, 3, 0, 0)
UNIQUE_CAMERA_MODEL = "Synthetic RAW 50MP Lab SIMULATED sensor (synthetic mosaic, not a camera)"
PROFILE_NAME = "Synthetic RAW 50MP Lab assumed colour matrix (simulated)"
D65_ILLUMINANT = 21  # EXIF LightSource code for D65
PREVIEW_MAX_EDGE = 256
LIBRAW_DESCRIPTION = "LibRaw (rawpy) AHD demosaic with the embedded ColorMatrix1 and AsShotNeutral"
_RATIONAL_DENOMINATOR = 1_000_000
_MATRIX_TOLERANCE = 2e-5  # rational encoding precision plus float32 rounding inside LibRaw
_LIBRAW_MATRIX_TOLERANCE = 1e-3

# TIFF/DNG tag codes
_NEW_SUBFILE_TYPE = 254
_ORIENTATION = 274
_SUB_IFDS = 330
_CFA_REPEAT_PATTERN_DIM = 33421
_CFA_PATTERN = 33422
_DNG_VERSION = 50706
_DNG_BACKWARD_VERSION = 50707
_UNIQUE_CAMERA_MODEL = 50708
_CFA_PLANE_COLOR = 50710
_CFA_LAYOUT = 50711
_BLACK_LEVEL = 50714
_WHITE_LEVEL = 50717
_COLOR_MATRIX_1 = 50721
_AS_SHOT_NEUTRAL = 50728
_CALIBRATION_ILLUMINANT_1 = 50778
_PROFILE_NAME = 50936
_PHOTOMETRIC_CFA = 32803


def write_synthetic_dng(
    path: Path, mosaic: SyntheticMosaic, metadata: dict, preview_rgb8: np.ndarray, *, run_exiftool: bool = True
) -> ExportResult:
    """Write a synthetic mosaic DNG and validate it; ``metadata`` is stored as JSON in ImageDescription."""
    import tifffile

    dn = mosaic.dn
    if dn.dtype != np.uint16 or dn.ndim != 2 or dn.shape[0] % 2 or dn.shape[1] % 2:
        raise ValueError(f"expected an (H, W) uint16 mosaic with even sides, got {dn.shape} {dn.dtype}")
    if preview_rgb8.dtype != np.uint8 or preview_rgb8.ndim != 3 or preview_rgb8.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) uint8 preview, got {preview_rgb8.shape} {preview_rgb8.dtype}")
    if dn.nbytes > 3.5e9:
        raise ValueError("DNG is a classic (32-bit offset) TIFF; this mosaic is too large for one file")
    if "synthetic" not in json.dumps(metadata).lower():
        raise ValueError("DNG metadata must carry the synthetic provenance label")
    model = mosaic.model
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    description = json.dumps(metadata, sort_keys=True)
    ifd0_tags = [
        (_DNG_VERSION, "B", 4, DNG_VERSION, True),
        (_DNG_BACKWARD_VERSION, "B", 4, DNG_BACKWARD_VERSION, True),
        (_UNIQUE_CAMERA_MODEL, "s", 0, UNIQUE_CAMERA_MODEL, True),
        (_COLOR_MATRIX_1, "2i", 9, _rationals(model.xyz_to_camera().ravel()), True),
        (_CALIBRATION_ILLUMINANT_1, "H", 1, D65_ILLUMINANT, True),
        (_AS_SHOT_NEUTRAL, "2I", 3, _rationals(model.as_shot_neutral()), True),
        (_PROFILE_NAME, "s", 0, PROFILE_NAME, True),
        (_ORIENTATION, "H", 1, 1, True),
    ]
    raw_tags = [
        (_CFA_REPEAT_PATTERN_DIM, "H", 2, (2, 2), True),
        (_CFA_PATTERN, "B", 4, tuple(int(c) for c in model.channel_map.ravel()), True),
        (_CFA_PLANE_COLOR, "B", 3, (0, 1, 2), True),
        (_CFA_LAYOUT, "H", 1, 1, True),
        (_BLACK_LEVEL, "I", 1, model.black, True),
        (_WHITE_LEVEL, "I", 1, model.white, True),
    ]
    with tifffile.TiffWriter(path, bigtiff=False, byteorder="<") as writer:
        writer.write(
            preview_rgb8,
            photometric="rgb",
            subfiletype=1,
            subifds=1,
            description=description,
            software=metadata.get("software", "synthetic_raw_lab"),
            extratags=ifd0_tags,
            metadata=None,
        )
        writer.write(dn, photometric=_PHOTOMETRIC_CFA, compression=None, subfiletype=0, extratags=raw_tags)

    expected_hash = pixel_sha256(dn)
    checks = structural_checks(path, mosaic, json.loads(description), expected_hash)
    checks.update(libraw_checks(path, mosaic))
    if run_exiftool:
        exiftool = exiftool_validate(path)
        if exiftool is not None:
            checks["exiftool_validate_ok"] = exiftool["ok"]
    result = ExportResult(
        path,
        "synthetic_mosaic_dng",
        dn.shape[1],
        dn.shape[0],
        "uint16",
        1,
        "NONE",
        sha256_file(path),
        expected_hash,
        path.stat().st_size,
        checks,
    )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ExportValidationError(f"{path.name} failed DNG validation: {', '.join(failed)}")
    return result


def structural_checks(path: Path, mosaic: SyntheticMosaic, metadata: dict, expected_hash: str) -> dict[str, bool]:
    """Re-read every DNG tag the format requires for a CFA image and compare it with the model."""
    import tifffile

    model = mosaic.model
    with tifffile.TiffFile(path) as tif:
        ifd0 = tif.pages[0]
        sub_offsets = ifd0.tags.get(_SUB_IFDS)
        raw_pages = ifd0.pages if sub_offsets is not None else None
        raw = raw_pages[0] if raw_pages is not None and len(raw_pages) else None

        def value(page, code):
            tag = page.tags.get(code) if page is not None else None
            return None if tag is None else tag.value

        checks = {
            "classic_tiff_little_endian": not tif.is_bigtiff and tif.byteorder == "<",
            "single_main_ifd": len(tif.pages) == 1,
            "dng_version": _bytes(value(ifd0, _DNG_VERSION)) == DNG_VERSION,
            "dng_backward_version": _bytes(value(ifd0, _DNG_BACKWARD_VERSION)) == DNG_BACKWARD_VERSION,
            "unique_camera_model_synthetic": "SIMULATED" in str(value(ifd0, _UNIQUE_CAMERA_MODEL) or ""),
            "preview_ifd0": value(ifd0, _NEW_SUBFILE_TYPE) == 1 and ifd0.photometric == tifffile.PHOTOMETRIC.RGB,
            "orientation": value(ifd0, _ORIENTATION) == 1,
            "color_matrix_1": _matrix_close(value(ifd0, _COLOR_MATRIX_1), model.xyz_to_camera().ravel()),
            "calibration_illuminant_d65": value(ifd0, _CALIBRATION_ILLUMINANT_1) == D65_ILLUMINANT,
            "as_shot_neutral": _matrix_close(value(ifd0, _AS_SHOT_NEUTRAL), model.as_shot_neutral()),
            "provenance_roundtrip": _json_or_none(ifd0.description) == metadata,
            "raw_subifd": raw is not None and value(raw, _NEW_SUBFILE_TYPE) == 0,
            "photometric_cfa": raw is not None and int(raw.photometric) == _PHOTOMETRIC_CFA,
            "samples_per_pixel_1": raw is not None and raw.samplesperpixel == 1,
            "bits_per_sample_16": raw is not None and raw.bitspersample == 16,
            "uncompressed": raw is not None and raw.compression == tifffile.COMPRESSION.NONE,
            "dimensions": raw is not None and (raw.imagewidth, raw.imagelength) == (mosaic.width, mosaic.height),
            "cfa_repeat_pattern_dim": _tuple(value(raw, _CFA_REPEAT_PATTERN_DIM)) == (2, 2),
            "cfa_pattern": _bytes(value(raw, _CFA_PATTERN)) == tuple(int(c) for c in model.channel_map.ravel()),
            "cfa_plane_color_rgb": _bytes(value(raw, _CFA_PLANE_COLOR)) == (0, 1, 2),
            "cfa_layout_rectangular": value(raw, _CFA_LAYOUT) == 1,
            "black_level": _scalar(value(raw, _BLACK_LEVEL)) == model.black,
            "white_level": _scalar(value(raw, _WHITE_LEVEL)) == model.white,
        }
        checks["pixels_identical"] = raw is not None and pixel_sha256(raw.asarray()) == expected_hash
    return checks


def libraw_checks(path: Path, mosaic: SyntheticMosaic) -> dict[str, bool]:
    """Open the DNG with LibRaw and check that it interprets every tag as written."""
    import rawpy

    model = mosaic.model
    try:
        raw = rawpy.imread(str(path))
    except Exception:  # noqa: BLE001 - any LibRaw failure means the file is not readable
        return {"libraw_opens": False}
    with raw:
        letters = raw.color_desc.decode()
        pattern = np.array([["RGB".index(letters[i]) for i in row] for row in raw.raw_pattern])
        balance = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
        expected_balance = 1.0 / model.as_shot_neutral()
        return {
            "libraw_opens": True,
            "libraw_bayer": raw.raw_type == rawpy.RawType.Flat and raw.num_colors == 3,
            "libraw_cfa_pattern": np.array_equal(pattern, model.channel_map),
            "libraw_black_level": all(int(b) == model.black for b in raw.black_level_per_channel),
            "libraw_white_level": int(raw.white_level) == model.white,
            "libraw_white_balance": np.allclose(balance / balance[1], expected_balance / expected_balance[1], 1e-4),
            "libraw_color_matrix": np.allclose(
                raw.color_matrix[:, :3], np.asarray(model.color_matrix), atol=_LIBRAW_MATRIX_TOLERANCE
            ),
            "libraw_mosaic_identical": np.array_equal(raw.raw_image_visible, mosaic.dn),
        }


def develop_with_libraw(path: Path, mosaic: SyntheticMosaic) -> np.ndarray:
    """Decode the DNG with LibRaw's own pipeline (AHD demosaic, embedded matrix) to working-space linear RGB.

    LibRaw outputs linear sRGB (Rec.709 primaries, D65) in 16 bits with the white
    level at 65535; dividing by the exposure scale puts it on the input's scale.
    Automatic brightening and LibRaw's data-driven maximum adjustment are off, so
    the scale comes from the DNG tags alone. LibRaw clips at its white level.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            adjust_maximum_thr=0.0,
            use_camera_wb=True,
            user_flip=0,
        )
    linear = np.empty(rgb16.shape, dtype=np.float32)
    scale = np.float32(1.0 / (65535.0 * mosaic.model.exposure_scale))
    for start in range(0, rgb16.shape[0], 512):
        linear[start : start + 512] = rgb16[start : start + 512].astype(np.float32) * scale
    return linear


def exiftool_validate(path: Path) -> dict | None:
    """Run ``exiftool -validate`` when ExifTool is installed; None when it is not available."""
    executable = shutil.which("exiftool")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "-validate", "-warning", "-error", "-a", "-s", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    fields = [tuple(part.strip() for part in line.split(":", 1)) for line in lines]
    issues = [value for name, value in fields if name != "Validate"]
    unexpected = [issue for issue in issues if issue not in _EXIFTOOL_ACCEPTED_WARNINGS]
    reported = any(name == "Validate" for name, _ in fields)
    return {"ok": result.returncode == 0 and reported and not unexpected, "output": lines, "unexpected": unexpected}


# ExifTool prefers LONG for SubIFDs; tifffile writes the IFD type, which TIFF Technical Note 1 and the
# DNG 1.4 specification both allow ("Type: LONG or IFD"). Every other warning or error fails the check.
_EXIFTOOL_ACCEPTED_WARNINGS = frozenset({"Non-standard format (ifd) for IFD0 0x014a A100DataOffset"})


def preview_from_linear(linear: np.ndarray, max_edge: int = PREVIEW_MAX_EDGE) -> np.ndarray:
    """8-bit sRGB preview of the processed RGB input for the DNG's IFD 0 (area downscale in linear light)."""
    from . import baselines
    from . import color_management as cm

    height, width = linear.shape[:2]
    scale = min(1.0, max_edge / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = baselines.downsample_area(np.ascontiguousarray(linear, dtype=np.float32), *size)
    encoded, _ = cm.encode_srgb_uint16(small)
    return cm.srgb_uint16_to_uint8(encoded)


# --------------------------------------------------------------------------- helpers


def _rationals(values) -> list[int]:
    out: list[int] = []
    for v in values:
        out += [int(round(float(v) * _RATIONAL_DENOMINATOR)), _RATIONAL_DENOMINATOR]
    return out


def _matrix_close(stored, expected) -> bool:
    if stored is None:
        return False
    pairs = np.asarray(stored, dtype=np.float64).reshape(-1, 2)
    decoded = pairs[:, 0] / pairs[:, 1]
    expected = np.asarray(expected, dtype=np.float64).ravel()
    return decoded.shape == expected.shape and bool(np.allclose(decoded, expected, atol=_MATRIX_TOLERANCE))


def _bytes(value) -> tuple | None:
    if value is None:
        return None
    return tuple(value) if not isinstance(value, int) else (value,)


def _tuple(value) -> tuple | None:
    if value is None:
        return None
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _scalar(value):
    if isinstance(value, (tuple, list)):
        return value[0] if len(value) == 1 else None
    return value


def write_mosaic_tiff(path: Path, mosaic: SyntheticMosaic, metadata: dict) -> ExportResult:
    """Simulated CFA data as a plain 16-bit greyscale TIFF (not a DNG), provenance JSON in ImageDescription."""
    import tifffile

    if "synthetic" not in json.dumps(metadata).lower():
        raise ValueError("mosaic metadata must carry the synthetic provenance label")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    description = json.dumps(metadata, sort_keys=True)
    expected_hash = pixel_sha256(mosaic.dn)
    tifffile.imwrite(
        path,
        mosaic.dn,
        photometric="minisblack",
        compression="zlib",
        predictor=2,
        description=description,
        software=metadata.get("software", "synthetic_raw_lab"),
        metadata=None,
    )
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        checks = {
            "decodes": True,
            "single_image": len(tif.pages) == 1,
            "dimensions": (page.imagewidth, page.imagelength) == (mosaic.width, mosaic.height),
            "bits_per_sample_16": page.bitspersample == 16,
            "lossless_compression": page.compression == tifffile.COMPRESSION.ADOBE_DEFLATE,
            "provenance_roundtrip": _json_or_none(page.description) == json.loads(description),
            "pixels_identical": pixel_sha256(page.asarray()) == expected_hash,
        }
    result = ExportResult(
        path,
        "synthetic_mosaic_tiff",
        mosaic.width,
        mosaic.height,
        "uint16",
        1,
        "ADOBE_DEFLATE",
        sha256_file(path),
        expected_hash,
        path.stat().st_size,
        checks,
    )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ExportValidationError(f"{path.name} failed validation: {', '.join(failed)}")
    return result
