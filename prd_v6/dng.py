"""16-bit DNG and TIFF writers for the three outputs, and their read-back checks (sections 5.4, 6.7).

The 50 MP output is a CFA DNG (one RGGB sample per photosite, like the ARW). The 2K and 4K outputs
are LinearRaw DNGs (three camera-RGB samples per pixel). Both are uncompressed, little-endian, with
the raw image in IFD 0, which Adobe Camera Raw, LibRaw and `dng_validate` all accept.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile

from prd_v6 import constants

PHOTOMETRIC_CFA = 32803
PHOTOMETRIC_LINEAR_RAW = 34892
D65 = 21
MAKE = "Synthetic"
MODEL = "Sony ILCE-1M2 synthetic (exp50 v6)"
UNIQUE_CAMERA_MODEL = "Synthetic Sony ILCE-1M2 exp50 v6"
DISCLOSURE = "Synthetic RAW reconstructed from an RGB render by prd_v6; not a camera capture."
# CFAPattern colour codes (TIFF/EP): 0 red, 1 green, 2 blue.
_COLOUR_CODE = {"R": 0, "G": 1, "B": 2}
_ROWS_PER_STRIP = 64


def _rationals(values, *, signed: bool) -> tuple[int, ...]:
    encoded: list[int] = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        fraction = Fraction(float(value)).limit_denominator(1_000_000)
        if not signed and fraction < 0:
            raise ValueError("negative value in an unsigned RATIONAL tag")
        encoded.extend((fraction.numerator, fraction.denominator))
    return tuple(encoded)


def write_dng(
    path: str | Path,
    samples: np.ndarray,
    *,
    cfa_pattern: str | None,
    color_matrix: np.ndarray,
    as_shot_neutral,
    black: int = constants.BLACK_16,
    white: int = constants.WHITE_16,
    description: str = DISCLOSURE,
) -> Path:
    """Write uint16 samples: (H, W) with a 2x2 `cfa_pattern` such as "RGGB", or (H, W, 3) LinearRaw."""
    samples = np.ascontiguousarray(samples)
    if samples.dtype != np.uint16:
        raise TypeError("DNG samples must be uint16")
    cfa = cfa_pattern is not None
    if cfa and (samples.ndim != 2 or len(cfa_pattern) != 4 or set(cfa_pattern) - set(_COLOUR_CODE)):
        raise ValueError("a CFA DNG needs (H, W) samples and a 2x2 pattern of R, G and B")
    if not cfa and (samples.ndim != 3 or samples.shape[2] != 3):
        raise ValueError("a LinearRaw DNG needs (H, W, 3) samples")
    height, width = samples.shape[:2]
    channels = 1 if cfa else 3
    tags = [
        (50706, "B", 4, (1, 4, 0, 0), False),  # DNGVersion
        (50707, "B", 4, (1, 1, 0, 0), False),  # DNGBackwardVersion
        (271, "s", 0, MAKE, False),
        (272, "s", 0, MODEL, False),
        (50708, "s", 0, UNIQUE_CAMERA_MODEL, False),
        (274, "H", 1, (1,), False),  # Orientation
        (50713, "H", 2, (2, 2) if cfa else (1, 1), False),  # BlackLevelRepeatDim
        (50714, "2I", 4 if cfa else 3, _rationals([black] * (4 if cfa else 3), signed=False), False),
        (50717, "I", channels, (white,) * channels, False),  # WhiteLevel
        (50721, "2i", 9, _rationals(color_matrix, signed=True), False),  # ColorMatrix1
        (50778, "H", 1, (D65,), False),  # CalibrationIlluminant1
        (50728, "2I", 3, _rationals(np.asarray(as_shot_neutral, dtype=np.float64)[:3], signed=False), False),
        (50719, "I", 2, (0, 0), False),  # DefaultCropOrigin
        (50720, "I", 2, (width, height), False),  # DefaultCropSize
        (50730, "2i", 1, _rationals([0.0], signed=True), False),  # BaselineExposure
    ]
    if cfa:
        tags += [
            (33421, "H", 2, (2, 2), False),  # CFARepeatPatternDim
            (33422, "B", 4, tuple(_COLOUR_CODE[c] for c in cfa_pattern), False),  # CFAPattern
            (50710, "B", 3, (0, 1, 2), False),  # CFAPlaneColor
            (50711, "H", 1, (1,), False),  # CFALayout
        ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, byteorder="<", bigtiff=False) as writer:
        writer.write(
            samples,
            photometric=PHOTOMETRIC_CFA if cfa else PHOTOMETRIC_LINEAR_RAW,
            planarconfig="contig" if not cfa else None,
            subfiletype=0,
            rowsperstrip=_ROWS_PER_STRIP,
            compression=None,
            description=description,
            software="prd_v6",
            metadata=None,
            extratags=tags,
        )
    return path


def read_dng_tags(path: str | Path) -> dict:
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        return {tag.name: tag.value for tag in page.tags.values()}


def read_samples(path: str | Path) -> np.ndarray:
    """The stored samples of IFD 0, exactly as written (section 6.2 reads LinearRaw outputs this way)."""
    with tifffile.TiffFile(path) as container:
        return container.pages[0].asarray()


def libraw_check(path: str | Path, *, expected_hw: tuple[int, int], cfa_pattern: str | None) -> dict:
    """LibRaw must parse the file at the expected size (and, for a CFA DNG, at the expected phase)."""
    import rawpy

    record: dict = {"file": Path(path).name, "expected": list(expected_hw)}
    try:
        with rawpy.imread(str(path)) as raw:
            record["size"] = [int(raw.sizes.height), int(raw.sizes.width)]
            record["black_level_per_channel"] = [int(v) for v in raw.black_level_per_channel]
            record["white_level"] = int(raw.white_level)
            problems = []
            if tuple(record["size"]) != tuple(expected_hw):
                problems.append(f"size {record['size']} != {list(expected_hw)}")
            if cfa_pattern is not None:
                desc = raw.color_desc.decode()
                phase = "".join(desc[int(i)] for i in np.asarray(raw.raw_colors_visible[:2, :2]).reshape(-1))
                record["cfa_phase"] = phase
                visible = raw.raw_image_visible
                record["visible_shape"] = list(visible.shape)
                if phase != cfa_pattern:
                    problems.append(f"CFA phase {phase} != {cfa_pattern}")
                if tuple(visible.shape) != tuple(expected_hw):
                    problems.append(f"visible area {list(visible.shape)} != {list(expected_hw)}")
            developed = raw.postprocess(half_size=True, output_bps=16, no_auto_bright=True, user_flip=0)
            record["developed_half_size"] = list(developed.shape)
            record["status"] = "failed" if problems else "passed"
            record["problems"] = problems
    except Exception as error:  # noqa: BLE001 - any LibRaw failure is a failed gate, recorded
        record.update(status="failed", problems=[f"{type(error).__name__}: {error}"])
    return record


def write_tiff16(path: str | Path, encoded: np.ndarray, icc: bytes) -> Path:
    """16-bit sRGB TIFF with an embedded ICC profile; Photoshop opens it as 16 bits/channel."""
    from prd_v6.colour import to_uint16

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        to_uint16(encoded),
        photometric="rgb",
        planarconfig="contig",
        compression=None,
        iccprofile=icc,
        rowsperstrip=_ROWS_PER_STRIP,
        metadata=None,
        software="prd_v6",
    )
    return path


def tiff16_check(path: str | Path, expected_hw: tuple[int, int]) -> dict:
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        shape = list(page.shape)
        record = {
            "file": Path(path).name,
            "shape": shape,
            "dtype": str(page.dtype),
            "bits_per_sample": page.bitspersample,
            "has_icc": "InterColorProfile" in page.tags,
        }
    ok = shape == [*expected_hw, 3] and record["dtype"] == "uint16" and record["has_icc"]
    record["status"] = "passed" if ok else "failed"
    return record
