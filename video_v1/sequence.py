"""The DNG sequence: one CFA DNG per frame with CinemaDNG frame metadata, plus its manifest.

Every file carries the tags of the v6 CFA DNG (`prd_v6.dng.dng_tags`, checked by a test) with the
clip's BaselineExposure, plus the three per-frame tags the CinemaDNG profile of DNG 1.4 defines or
uses: FrameRate (51044), TimeCodes (51043, SMPTE 12M) and ImageNumber (37393, TIFF/EP). Files are
named `<clip>_<frame:06d>.dng` from frame 0, so any CinemaDNG reader orders them by name.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile

from prd_v6 import constants
from prd_v6 import dng as v6dng

SOFTWARE = "video_v1"
MAX_TIMECODE_FPS = 30  # SMPTE 12M frame numbers stop at 29; faster clips get no TimeCodes tag
DROP_FRAME_RATES = {Fraction(30000, 1001)}
_CLIP_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def clip_name(source: str | Path) -> str:
    """A file-safe sequence name from the source file stem (letters, digits, '-' and '_'; 40 at most)."""
    stem = _CLIP_NAME.sub("_", Path(str(source)).stem).strip("_-")[:40].strip("_-")
    return stem or "clip"


def frame_name(clip: str, index: int) -> str:
    return f"{clip}_{index:06d}.dng"


def description(index: int, frames: int, clip: str) -> str:
    return (
        f"Frame {index} of {frames} (0-based) of {clip}. Synthetic RAW reconstructed from a video frame by "
        "video_v1 with the prd_v6 pipeline; not a camera capture."
    )


def nominal_fps(rate: Fraction) -> int:
    return int(round(float(rate)))


def timecode(index: int, rate: Fraction) -> tuple[int, int, int, int, bool] | None:
    """(hours, minutes, seconds, frames, drop_frame) of frame `index` from 00:00:00:00, or None above 30 fps."""
    base = nominal_fps(rate)
    if base > MAX_TIMECODE_FPS or base < 1:
        return None
    drop = Fraction(rate) in DROP_FRAME_RATES
    n = int(index)
    if drop:  # 29.97 drop-frame: skip frame numbers 0 and 1 each minute except every tenth minute
        tens, rest = divmod(n, 17982)
        n += 18 * tens + (2 * ((rest - 2) // 1798) if rest >= 2 else 0)
    frames = n % base
    seconds = n // base % 60
    minutes = n // (base * 60) % 60
    hours = n // (base * 3600) % 24
    return hours, minutes, seconds, frames, drop


def timecode_text(index: int, rate: Fraction) -> str | None:
    code = timecode(index, rate)
    if code is None:
        return None
    hours, minutes, seconds, frames, drop = code
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{';' if drop else ':'}{frames:02d}"


def _bcd(value: int) -> int:
    return (value // 10) << 4 | value % 10


def timecode_bytes(index: int, rate: Fraction) -> tuple[int, ...] | None:
    """The 8-byte SMPTE 12M time code DNG stores (SMPTE 331M section 8.3): time bytes in BCD, user bits 0."""
    code = timecode(index, rate)
    if code is None:
        return None
    hours, minutes, seconds, frames, drop = code
    return (_bcd(frames) | (0x40 if drop else 0), _bcd(seconds), _bcd(minutes), _bcd(hours), 0, 0, 0, 0)


def frame_rate_rational(rate: Fraction) -> tuple[int, int]:
    rate = Fraction(rate)
    return rate.numerator, rate.denominator


def base_tags(
    width: int,
    height: int,
    *,
    cfa_pattern: str,
    color_matrix,
    as_shot_neutral,
    black: int = constants.BLACK_16,
    white: int = constants.WHITE_16,
    baseline_exposure: float = 0.0,
) -> list[tuple]:
    """The v6 CFA DNG tag set (`prd_v6.dng.dng_tags`), with the clip's BaselineExposure."""
    return v6dng.dng_tags(
        width, height, cfa_pattern=cfa_pattern, color_matrix=color_matrix, as_shot_neutral=as_shot_neutral,
        black=black, white=white, baseline_exposure=baseline_exposure,
    )  # fmt: skip


def frame_tags(index: int, rate: Fraction) -> list[tuple]:
    numerator, denominator = frame_rate_rational(rate)
    tags = [
        (37393, "I", 1, (int(index),), False),  # ImageNumber (TIFF/EP)
        (51044, "2i", 1, (numerator, denominator), False),  # FrameRate (CinemaDNG)
    ]
    code = timecode_bytes(index, rate)
    if code is not None:
        tags.append((51043, "B", 8, code, False))  # TimeCodes (CinemaDNG)
    return tags


def write_frame(
    path: str | Path,
    samples: np.ndarray,
    *,
    index: int,
    frames: int,
    clip: str,
    rate: Fraction,
    color_matrix,
    as_shot_neutral,
    baseline_exposure: float = 0.0,
    cfa_pattern: str = constants.CFA_PATTERN,
) -> Path:
    """Write one uint16 (H, W) RGGB frame as the v6 CFA DNG, plus the frame tags."""
    samples = np.ascontiguousarray(samples)
    if samples.dtype != np.uint16 or samples.ndim != 2:
        raise TypeError("a frame DNG needs (H, W) uint16 samples")
    height, width = samples.shape
    if height % 2 or width % 2:
        raise ValueError("a CFA frame needs even height and width")
    tags = base_tags(
        width, height, cfa_pattern=cfa_pattern, color_matrix=color_matrix, as_shot_neutral=as_shot_neutral,
        baseline_exposure=baseline_exposure,
    )  # fmt: skip
    tags += frame_tags(index, rate)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, byteorder="<", bigtiff=False) as writer:
        writer.write(
            samples,
            photometric=v6dng.PHOTOMETRIC_CFA,
            subfiletype=0,
            rowsperstrip=v6dng._ROWS_PER_STRIP,
            compression=None,
            extrasamples=False,
            description=description(index, frames, clip),
            software=SOFTWARE,
            metadata=None,
            extratags=tags,
        )
    return path
