"""Decoded video planes to linear light in BT.709 primaries (ITU-T H.273, ITU-R BT.709, BT.2020, BT.2100).

The v6 pipeline takes linear sRGB (BT.709 primaries, D65, 1.0 = render white = sensor clip). A video
frame gets there in four exact steps: chroma upsampling at the stream's chroma siting, the YCbCr
matrix with the stream's range (limited or full, any bit depth), the inverse of the stream's transfer
function, and a primaries change to BT.709 when the stream uses other primaries. Values outside
[0, 1] are kept: the camera gamut is wider than BT.709, and the v6 output stage clips at the sensor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from prd_v6 import colour

# H.273 matrix coefficients (Kr, Kb) by ffprobe `color_space` name.
MATRICES = {
    "bt709": (0.2126, 0.0722),
    "smpte170m": (0.299, 0.114),
    "bt470bg": (0.299, 0.114),
    "bt2020nc": (0.2627, 0.0593),
    "smpte240m": (0.212, 0.087),
    "fcc": (0.30, 0.11),
}
RGB_MATRIX = "gbr"  # the planes already hold R'G'B' (ffmpeg's GBR order)

# CIE 1931 xy of the red, green and blue primaries by ffprobe `color_primaries` name; D65 white for all.
PRIMARIES = {
    "bt709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "bt470bg": ((0.640, 0.330), (0.290, 0.600), (0.150, 0.060)),
    "smpte170m": ((0.630, 0.340), (0.310, 0.595), (0.155, 0.070)),
    "smpte240m": ((0.630, 0.340), (0.310, 0.595), (0.155, 0.070)),
    "bt2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    "smpte432": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),  # Display P3
}
D65_XY = (0.3127, 0.3290)

# Transfer functions by ffprobe `color_transfer` name. The BT.709 family shares one OETF.
BT709_FAMILY = {"bt709", "smpte170m", "bt2020-10", "bt2020-12", "iec61966-2-4", "bt1361e"}
TRANSFERS = BT709_FAMILY | {"iec61966-2-1", "arib-std-b67", "smpte2084", "linear"}
# PQ is display light in nits; this level maps to the sensor clip (a common HDR10 mastering peak).
PQ_CLIP_NITS = 1000.0

# Chroma siting (H.273 chroma_sample_loc_type, ffprobe `chroma_location`): sample offset in luma pixels.
CHROMA_SITING = {
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "topleft": (0.0, 0.0),
    "top": (0.5, 0.0),
    "bottomleft": (0.0, 1.0),
    "bottom": (0.5, 1.0),
}

UNSPECIFIED = {None, "", "unknown", "unspecified", "reserved"}


@dataclass(frozen=True)
class Signal:
    """How the decoded planes encode colour; `assumed` names the fields that were not tagged."""

    matrix: str
    primaries: str
    transfer: str
    full_range: bool
    bits: int
    subsampling: tuple[int, int]  # (horizontal, vertical) chroma subsampling factors
    chroma_location: str
    assumed: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


def resolve(
    tags: dict, *, width: int, height: int, bits: int, subsampling, rgb: bool, full_range_format: bool
) -> Signal:
    """Fill untagged fields the way players do (HD: BT.709; SD: BT.601) and record every assumption."""
    assumed: list[str] = []
    warnings: list[str] = []
    hd = height >= 720 or width >= 1280

    matrix = tags.get("color_space")
    if rgb:
        matrix = RGB_MATRIX
    elif matrix in UNSPECIFIED:
        matrix = "bt709" if hd else "smpte170m"
        assumed.append("matrix")
    elif matrix == "bt2020c":
        warnings.append("constant-luminance BT.2020 is decoded as non-constant luminance")
        matrix = "bt2020nc"
    elif matrix not in MATRICES:
        warnings.append(f"matrix {matrix!r} is not supported; decoded as BT.709")
        matrix = "bt709"

    primaries = tags.get("color_primaries")
    if primaries in UNSPECIFIED:
        primaries = "bt2020" if matrix == "bt2020nc" else ("bt709" if hd or rgb else "smpte170m")
        assumed.append("primaries")
    elif primaries not in PRIMARIES:
        warnings.append(f"primaries {primaries!r} are not supported; decoded as BT.709")
        primaries = "bt709"

    transfer = tags.get("color_transfer")
    if transfer in UNSPECIFIED:
        transfer = "iec61966-2-1" if rgb else "bt709"
        assumed.append("transfer")
    elif transfer not in TRANSFERS:
        warnings.append(f"transfer {transfer!r} is not supported; decoded with the BT.709 OETF")
        transfer = "bt709"

    tagged_range = tags.get("color_range")
    if full_range_format or tagged_range in {"pc", "jpeg", "full"}:
        full_range = True
    elif tagged_range in {"tv", "mpeg", "limited"}:
        full_range = False
    else:
        full_range = rgb
        assumed.append("range")

    location = tags.get("chroma_location")
    if tuple(subsampling) == (1, 1):
        location = "none"
    elif location in UNSPECIFIED or location not in CHROMA_SITING:
        location = "topleft" if matrix == "bt2020nc" else "left"
        assumed.append("chroma_location")

    return Signal(
        matrix=matrix,
        primaries=primaries,
        transfer=transfer,
        full_range=full_range,
        bits=int(bits),
        subsampling=(int(subsampling[0]), int(subsampling[1])),
        chroma_location=location,
        assumed=tuple(assumed),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------- chroma


def _keys_weights(t: np.ndarray) -> np.ndarray:
    """Catmull-Rom (Keys, a = -0.5) weights of taps -1, 0, 1, 2 at fraction t; exact on linear ramps."""
    t = t[:, None]
    d = np.abs(np.array([-1.0, 0.0, 1.0, 2.0])[None, :] - t)
    near = 1.5 * d**3 - 2.5 * d**2 + 1.0
    far = -0.5 * d**3 + 2.5 * d**2 - 4.0 * d + 2.0
    return np.where(d <= 1.0, near, np.where(d < 2.0, far, 0.0))


def _upsample_axis(plane: np.ndarray, length: int, factor: int, offset: float, axis: int) -> np.ndarray:
    """Output index i reads input position (i - offset) / factor; edges are replicated."""
    position = (np.arange(length) - offset) / factor
    base = np.floor(position)
    weights = _keys_weights(position - base).astype(np.float32)
    taps = np.clip(base[:, None].astype(np.intp) + np.arange(-1, 3)[None, :], 0, plane.shape[axis] - 1)
    gathered = np.take(plane, taps, axis=axis)  # the axis becomes (length, 4)
    shape = [1] * gathered.ndim
    shape[axis], shape[axis + 1] = length, 4
    return (gathered * weights.reshape(shape)).sum(axis=axis + 1, dtype=np.float32)


def upsample_chroma(plane: np.ndarray, out_hw: tuple[int, int], subsampling, location: str) -> np.ndarray:
    """Bicubic (Catmull-Rom) chroma upsampling that puts each chroma sample at its sited luma position."""
    plane = np.asarray(plane, dtype=np.float32)
    sx, sy = subsampling
    if (sx, sy) == (1, 1):
        return plane
    dx, dy = CHROMA_SITING[location]
    if sx > 1:
        plane = _upsample_axis(plane, out_hw[1], sx, dx, axis=1)
    if sy > 1:
        plane = _upsample_axis(plane, out_hw[0], sy, dy, axis=0)
    return plane


# ---------------------------------------------------------------- matrix and range


def _normalise(plane: np.ndarray, bits: int, full_range: bool, *, chroma: bool) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float32)
    if full_range:
        top = float(2**bits - 1)
        return (plane - 2 ** (bits - 1)) / top if chroma else plane / top
    scale = float(2 ** (bits - 8))
    return (plane - 128.0 * scale) / (224.0 * scale) if chroma else (plane - 16.0 * scale) / (219.0 * scale)


def planes_to_rgb(planes, signal: Signal) -> np.ndarray:
    """Integer planes (Y, Cb, Cr) or (G, B, R) to non-linear R'G'B' floats, chroma upsampled first."""
    first = np.asarray(planes[0])
    hw = first.shape[:2]
    if signal.matrix == RGB_MATRIX:
        g, b, r = (_normalise(p, signal.bits, signal.full_range, chroma=False) for p in planes)
        return np.stack([r, g, b], axis=-1)
    y = _normalise(first, signal.bits, signal.full_range, chroma=False)
    cb, cr = (
        upsample_chroma(
            _normalise(p, signal.bits, signal.full_range, chroma=True), hw, signal.subsampling, signal.chroma_location
        )
        for p in planes[1:3]
    )
    kr, kb = MATRICES[signal.matrix]
    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / (1.0 - kr - kb)
    return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)


# ---------------------------------------------------------------- transfer


def bt709_inverse_oetf(encoded: np.ndarray) -> np.ndarray:
    """Scene light from a BT.709 (and BT.601, BT.2020) signal; the linear toe extends below zero."""
    v = np.asarray(encoded, dtype=np.float32)
    power = ((np.maximum(v, 0.081) + 0.099) / 1.099) ** (1.0 / 0.45)
    return np.where(v < 0.081, v / 4.5, power).astype(np.float32)


def bt709_oetf(linear: np.ndarray) -> np.ndarray:
    v = np.asarray(linear, dtype=np.float32)
    return np.where(v < 0.018, 4.5 * v, 1.099 * np.maximum(v, 0.018) ** 0.45 - 0.099).astype(np.float32)


HLG_A = 0.17883277
HLG_B = 1.0 - 4.0 * HLG_A
HLG_C = 0.5 - HLG_A * np.log(4.0 * HLG_A)


def hlg_inverse_oetf(encoded: np.ndarray) -> np.ndarray:
    """BT.2100 HLG scene light in [0, 1] (1 = nominal peak, reference white near 0.26)."""
    v = np.clip(np.asarray(encoded, dtype=np.float32), 0.0, 1.0)
    log_part = (np.exp((np.maximum(v, 0.5) - HLG_C) / HLG_A) + HLG_B) / 12.0
    return np.where(v <= 0.5, v * v / 3.0, log_part).astype(np.float32)


PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0


def pq_eotf_nits(encoded: np.ndarray) -> np.ndarray:
    """BT.2100 PQ display light in cd/m2."""
    v = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, 1.0) ** (1.0 / PQ_M2)
    return (10000.0 * (np.maximum(v - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * v)) ** (1.0 / PQ_M1)).astype(np.float32)


def to_linear(encoded: np.ndarray, transfer: str) -> np.ndarray:
    if transfer in BT709_FAMILY:
        return bt709_inverse_oetf(encoded)
    if transfer == "iec61966-2-1":
        return colour.srgb_decode(encoded)
    if transfer == "arib-std-b67":
        return hlg_inverse_oetf(encoded)
    if transfer == "smpte2084":
        return pq_eotf_nits(encoded) / np.float32(PQ_CLIP_NITS)
    if transfer == "linear":
        return np.asarray(encoded, dtype=np.float32)
    raise ValueError(f"unsupported transfer {transfer!r}")


# ---------------------------------------------------------------- primaries


def rgb_to_xyz(primaries) -> np.ndarray:
    """Normalised primary matrix for D65 white (SMPTE RP 177)."""
    xyz = np.array([[x / y, 1.0, (1.0 - x - y) / y] for x, y in primaries]).T
    wx, wy = D65_XY
    white = np.array([wx / wy, 1.0, (1.0 - wx - wy) / wy])
    return xyz * np.linalg.solve(xyz, white)


def to_bt709_matrix(primaries: str) -> np.ndarray:
    """Linear RGB in `primaries` to linear BT.709 RGB (identity for BT.709)."""
    if primaries == "bt709":
        return np.eye(3)
    return np.linalg.inv(rgb_to_xyz(PRIMARIES["bt709"])) @ rgb_to_xyz(PRIMARIES[primaries])


def frame_to_linear(planes, signal: Signal) -> np.ndarray:
    """Decoded planes to linear BT.709 RGB, float32, 1.0 = sensor clip, unclipped."""
    linear = to_linear(planes_to_rgb(planes, signal), signal.transfer)
    if signal.primaries != "bt709":
        linear = linear @ to_bt709_matrix(signal.primaries).T.astype(np.float32)
    return np.ascontiguousarray(linear, dtype=np.float32)
