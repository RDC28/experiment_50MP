"""Colour interpretation and working-space conversion (PRD sections 7.1, 7.2 and 15).

The working space is linear-light RGB with Rec.709/sRGB primaries and a D65
white point, stored as float32. Every array carries an explicit signal label so
reports never confuse the four states PRD section 7.2 separates:

* encoded RGB code values (what the file stores),
* linear-light RGB (after removing the transfer function),
* display-referred data (an AI render or a graded image), and
* scene-referred data (calibrated scene radiance).

A rendered PNG is display-referred. Linearising it produces linear-light,
display-referred values; it never produces scene-referred radiance, so no label
in this module claims that.

Source colour is resolved from an embedded ICC profile when one exists. Only
matrix/TRC ("matrix-shaper") RGB and grey profiles are evaluated; anything else
must be overridden explicitly, because a silent guess is how colour errors slip
into a pipeline.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

WORKING_SPACE = "linear_rec709"
WORKING_SPACE_DESCRIPTION = "linear-light RGB, Rec.709/sRGB primaries, D65 white, float32"

SIGNAL_ENCODED_DISPLAY = "encoded display-referred RGB"
SIGNAL_LINEAR_DISPLAY = "linear-light display-referred RGB (not scene-referred radiance)"
SIGNAL_LINEAR_UNKNOWN = "linear-light RGB of unknown reference (treated as display-referred)"

SRGB_PRIMARIES = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060))
DISPLAY_P3_PRIMARIES = ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))
ADOBE_RGB_PRIMARIES = ((0.640, 0.330), (0.210, 0.710), (0.150, 0.060))
D65_XY = (0.3127, 0.3290)
ICC_D50_XYZ = np.array([0.9642, 1.0, 0.8249])
ADOBE_RGB_GAMMA = 563 / 256

SOURCE_COLOR_SPACES = ("auto", "srgb", "linear_srgb", "display_p3", "adobe_rgb")

_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)

# Tolerances for treating an embedded profile as plain sRGB. They absorb the
# s15Fixed16 rounding and chromatic-adaptation differences between published
# sRGB profiles (both are below one 8-bit code value).
_SRGB_MATRIX_TOLERANCE = 3e-3
_SRGB_CURVE_TOLERANCE = 2e-3


class UnsupportedColorProfile(ValueError):
    """The embedded ICC profile cannot be evaluated by the matrix-shaper path."""


# --------------------------------------------------------------------------- transfer functions


def srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB decoding (encoded code values in [0, 1] -> linear light)."""
    x = np.asarray(encoded, dtype=np.float32)
    low = x / np.float32(12.92)
    high = np.power((np.maximum(x, 0) + np.float32(0.055)) / np.float32(1.055), np.float32(2.4))
    return np.where(x <= np.float32(0.04045), low, high).astype(np.float32, copy=False)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB encoding (linear light in [0, 1] -> encoded code values)."""
    x = np.asarray(linear, dtype=np.float32)
    low = x * np.float32(12.92)
    high = np.float32(1.055) * np.power(np.maximum(x, 0), np.float32(1 / 2.4)) - np.float32(0.055)
    return np.where(x <= np.float32(0.0031308), low, high).astype(np.float32, copy=False)


@lru_cache(maxsize=1)
def srgb_uint16_to_linear_lut() -> np.ndarray:
    """Lookup table decoding every 16-bit sRGB code value to linear float32."""
    return srgb_to_linear(np.arange(65536, dtype=np.float32) / np.float32(65535))


# --------------------------------------------------------------------------- matrices


def xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def rgb_to_xyz_matrix(primaries, white_xy) -> np.ndarray:
    """Linear RGB -> CIE XYZ matrix for the given primaries and white point."""
    columns = np.stack([xy_to_xyz(p) for p in primaries], axis=1)
    scale = np.linalg.solve(columns, xy_to_xyz(white_xy))
    return columns * scale


def bradford_adaptation(source_white_xyz, target_white_xyz) -> np.ndarray:
    source_cone = _BRADFORD @ np.asarray(source_white_xyz, dtype=float)
    target_cone = _BRADFORD @ np.asarray(target_white_xyz, dtype=float)
    return np.linalg.inv(_BRADFORD) @ np.diag(target_cone / source_cone) @ _BRADFORD


XYZ_TO_LINEAR_SRGB = np.linalg.inv(rgb_to_xyz_matrix(SRGB_PRIMARIES, D65_XY))


def matrix_to_working(primaries, white_xy) -> np.ndarray:
    """Matrix from a linear RGB space to the linear Rec.709 D65 working space."""
    adaptation = bradford_adaptation(xy_to_xyz(white_xy), xy_to_xyz(D65_XY))
    return XYZ_TO_LINEAR_SRGB @ adaptation @ rgb_to_xyz_matrix(primaries, white_xy)


# --------------------------------------------------------------------------- source colour spaces


@dataclass(frozen=True)
class SourceColorSpace:
    """How source code values become working-space linear RGB."""

    name: str
    description: str
    transfer: str
    origin: str  # icc_profile | untagged_default | user_override | exr_header
    signal: str
    matrix: np.ndarray | None = field(default=None, repr=False)  # None means identity
    curves: tuple[Callable[[np.ndarray], np.ndarray], ...] | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "transfer": self.transfer,
            "origin": self.origin,
            "signal": self.signal,
            "matrix_to_working": None if self.matrix is None else np.round(self.matrix, 6).tolist(),
        }


def named_color_space(name: str, origin: str) -> SourceColorSpace:
    if name == "srgb":
        return SourceColorSpace("srgb", "sRGB IEC 61966-2-1", "sRGB piecewise", origin, SIGNAL_ENCODED_DISPLAY)
    if name == "linear_srgb":
        return SourceColorSpace(
            "linear_srgb", "linear Rec.709/sRGB primaries, D65", "linear", origin, SIGNAL_LINEAR_UNKNOWN
        )
    if name == "display_p3":
        return SourceColorSpace(
            "display_p3",
            "Display P3 (P3 primaries, D65, sRGB transfer)",
            "sRGB piecewise",
            origin,
            SIGNAL_ENCODED_DISPLAY,
            matrix=matrix_to_working(DISPLAY_P3_PRIMARIES, D65_XY),
        )
    if name == "adobe_rgb":
        return SourceColorSpace(
            "adobe_rgb",
            "Adobe RGB (1998)",
            f"gamma {ADOBE_RGB_GAMMA:.8f}",
            origin,
            SIGNAL_ENCODED_DISPLAY,
            matrix=matrix_to_working(ADOBE_RGB_PRIMARIES, D65_XY),
            curves=(_gamma_curve(ADOBE_RGB_GAMMA),) * 3,
        )
    raise ValueError(f"unknown colour space {name!r}; expected one of {SOURCE_COLOR_SPACES[1:]}")


def linear_color_space_from_chromaticities(chromaticities, origin: str = "exr_header") -> SourceColorSpace:
    """Linear RGB described by an OpenEXR ``chromaticities`` attribute (rx ry gx gy bx by wx wy)."""
    c = [float(v) for v in chromaticities]
    primaries, white = ((c[0], c[1]), (c[2], c[3]), (c[4], c[5])), (c[6], c[7])
    matrix = matrix_to_working(primaries, white)
    if np.allclose(matrix, np.eye(3), atol=_SRGB_MATRIX_TOLERANCE):
        return named_color_space("linear_srgb", origin)
    return SourceColorSpace(
        "linear_custom",
        f"linear RGB with chromaticities {tuple(round(v, 4) for v in c)}",
        "linear",
        origin,
        SIGNAL_LINEAR_UNKNOWN,
        matrix=matrix,
    )


def resolve_source_color_space(
    requested: str, icc_profile: bytes | None, *, is_linear_container: bool = False
) -> SourceColorSpace:
    """Pick the source interpretation: explicit override, embedded ICC, or a documented default."""
    if requested not in SOURCE_COLOR_SPACES:
        raise ValueError(f"source_color_space must be one of {SOURCE_COLOR_SPACES}, got {requested!r}")
    if requested != "auto":
        return named_color_space(requested, "user_override")
    if icc_profile:
        return color_space_from_icc(icc_profile)
    if is_linear_container:
        return named_color_space("linear_srgb", "exr_header")
    return named_color_space("srgb", "untagged_default")


def to_working_space(normalized: np.ndarray, color_space: SourceColorSpace) -> np.ndarray:
    """Convert normalised code values (H, W, 1|3, float32) to linear Rec.709 float32 (H, W, 3)."""
    if normalized.ndim != 3 or normalized.shape[2] not in (1, 3):
        raise ValueError(f"expected an (H, W, 1|3) array, got shape {normalized.shape}")
    if color_space.curves is not None:
        curves = color_space.curves
        if len(curves) == 1 and normalized.shape[2] == 3:
            curves = curves * 3
        linear = np.empty(normalized.shape, dtype=np.float32)
        for channel, curve in enumerate(curves[: normalized.shape[2]]):
            linear[..., channel] = curve(normalized[..., channel])
    elif color_space.transfer == "linear":
        linear = normalized.astype(np.float32, copy=True)
    else:
        linear = srgb_to_linear(normalized)
    if linear.shape[2] == 1:
        linear = np.repeat(linear, 3, axis=2)
    if color_space.matrix is not None:
        linear = np.einsum("ij,hwj->hwi", color_space.matrix.astype(np.float32), linear, optimize=True)
    return np.ascontiguousarray(linear, dtype=np.float32)


# --------------------------------------------------------------------------- encoding for export


@dataclass
class ClipStats:
    samples: int = 0
    below_zero: int = 0
    above_one: int = 0
    nonfinite: int = 0

    def to_dict(self) -> dict:
        total = max(self.samples, 1)
        return {
            "samples": self.samples,
            "below_zero": self.below_zero,
            "above_one": self.above_one,
            "nonfinite": self.nonfinite,
            "below_zero_fraction": self.below_zero / total,
            "above_one_fraction": self.above_one / total,
        }


def encode_srgb_uint16(linear: np.ndarray, chunk_rows: int = 256) -> tuple[np.ndarray, ClipStats]:
    """Encode working-space linear RGB to 16-bit sRGB code values, row block by row block.

    Values outside [0, 1] (interpolation overshoot) are clipped and counted so
    the report can show how much the integer master lost.
    """
    out = np.empty(linear.shape, dtype=np.uint16)
    stats = ClipStats(samples=int(linear.size))
    for start in range(0, linear.shape[0], chunk_rows):
        block = linear[start : start + chunk_rows]
        finite = np.isfinite(block)
        stats.nonfinite += int(block.size - np.count_nonzero(finite))
        stats.below_zero += int(np.count_nonzero(block < 0))
        stats.above_one += int(np.count_nonzero(block > 1))
        clean = np.clip(np.where(finite, block, 0), 0, 1)
        out[start : start + chunk_rows] = np.rint(linear_to_srgb(clean) * np.float32(65535)).astype(np.uint16)
    return out, stats


def srgb_uint16_to_linear(encoded: np.ndarray) -> np.ndarray:
    return srgb_uint16_to_linear_lut()[encoded]


def srgb_uint16_to_uint8(encoded: np.ndarray) -> np.ndarray:
    return ((encoded.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)


@lru_cache(maxsize=1)
def srgb_icc_profile() -> bytes:
    """The LittleCMS built-in sRGB ICC profile, embedded in every display-referred export."""
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# --------------------------------------------------------------------------- ICC parsing


@dataclass(frozen=True)
class IccProfileInfo:
    description: str
    version: str
    device_class: str
    color_space: str
    pcs: str
    colorants: np.ndarray | None  # columns are the rXYZ, gXYZ, bXYZ tags (PCS D50)
    curves: tuple[Callable[[np.ndarray], np.ndarray], ...] | None
    curve_types: tuple[str, ...]


def parse_icc_profile(data: bytes) -> IccProfileInfo:
    if len(data) < 132 or data[36:40] != b"acsp":
        raise UnsupportedColorProfile("embedded ICC profile is truncated or lacks the 'acsp' signature")
    version = f"{data[8]}.{data[9] >> 4}.{data[9] & 0xF}"
    device_class = data[12:16].decode("latin-1").strip()
    color_space = data[16:20].decode("latin-1").strip()
    pcs = data[20:24].decode("latin-1").strip()
    (count,) = struct.unpack(">I", data[128:132])
    tags = {}
    for index in range(count):
        entry = 132 + 12 * index
        if entry + 12 > len(data):
            raise UnsupportedColorProfile("ICC tag table runs past the end of the profile")
        signature, offset, size = struct.unpack(">4sII", data[entry : entry + 12])
        tags[signature.decode("latin-1")] = data[offset : offset + size]

    description = _icc_text(tags.get("desc")) or "(no description)"
    colorants = curves = None
    curve_types: tuple[str, ...] = ()
    if color_space == "RGB" and all(t in tags for t in ("rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC")):
        colorants = np.stack([_icc_xyz(tags[t]) for t in ("rXYZ", "gXYZ", "bXYZ")], axis=1)
        parsed = [_icc_curve(tags[t]) for t in ("rTRC", "gTRC", "bTRC")]
        curves = tuple(p[0] for p in parsed)
        curve_types = tuple(p[1] for p in parsed)
    elif color_space == "GRAY" and "kTRC" in tags:
        curve, kind = _icc_curve(tags["kTRC"])
        curves, curve_types = (curve,), (kind,)
    return IccProfileInfo(description, version, device_class, color_space, pcs, colorants, curves, curve_types)


def color_space_from_icc(data: bytes) -> SourceColorSpace:
    info = parse_icc_profile(data)
    if info.pcs != "XYZ" or info.curves is None:
        raise UnsupportedColorProfile(
            f"ICC profile {info.description!r} ({info.color_space}, PCS {info.pcs}) is not a matrix/TRC "
            "RGB or grey profile; set source_color_space explicitly to override it"
        )
    if info.color_space == "GRAY":
        return SourceColorSpace(
            f"icc:{info.description}",
            f"ICC grey profile {info.description!r}",
            f"ICC {info.curve_types[0]}",
            "icc_profile",
            SIGNAL_ENCODED_DISPLAY,
            curves=info.curves,
        )
    matrix = XYZ_TO_LINEAR_SRGB @ bradford_adaptation(ICC_D50_XYZ, xy_to_xyz(D65_XY)) @ info.colorants
    probe = np.linspace(0, 1, 4097, dtype=np.float32)
    srgb_curve = srgb_to_linear(probe)
    curves_match = all(np.max(np.abs(c(probe) - srgb_curve)) <= _SRGB_CURVE_TOLERANCE for c in info.curves)
    if curves_match and np.allclose(matrix, np.eye(3), atol=_SRGB_MATRIX_TOLERANCE):
        space = named_color_space("srgb", "icc_profile")
        return SourceColorSpace(
            space.name, f"sRGB (embedded ICC {info.description!r})", space.transfer, "icc_profile", space.signal
        )
    return SourceColorSpace(
        f"icc:{info.description}",
        f"ICC matrix/TRC profile {info.description!r} (v{info.version})",
        "ICC " + "/".join(info.curve_types),
        "icc_profile",
        SIGNAL_ENCODED_DISPLAY,
        matrix=matrix,
        curves=info.curves,
    )


def _icc_text(tag: bytes | None) -> str | None:
    if not tag or len(tag) < 12:
        return None
    kind = tag[:4]
    if kind == b"desc":
        (length,) = struct.unpack(">I", tag[8:12])
        return tag[12 : 12 + length].split(b"\0", 1)[0].decode("latin-1").strip() or None
    if kind == b"mluc" and len(tag) >= 28:
        length, offset = struct.unpack(">II", tag[20:28])
        return tag[offset : offset + length].decode("utf-16-be", errors="replace").strip("\0 ") or None
    if kind == b"text":
        return tag[8:].split(b"\0", 1)[0].decode("latin-1").strip() or None
    return None


def _icc_xyz(tag: bytes) -> np.ndarray:
    if tag[:4] != b"XYZ " or len(tag) < 20:
        raise UnsupportedColorProfile("malformed ICC XYZ tag")
    return np.array(struct.unpack(">3i", tag[8:20]), dtype=float) / 65536.0


def _icc_curve(tag: bytes) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    kind = tag[:4]
    if kind == b"curv":
        (count,) = struct.unpack(">I", tag[8:12])
        if count == 0:
            return (lambda x: np.asarray(x, dtype=np.float32)), "identity"
        if count == 1:
            (raw,) = struct.unpack(">H", tag[12:14])
            gamma = raw / 256.0
            return _gamma_curve(gamma), f"gamma {gamma:g}"
        table = np.frombuffer(tag[12 : 12 + 2 * count], dtype=">u2").astype(np.float32) / 65535.0
        grid = np.linspace(0, 1, count, dtype=np.float32)
        return (lambda x: np.interp(np.clip(x, 0, 1), grid, table).astype(np.float32)), f"table[{count}]"
    if kind == b"para":
        (function,) = struct.unpack(">H", tag[8:10])
        n_params = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(function)
        if n_params is None:
            raise UnsupportedColorProfile(f"unknown ICC parametric curve type {function}")
        params = np.array(struct.unpack(f">{n_params}i", tag[12 : 12 + 4 * n_params]), dtype=float) / 65536.0
        return _parametric_curve(function, params), f"parametric type {function}"
    raise UnsupportedColorProfile(f"unsupported ICC curve type {kind!r}")


def _gamma_curve(gamma: float) -> Callable[[np.ndarray], np.ndarray]:
    def curve(x: np.ndarray) -> np.ndarray:
        return np.power(np.clip(x, 0, 1), np.float32(gamma)).astype(np.float32)

    return curve


def _parametric_curve(function: int, p: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """ICC parametric curve, written in the general form Y = (aX+b)^g + e for X >= d, else cX + f."""
    g, a, b, c, d, e, f = p[0], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if function in (1, 2):
        # Y = (aX+b)^g [+ c] above X = -b/a, constant (0 [or c]) below it.
        a, b = p[1], p[2]
        offset = p[3] if function == 2 else 0.0
        d = -b / a if a else 0.0
        e = f = offset
    elif function == 3:
        a, b, c, d = p[1:5]
    elif function == 4:
        a, b, c, d, e, f = p[1:7]

    def curve(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), 0, 1)
        high = np.power(np.maximum(a * x + b, 0), g) + e
        low = c * x + f
        return np.where(x >= d, high, low).astype(np.float32)

    return curve


def icc_description(data: bytes | None) -> str | None:
    if not data:
        return None
    try:
        return parse_icc_profile(data).description
    except UnsupportedColorProfile:
        return None
