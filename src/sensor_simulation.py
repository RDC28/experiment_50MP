"""Synthetic sensor-domain simulation and its inverse (PRD sections 14, 17.3 and experiment E08).

The forward path turns processed linear RGB (the working space: linear Rec.709,
D65) into a synthetic colour-filter-array mosaic of integer code values:

1. assumed camera colour transformation (``color_matrix`` and
   ``white_balance_multipliers``, both stated assumptions, not measurements),
2. synthetic exposure scaling,
3. synthetic CFA (Bayer) sampling,
4. optional shot and read noise from a stated hypothetical sensor model,
5. quantisation to the configured ADC bit depth with a black level offset.

The inverse path (:func:`develop`) normalises the code values, demosaics them
and maps camera RGB back to the working space, so the round trip can be
measured against the input with the Phase 1 metrics.

PRD 14.3: converting RGB into a Bayer-like mosaic simulates sensor sampling. It
does not reconstruct the samples a real camera would have measured, and
simulated shot noise is not evidence of a physical exposure. Every mosaic this
module returns carries :data:`SYNTHETIC_RAW_LABEL` and an information
provenance record that says so.

Hypothetical sensor model (illustrative, not any real product): a linear
photodetector with ``full_well_electrons`` at base gain, Poisson shot noise,
signal-independent Gaussian read noise of ``read_noise_std`` electrons (RMS,
input-referred) and an ideal ADC whose range from ``black_level`` to
``white_level`` spans the full well at base gain. ``analog_gain`` simulates a
higher ISO: the exposure collects ``1 / analog_gain`` as many electrons and the
gain restores the code values, so the image brightness is unchanged and the
relative shot noise grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import color_management as cm

SYNTHETIC_RAW_LABEL = (
    "SYNTHETIC MOSAIC: simulated CFA sampling of processed RGB through an assumed camera model. "
    "Not camera RAW and not sensor measurements (PRD 14.3)."
)

# Channel index (0 = R, 1 = G, 2 = B) at each position of the 2 x 2 CFA tile, rows first.
CFA_PATTERNS = {
    "RGGB": ((0, 1), (1, 2)),
    "BGGR": ((2, 1), (1, 0)),
    "GRBG": ((1, 0), (2, 1)),
    "GBRG": ((1, 2), (0, 1)),
}
ADC_BIT_DEPTHS = (10, 12, 14, 16)
DEMOSAIC_METHODS = ("bilinear", "malvar")
DEMOSAIC_DESCRIPTIONS = {
    "bilinear": "bilinear interpolation of each colour plane (3x3 kernels)",
    "malvar": "Malvar-He-Cutler gradient-corrected linear interpolation (5x5 kernels, ICASSP 2004)",
}

# Illustrative camera-to-reference matrix: maps white-balanced camera RGB to linear Rec.709. It is the
# inverse of an assumed spectral crosstalk ([0.70 0.24 0.06], [0.10 0.78 0.12], [0.04 0.28 0.68]),
# rounded so that every row sums to one (white-balanced neutrals stay neutral). It is not taken from,
# and does not describe, any real camera.
DEFAULT_COLOR_MATRIX = (
    (1.4946, -0.4404, -0.0542),
    (-0.1901, 1.4248, -0.2347),
    (-0.0096, -0.5608, 1.5704),
)
DEFAULT_WHITE_BALANCE = (2.0, 1.0, 1.6)  # R, G, B gains that neutralise the assumed D65 illuminant

_CHUNK_ROWS = 512
_DEMOSAIC_BLOCK_ROWS = 1024
_DEMOSAIC_HALO = 2  # even, so every block starts on the same CFA phase; covers the 5x5 kernels


@dataclass(frozen=True)
class SensorModel:
    """PRD 14.4 synthetic sensor parameters, validated on construction."""

    cfa_pattern: str = "RGGB"
    adc_bit_depth: int = 14
    black_level: int | None = None  # None: 2 ** (adc_bit_depth - 5), e.g. 512 at 14 bits
    white_level: int | None = None  # None: 2 ** adc_bit_depth - 1, the largest ADC code
    analog_gain: float = 1.0
    exposure_scale: float = 0.8
    full_well_electrons: float = 45_000.0
    shot_noise_enabled: bool = False
    read_noise_enabled: bool = False
    read_noise_std: float = 2.5
    color_matrix: tuple = DEFAULT_COLOR_MATRIX
    white_balance_multipliers: tuple = DEFAULT_WHITE_BALANCE
    synthetic_raw_label: str = SYNTHETIC_RAW_LABEL

    def __post_init__(self) -> None:
        if self.cfa_pattern not in CFA_PATTERNS:
            raise ValueError(f"cfa_pattern must be one of {tuple(CFA_PATTERNS)}, got {self.cfa_pattern!r}")
        if self.adc_bit_depth not in ADC_BIT_DEPTHS:
            raise ValueError(f"adc_bit_depth must be one of {ADC_BIT_DEPTHS}, got {self.adc_bit_depth!r}")
        if not 0 <= self.black < self.white <= 2**self.adc_bit_depth - 1:
            raise ValueError(
                f"need 0 <= black_level < white_level <= {2**self.adc_bit_depth - 1}, "
                f"got black {self.black}, white {self.white}"
            )
        if self.white - self.black < 255:
            raise ValueError("the range between black_level and white_level must hold at least 256 codes")
        if not 1.0 <= self.analog_gain <= 64.0:
            raise ValueError(f"analog_gain must be between 1 and 64, got {self.analog_gain}")
        if not 0.01 <= self.exposure_scale <= 1.0:
            raise ValueError(f"exposure_scale must be between 0.01 and 1, got {self.exposure_scale}")
        if not 100.0 <= self.full_well_electrons <= 1e7:
            raise ValueError(f"full_well_electrons must be between 100 and 1e7, got {self.full_well_electrons}")
        if not 0.0 <= self.read_noise_std <= 1000.0:
            raise ValueError(f"read_noise_std must be between 0 and 1000 electrons, got {self.read_noise_std}")
        matrix = np.asarray(self.color_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("color_matrix must be a finite 3 x 3 matrix")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError("every color_matrix row must sum to 1 so white-balanced neutrals stay neutral")
        if abs(np.linalg.det(matrix)) < 1e-6:
            raise ValueError("color_matrix must be invertible")
        balance = np.asarray(self.white_balance_multipliers, dtype=np.float64)
        if balance.shape != (3,) or not np.all((balance >= 0.1) & (balance <= 10)):
            raise ValueError("white_balance_multipliers must be three gains between 0.1 and 10")
        if "synthetic" not in self.synthetic_raw_label.lower():
            raise ValueError("synthetic_raw_label is a required provenance marker and must say 'synthetic'")

    @property
    def black(self) -> int:
        return 2 ** (self.adc_bit_depth - 5) if self.black_level is None else int(self.black_level)

    @property
    def white(self) -> int:
        return 2**self.adc_bit_depth - 1 if self.white_level is None else int(self.white_level)

    @property
    def channel_map(self) -> np.ndarray:
        """(2, 2) channel index of each CFA tile position."""
        return np.array(CFA_PATTERNS[self.cfa_pattern], dtype=np.int64)

    @property
    def camera_from_working(self) -> np.ndarray:
        """Linear Rec.709 -> camera-native RGB (before white balance): diag(1 / wb) @ inv(color_matrix)."""
        balance = np.asarray(self.white_balance_multipliers, dtype=np.float64)
        return np.diag(1.0 / balance) @ np.linalg.inv(np.asarray(self.color_matrix, dtype=np.float64))

    @property
    def dn_per_electron(self) -> float:
        """Conversion gain in code values per electron at the configured analog gain."""
        return (self.white - self.black) * self.analog_gain / self.full_well_electrons

    def as_shot_neutral(self) -> np.ndarray:
        """Camera-native values of a neutral (white) surface, normalised to a maximum of 1 (DNG AsShotNeutral)."""
        neutral = 1.0 / np.asarray(self.white_balance_multipliers, dtype=np.float64)
        return neutral / neutral.max()

    def xyz_to_camera(self) -> np.ndarray:
        """DNG ColorMatrix1: CIE XYZ -> camera native under D65, scaled so the D65 white maps to max 1."""
        matrix = self.camera_from_working @ cm.XYZ_TO_LINEAR_SRGB
        return matrix / (matrix @ cm.xy_to_xyz(cm.D65_XY)).max()

    def noise_description(self) -> str:
        parts = []
        if self.shot_noise_enabled:
            parts.append(f"Poisson shot noise (full well {self.full_well_electrons:,.0f} e- at gain 1)")
        if self.read_noise_enabled:
            parts.append(f"Gaussian read noise {self.read_noise_std:g} e- RMS")
        return ", ".join(parts) or "none (noise-free quantisation)"

    def to_dict(self) -> dict:
        return {
            "cfa_pattern": self.cfa_pattern,
            "adc_bit_depth": self.adc_bit_depth,
            "black_level": self.black,
            "white_level": self.white,
            "analog_gain": self.analog_gain,
            "exposure_scale": self.exposure_scale,
            "full_well_electrons": self.full_well_electrons,
            "shot_noise_enabled": self.shot_noise_enabled,
            "read_noise_enabled": self.read_noise_enabled,
            "read_noise_std_electrons": self.read_noise_std,
            "dn_per_electron": round(self.dn_per_electron, 6),
            "color_matrix": [list(row) for row in self.color_matrix],
            "white_balance_multipliers": list(self.white_balance_multipliers),
            "as_shot_neutral": np.round(self.as_shot_neutral(), 6).tolist(),
            "xyz_to_camera_d65": np.round(self.xyz_to_camera(), 6).tolist(),
            "synthetic_raw_label": self.synthetic_raw_label,
            "sensor_model": "hypothetical linear sensor (illustrative parameters, not a real camera)",
        }


@dataclass
class SyntheticMosaic:
    """Integer CFA code values plus everything needed to interpret them; always labelled synthetic."""

    dn: np.ndarray  # (H, W) uint16
    model: SensorModel
    seed: int
    stats: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.model.synthetic_raw_label

    @property
    def height(self) -> int:
        return self.dn.shape[0]

    @property
    def width(self) -> int:
        return self.dn.shape[1]

    def provenance(self) -> dict:
        return {
            "synthetic_raw_label": self.label,
            "classification": "Simulation (PRD 5.2)",
            "information": mosaic_information(self.model),
            "sensor": self.model.to_dict(),
            "random_seed": self.seed,
            "statistics": self.stats,
        }


# --------------------------------------------------------------------------- forward simulation


def cfa_masks(height: int, width: int, pattern: str) -> np.ndarray:
    """(3, H, W) boolean masks: which pixels sample R, G and B."""
    channel_map = np.array(CFA_PATTERNS[pattern])
    rows = np.arange(height)[:, None] % 2
    cols = np.arange(width)[None, :] % 2
    sites = channel_map[rows, cols]
    return np.stack([sites == c for c in range(3)])


def sample_cfa(linear: np.ndarray, model: SensorModel) -> np.ndarray:
    """Assumed camera colour transform, exposure scaling and CFA sampling.

    Returns the (H, W) float32 signal of each photosite as a fraction of the
    ADC range (1.0 = white level). Only the one channel each site samples is
    computed, so no full-size camera RGB image is held in memory.
    """
    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) linear RGB array, got {linear.shape}")
    rows = (model.camera_from_working * model.exposure_scale).astype(np.float32)
    signal = np.empty(linear.shape[:2], dtype=np.float32)
    for sy in range(2):
        for sx in range(2):
            channel = model.channel_map[sy, sx]
            signal[sy::2, sx::2] = linear[sy::2, sx::2] @ rows[channel]
    return signal


def quantize(signal: np.ndarray, model: SensorModel, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Optional shot and read noise, then ADC quantisation to uint16 code values (row block by row block).

    Shot and read noise draw from independent generators spawned from ``seed``,
    so results are reproducible and do not depend on the block size.
    """
    shot_rng, read_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    black, white = model.black, model.white
    electrons_per_unit = model.full_well_electrons / model.analog_gain
    dn = np.empty(signal.shape, dtype=np.uint16)
    counts = {"negative_signal": 0, "saturated_signal": 0, "clipped_at_white": 0, "below_black": 0, "at_zero": 0}
    for start in range(0, signal.shape[0], _CHUNK_ROWS):
        block = signal[start : start + _CHUNK_ROWS].astype(np.float64)
        counts["negative_signal"] += int(np.count_nonzero(block < 0))
        counts["saturated_signal"] += int(np.count_nonzero(block >= 1))
        # No negative light reaches a photosite; anything above the white level saturates below.
        electrons = np.clip(block, 0, None) * electrons_per_unit
        if model.shot_noise_enabled:
            electrons = shot_rng.poisson(electrons).astype(np.float64)
        if model.read_noise_enabled:
            electrons += read_rng.normal(0.0, model.read_noise_std, size=electrons.shape)
        codes = np.rint(black + electrons * model.dn_per_electron)
        counts["clipped_at_white"] += int(np.count_nonzero(codes >= white))
        counts["below_black"] += int(np.count_nonzero(codes < black))
        counts["at_zero"] += int(np.count_nonzero(codes <= 0))
        dn[start : start + _CHUNK_ROWS] = np.clip(codes, 0, white).astype(np.uint16)
    total = max(signal.size, 1)
    stats = {f"{name}_fraction": value / total for name, value in counts.items()}
    stats["samples"] = int(signal.size)
    return dn, stats


def simulate(linear: np.ndarray, model: SensorModel, seed: int = 42) -> SyntheticMosaic:
    """Full forward path: processed linear RGB -> labelled synthetic mosaic (PRD 14.2)."""
    signal = sample_cfa(linear, model)
    dn, stats = quantize(signal, model, seed)
    return SyntheticMosaic(dn, model, seed, stats)


def expected_code_statistics(signal: float, model: SensorModel) -> tuple[float, float]:
    """Model mean and variance of the code value for a constant signal (fraction of the ADC range).

    Variance = conversion_gain^2 * (shot electrons + read_noise^2) + 1/12 (uniform quantisation error),
    valid away from the black floor and the white clip.
    """
    electrons = max(signal, 0.0) * model.full_well_electrons / model.analog_gain
    variance = 1.0 / 12.0
    if model.shot_noise_enabled:
        variance += model.dn_per_electron**2 * electrons
    if model.read_noise_enabled:
        variance += model.dn_per_electron**2 * model.read_noise_std**2
    return model.black + electrons * model.dn_per_electron, variance


# --------------------------------------------------------------------------- inverse path


def normalize(dn: np.ndarray, model: SensorModel) -> np.ndarray:
    """Code values -> camera-native signal relative to the working-space white (undoes black, range, exposure)."""
    scale = np.float32(1.0 / ((model.white - model.black) * model.exposure_scale))
    return (dn.astype(np.float32) - np.float32(model.black)) * scale


def demosaic(native: np.ndarray, pattern: str, method: str = "malvar") -> np.ndarray:
    """Interpolate an (H, W) float32 CFA image to (H, W, 3) float32, in row blocks to bound memory."""
    if method not in DEMOSAIC_METHODS:
        raise ValueError(f"demosaic method must be one of {DEMOSAIC_METHODS}, got {method!r}")
    if native.ndim != 2:
        raise ValueError(f"expected an (H, W) mosaic, got {native.shape}")
    channel_map = np.array(CFA_PATTERNS[pattern])
    native = np.asarray(native, dtype=np.float32)
    height = native.shape[0]
    out = np.empty((*native.shape, 3), dtype=np.float32)
    for start in range(0, height, _DEMOSAIC_BLOCK_ROWS):
        stop = min(start + _DEMOSAIC_BLOCK_ROWS, height)
        top = max(start - _DEMOSAIC_HALO, 0)
        bottom = min(stop + _DEMOSAIC_HALO, height)
        interpolate = _bilinear if method == "bilinear" else _malvar
        block = interpolate(native[top:bottom], channel_map)
        out[start:stop] = block[start - top : start - top + (stop - start)]
    return out


def develop(dn: np.ndarray, model: SensorModel, method: str = "malvar") -> np.ndarray:
    """Synthetic mosaic -> working-space linear RGB: normalise, demosaic, white balance, colour matrix."""
    rgb = demosaic(normalize(dn, model), model.cfa_pattern, method)
    to_working = (np.asarray(model.color_matrix) @ np.diag(model.white_balance_multipliers)).astype(np.float32)
    for start in range(0, rgb.shape[0], _CHUNK_ROWS):
        rows = rgb[start : start + _CHUNK_ROWS]
        rows[...] = rows @ to_working.T
    return rgb


# --------------------------------------------------------------------------- provenance


def mosaic_information(model: SensorModel) -> dict:
    """PRD 1.1 information categories for a synthetic mosaic."""
    noise = model.noise_description()
    return {
        "preserved": "one linear combination of the input RGB per photosite (the assumed camera channel it samples)",
        "estimated": "none",
        "synthesized": "none (no learned model)",
        "simulated": (
            f"assumed camera colour response and white balance, {model.cfa_pattern} CFA sampling, "
            f"exposure scaling, noise: {noise}, {model.adc_bit_depth}-bit ADC quantisation"
        ),
        "export_only": "container and metadata (DNG or TIFF tags, black and white levels, colour matrix)",
    }


def developed_information(model: SensorModel, demosaic_description: str) -> dict:
    """PRD 1.1 information categories for an RGB image developed back from a synthetic mosaic."""
    return {
        "preserved": "one sampled channel per pixel, after the simulated quantisation and noise",
        "estimated": f"the two unsampled channels per pixel, by {demosaic_description}",
        "synthesized": "none (no learned model)",
        "simulated": mosaic_information(model)["simulated"],
        "export_only": "none",
    }


# --------------------------------------------------------------------------- demosaic kernels

_BILINEAR_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4
_BILINEAR_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4
_MALVAR_G_AT_RB = (
    np.array(
        [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype=np.float32
    )
    / 8
)
_MALVAR_ROW = (  # colour of the horizontal neighbours, at a green site
    np.array(
        [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]],
        dtype=np.float32,
    )
    / 8
)
_MALVAR_COL = np.ascontiguousarray(_MALVAR_ROW.T)  # colour of the vertical neighbours, at a green site
_MALVAR_DIAGONAL = (  # red at blue sites and blue at red sites
    np.array(
        [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]],
        dtype=np.float32,
    )
    / 8
)


def _filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Kernels are point-symmetric, so correlation equals convolution. Reflect-101 mirrors about the edge
    # pixel, which keeps the CFA phase (pixel -1 maps to pixel 1, the same colour).
    return cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)


def _site_grid(shape: tuple[int, int], channel_map: np.ndarray) -> np.ndarray:
    rows = np.arange(shape[0])[:, None] % 2
    cols = np.arange(shape[1])[None, :] % 2
    return channel_map[rows, cols]


def _bilinear(native: np.ndarray, channel_map: np.ndarray) -> np.ndarray:
    sites = _site_grid(native.shape, channel_map)
    out = np.empty((*native.shape, 3), dtype=np.float32)
    for channel in range(3):
        masked = np.where(sites == channel, native, np.float32(0))
        out[..., channel] = _filter(masked, _BILINEAR_G if channel == 1 else _BILINEAR_RB)
    return out


def _malvar(native: np.ndarray, channel_map: np.ndarray) -> np.ndarray:
    out = np.repeat(native[..., None], 3, axis=2)
    g_at_rb = _filter(native, _MALVAR_G_AT_RB)
    row = _filter(native, _MALVAR_ROW)
    col = _filter(native, _MALVAR_COL)
    diagonal = _filter(native, _MALVAR_DIAGONAL)
    for sy in range(2):
        for sx in range(2):
            site = (slice(sy, None, 2), slice(sx, None, 2))
            channel = channel_map[sy, sx]
            if channel == 1:
                horizontal = channel_map[sy, 1 - sx]
                out[(*site, horizontal)] = row[site]
                out[(*site, 2 - horizontal)] = col[site]
            else:
                out[(*site, 1)] = g_at_rb[site]
                out[(*site, 2 - channel)] = diagonal[site]
    return out
