"""Exact sRGB transfer, the LibRaw-matching camera transform and the illuminant prior (sections 5.2, 5.3).

LibRaw renders a raw to linear sRGB in two steps: white balance multipliers normalised so the smallest
is 1, then `rgb_cam`, the inverse of the row-normalised `cam_xyz @ xyz_rgb`. The functions here are
the exact inverse of that rendering, so an unclipped render inverts to the raw values up to the
illuminant and the exposure placement.
"""

from __future__ import annotations

import numpy as np

from prd_v6 import constants

# LibRaw's xyz_rgb: linear sRGB (D65) to XYZ.
SRGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """Exact piecewise sRGB EOTF (IEC 61966-2-1) on [0, 1] floats, float32 out."""
    v = np.asarray(encoded, dtype=np.float32)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Exact inverse of `srgb_decode`; input is clipped to [0, 1] first."""
    v = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055).astype(np.float32)


def to_uint8(encoded: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(encoded, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def to_uint16(encoded: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(encoded, dtype=np.float32) * 65535.0), 0, 65535).astype(np.uint16)


def camera_from_srgb(cam_xyz: np.ndarray) -> np.ndarray:
    """LibRaw's row-normalised cam_rgb: linear sRGB to white-balanced camera RGB."""
    cam_rgb = np.asarray(cam_xyz, dtype=np.float64) @ SRGB_TO_XYZ
    return cam_rgb / cam_rgb.sum(axis=1, keepdims=True)


def normalise_neutral(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)[:3]
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"invalid neutral {values}")
    return values / values[1]


def daylight_neutral(cam_xyz: np.ndarray) -> np.ndarray:
    """Camera response to the sRGB (D65) white, normalised to G."""
    return normalise_neutral(np.asarray(cam_xyz, dtype=np.float64) @ SRGB_TO_XYZ @ np.ones(3))


def planckian_xy(cct_k: float) -> tuple[float, float]:
    """CIE 1931 xy of a blackbody (Kim et al. 2002 cubic spline, 1667 K to 25000 K)."""
    t = float(cct_k)
    if not 1667.0 <= t <= 25000.0:
        raise ValueError("CCT outside 1667-25000 K")
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def neutral_for_cct(cct_k: float, cam_xyz: np.ndarray) -> np.ndarray:
    x, y = planckian_xy(cct_k)
    xyz = np.array([x / y, 1.0, (1.0 - x - y) / y])
    return normalise_neutral(np.asarray(cam_xyz, dtype=np.float64) @ xyz)


def prior_neutral(cam_xyz: np.ndarray) -> np.ndarray:
    """Section 5.3: the camera neutral at the mired midpoint of 2850 K to 7500 K."""
    return neutral_for_cct(constants.PRIOR_CCT_K, cam_xyz)


def neutral_from_multipliers(multipliers) -> np.ndarray:
    """LibRaw's as-shot multipliers (R, G, B, G2) are reciprocal to the camera neutral."""
    values = np.asarray(multipliers, dtype=np.float64)[:3]
    return normalise_neutral(1.0 / values)


def balance_gains(neutral) -> np.ndarray:
    """Per-channel factor from white-balanced camera RGB to raw signal: n_c / max(n).

    This is the reciprocal of LibRaw's pre_mul after its normalisation to a smallest multiplier of 1.
    """
    neutral = normalise_neutral(neutral)
    return neutral / neutral.max()


def linear_srgb_to_camera(linear: np.ndarray, neutral, cam_xyz: np.ndarray, white: float) -> np.ndarray:
    """Linear sRGB (render scale, 1.0 = render white) to black-subtracted camera DN, float32.

    `white` is the raw signal that maps to render white in the smallest-multiplier channel. Values are
    not clipped here; negative and over-white values are clipped by the caller's output stage.
    """
    matrix = camera_from_srgb(cam_xyz) * (balance_gains(neutral) * white)[:, None]
    out = np.asarray(linear, dtype=np.float32) @ matrix.T.astype(np.float32)
    return out.astype(np.float32, copy=False)


def camera_to_linear_srgb(camera: np.ndarray, neutral, cam_xyz: np.ndarray, white: float) -> np.ndarray:
    """Exact inverse of `linear_srgb_to_camera` (used to develop the 2K and 4K outputs)."""
    matrix = camera_from_srgb(cam_xyz) * (balance_gains(neutral) * white)[:, None]
    inverse = np.linalg.inv(matrix)
    return (np.asarray(camera, dtype=np.float32) @ inverse.T.astype(np.float32)).astype(np.float32, copy=False)


def filmic(linear: np.ndarray) -> np.ndarray:
    """Narkowicz ACES filmic fit, the R2 stress render's tone curve (section 6.1)."""
    x = np.maximum(np.asarray(linear, dtype=np.float32), 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0).astype(np.float32)


def srgb_icc_profile() -> bytes:
    """An sRGB ICC profile for the renders and developed TIFFs."""
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
