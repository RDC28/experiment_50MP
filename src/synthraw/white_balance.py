"""Approximate white-balance hypotheses for PRD_v3's camera milestone.

Inputs are scene-linear sRGB, not encoded pixels or camera-native samples.
Single-image illuminant estimates assume neutral scene content; confidence is a
heuristic agreement score, not a calibrated probability of the true illuminant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthraw.camera_colour import CameraColourModel
from synthraw.camera_profile import CameraProfile
from synthraw.provenance import FieldRecord, Provenance


@dataclass
class WhiteBalanceResult:
    """Illuminant metadata and green-normalised camera-channel gains for the inverse ISP.

    AsShotNeutral in ``records`` is the camera-native neutral the DNG carries; dividing the
    camera response of a unit linear-sRGB grey by ``gains`` reproduces it.
    """

    records: dict[str, FieldRecord]
    gains: tuple[float, float, float]


def _illuminant_rgb(temperature: np.ndarray) -> np.ndarray:
    """Approximate Planckian-locus xy (1667-25000 K), converted to linear sRGB."""
    t = np.asarray(temperature, dtype=float)
    x = np.where(
        t <= 4000,
        -0.2661239e9 / t**3 - 0.2343580e6 / t**2 + 0.8776956e3 / t + 0.179910,
        -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390,
    )
    y = np.where(
        t <= 4000,
        -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867,
        3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483,
    )
    xyz = np.stack((x / y, np.ones_like(t), (1 - x - y) / y), axis=-1)
    rgb = (
        xyz
        @ np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        ).T
    )
    return rgb / rgb[..., 1:2]


class WhiteBalanceEstimator:
    """Infer neutral coordinates or record an explicit user-selected hypothesis."""

    def estimate(
        self,
        scene_linear_rgb: np.ndarray | None,
        profile: CameraProfile,
        *,
        method: str = "COMBINED",
        manual_cct: float | None = None,
        manual_tint: float = 0.0,
        colour_model: CameraColourModel | None = None,
    ) -> WhiteBalanceResult:
        """Return CCT, log-green tint, neutrals and EXIF LightSource candidate.

        Tint is log(G/sqrt(R*B)) relative to the fitted Planckian illuminant.
        Manual settings are fabricated hypotheses with user_override recorded.
        Inferred CCT is clamped to the profile envelope; manual CCT must fit it.
        ``illuminant_neutral_srgb`` is the illuminant colour in linear sRGB; ``AsShotNeutral`` is
        its camera-native response through ``colour_model`` (identity camera when omitted).
        """
        low, high = profile.camera["colour_temperature_range"].value
        if not 1667 <= low < high <= 25000:
            raise ValueError("Profile temperature range must lie within 1667-25000 K")
        if method not in ("GREY_WORLD", "WHITE_PATCH", "COMBINED"):
            raise ValueError("Unknown white-balance method")
        if not np.isfinite(manual_tint) or abs(manual_tint) > 1:
            raise ValueError("manual_tint must be finite and between -1 and 1")
        manual = manual_cct is not None
        if manual:
            if not np.isfinite(manual_cct) or not low <= manual_cct <= high:
                raise ValueError("Manual CCT must lie within the profile temperature range")
            cct = float(manual_cct)
            neutral = _illuminant_rgb(np.array(cct))
            neutral[[0, 2]] *= np.exp(-manual_tint)
            tint = float(manual_tint)
            confidence = None
            description = "User-selected white balance; linear-sRGB Planckian approximation"
        else:
            if manual_tint != 0:
                raise ValueError("manual_tint requires manual_cct")
            rgb = np.asarray(scene_linear_rgb, dtype=float)
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or not rgb.size:
                raise ValueError("Expected a nonempty HxWx3 scene-linear sRGB image")
            if not np.isfinite(rgb).all() or np.any(rgb < 0):
                raise ValueError("Scene-linear RGB must be finite and nonnegative")
            pixels = rgb.reshape(-1, 3)
            # Exclude black pixels and unusable channel statistics, never invent gains.
            pixels = pixels[np.max(pixels, axis=1) > 0]
            if not len(pixels):
                raise ValueError("No illuminated pixels for white-balance inference")
            grey = pixels.mean(axis=0)
            # Select bright pixels jointly rather than unrelated per-channel maxima.
            brightness = pixels @ np.array([0.2126, 0.7152, 0.0722])
            patch = pixels[brightness >= np.quantile(brightness, 0.95)].mean(axis=0)
            if np.any(grey <= 0) or np.any(patch <= 0):
                raise ValueError("Insufficient channel support for white-balance inference")
            grey, patch = grey / grey[1], patch / patch[1]
            neutral = {"GREY_WORLD": grey, "WHITE_PATCH": patch, "COMBINED": np.sqrt(grey * patch)}[method]
            temperatures = np.linspace(low, high, 2049)
            locus = _illuminant_rgb(temperatures)
            # R/B identifies temperature independently of the green tint parameter.
            index = np.argmin(np.abs(np.log(locus[:, 0] / locus[:, 2]) - np.log(neutral[0] / neutral[2])))
            cct = float(temperatures[index])
            tint = float(0.5 * np.log(np.prod(locus[index, [0, 2]]) / np.prod(neutral[[0, 2]])))
            agreement = np.exp(-np.linalg.norm(np.log(grey / patch)))
            confidence = float(0.5 * agreement)  # Neutral-content ambiguity remains even at perfect agreement.
            description = f"{method} neutral-content hypothesis in linear sRGB; heuristic confidence"
        provenance = Provenance.FABRICATED if manual else Provenance.INFERRED
        # EXIF candidates: tungsten, daylight, cloudy. These are hypotheses, never measured light sources.
        light_source = 3 if cct < 4000 else (10 if cct > 6500 else 1)
        values = {"CCT": cct, "tint": tint, "illuminant_neutral_srgb": neutral.tolist(), "LightSource": light_source}
        records = {
            key: FieldRecord(
                key,
                value,
                provenance,
                description,
                unit="K" if key == "CCT" else None,
                confidence=confidence,
                user_override=manual,
                dependencies=["colour_temperature_range"] + ([] if manual else ["scene_linear_srgb"]),
                disclaimer="Approximate illuminant hypothesis, not a recovered camera white balance.",
            )
            for key, value in values.items()
        }
        model = colour_model or CameraColourModel()
        camera_neutral = model.camera_neutral(neutral)
        # The inverse ISP divides the camera response of a unit grey by these gains, so the written
        # neutral equals the illuminant's camera response rather than the sRGB neutral inverted.
        gains = model.camera_neutral(np.ones(3)) / camera_neutral
        gains = gains / gains[1]
        # The camera neutral inherits the hypothesis' provenance: it is that hypothesis seen by the camera.
        records["AsShotNeutral"] = FieldRecord(
            "AsShotNeutral",
            camera_neutral.tolist(),
            provenance,
            f"{description}; camera-native response through the camera colour model",
            confidence=confidence,
            user_override=manual,
            dependencies=["illuminant_neutral_srgb", model.source.field],
            disclaimer="Camera-space neutral of an approximate illuminant hypothesis.",
        )
        return WhiteBalanceResult(records, tuple(float(value) for value in gains))
