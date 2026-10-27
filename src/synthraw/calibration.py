"""Paired dark/flat photon-transfer measurements for PRD_v3 full calibration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from synthraw.config import ExperimentConfig
from synthraw.provenance import FieldRecord, Provenance

if TYPE_CHECKING:
    from synthraw.camera_profile import CameraProfile


def calibrate_sensor(profile: CameraProfile, config: ExperimentConfig, reference: Path) -> None:
    """Measure pooled Bayer-plane statistics without stacking input captures.

    Darks must share exposure/temperature with the flats. Consecutive flat paths
    form equal-illumination pairs; supply at least two distinct unsaturated levels.
    Uniform illumination and stable gain are capture requirements, not inferred facts.
    Temporal pair differences remove body pattern from photon-transfer variance.
    """
    darks, flats = config.DARK_FRAME_PATHS, config.FLAT_FIELD_PATHS
    if len(darks) < 2 or len(darks) % 2:
        raise ValueError("DARK_FRAME_PATHS requires an even number of at least two matched dark frames")
    if len(flats) < 4 or len(flats) % 2:
        raise ValueError("FLAT_FIELD_PATHS requires at least two flat pairs at distinct signal levels")
    shape = None
    phase = profile.sensor["cfa_pattern"].value
    white = profile.sensor["WhiteLevel"].value

    def read(path: str) -> np.ndarray:
        nonlocal shape
        import rawpy

        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing calibration asset: {path}")
        try:
            with rawpy.imread(path) as raw:
                pattern = raw.raw_pattern
                decoded = "" if pattern is None else "".join(chr(raw.color_desc[int(i)]) for i in pattern.ravel())
                if decoded != phase or raw.white_level != white:
                    raise ValueError("CFA/WhiteLevel does not match reference")
                data = raw.raw_image_visible.astype(np.float64)
        except Exception as error:
            raise ValueError(f"Unreadable or incompatible calibration asset {path}: {error}") from error
        if shape is None:
            shape = data.shape
        if data.shape != shape or min(data.shape) < 4:
            raise ValueError(f"Calibration dimensions mismatch or too small: {path}")
        if np.any(data >= white) or np.any(data <= 0):
            raise ValueError(f"Clipped calibration asset: {path}")
        return data

    # Per-phase spatial variance removes Bayer channel mean differences.
    def variance(data: np.ndarray) -> float:
        return float(np.mean([np.var(data[y::2, x::2], ddof=1) for y in range(2) for x in range(2)]))

    read(str(reference))  # Establish the common raster shape from the reference capture.
    dark_mean = None
    read_var = 0.0
    for index in range(0, len(darks), 2):
        a, b = read(darks[index]), read(darks[index + 1])
        pair = (a + b) / len(darks)
        dark_mean = pair if dark_mean is None else dark_mean + pair
        read_var += variance(a - b) / len(darks)
    black = float(np.mean(dark_mean))
    dsnu = np.sqrt(max(0.0, variance(dark_mean) - read_var / len(darks)))
    signals, temporal, spatial = [], [], []
    for index in range(0, len(flats), 2):
        a, b = read(flats[index]), read(flats[index + 1])
        mean = (a + b) / 2 - dark_mean
        signal = float(np.mean(mean))
        if signal <= 0:
            raise ValueError(f"Flat signal must exceed dark level: {flats[index]}")
        signals.append(signal)
        temporal.append(variance(a - b) / 2)
        # Normalise each Bayer plane to its own mean before pooling PRNU.
        spatial.append(
            float(
                np.mean(
                    [
                        np.var(mean[y::2, x::2], ddof=1) / np.mean(mean[y::2, x::2]) ** 2
                        for y in range(2)
                        for x in range(2)
                    ]
                )
            )
        )
    if np.ptp(signals) <= max(signals) * 0.01:
        raise ValueError("Flat pairs require distinct signal levels for photon-transfer gain")
    slope, _ = np.polyfit(signals, temporal, 1)
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("Flat photon-transfer slope must be finite and positive")
    gain = 1 / slope
    prnu_var = np.mean(
        [
            spatial_var - (noise / 2 + read_var / len(darks)) / signal**2
            for spatial_var, noise, signal in zip(spatial, temporal, signals, strict=True)
        ]
    )
    profile.sensor["calibration_frames"] = FieldRecord(
        "calibration_frames",
        [str(frame) for frame in (*darks, *flats)],
        Provenance.DETERMINISTIC,
        "User-supplied dark and flat calibration captures",
    )
    dependencies = ["reference_raw", "calibration_frames"]
    for field, value, unit in (
        ("BlackLevel", black, "DN"),
        ("conversion_gain_e_per_dn", float(gain), "electrons/DN"),
        ("read_noise_electrons", float(np.sqrt(read_var) * gain), "electrons RMS"),
        ("dsnu_sigma_dn", float(dsnu), "DN"),
        ("prnu_sigma", float(np.sqrt(max(0.0, prnu_var))), "fraction"),
    ):
        profile.sensor[field] = FieldRecord(
            field,
            value,
            Provenance.MEASURED,
            "Paired dark/flat photon transfer; matched exposure, temperature and uniform flat illumination required",
            unit=unit,
            dependencies=dependencies,
            disclaimer="Measured from supplied calibration captures, not the synthetic exposure.",
        )
    profile.camera["profile_source"] = FieldRecord(
        "profile_source", "FULL_CALIBRATION", Provenance.DETERMINISTIC, "Explicit user profile selection"
    )
