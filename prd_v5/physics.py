"""Small independent flat/dark measurements of baseline temporal sensor noise."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import ndtr

from prd_v5.sensor import BayerSensor


def photon_transfer(camera, *, tolerance=0.2, side=64):
    """Measure ADC slope and rounded dark variance, not real-camera calibration.

    Body maps and nonlinear charge response are deliberately excluded: DNG's
    simple affine NoiseProfile describes only the baseline temporal model.
    """
    if not 0 < tolerance < 1 or type(side) is not int or not 16 <= side <= 256:
        raise ValueError("Invalid flat-field measurement budget or tolerance")
    scale = camera.analogue_gain / camera.conversion_gain_e_per_dn
    sigma = camera.read_noise_e * scale if camera.read_noise else 0
    if sigma * 8 > camera.black_level:
        return {"status": "not measured", "reason": "Dark pedestal clips electronic noise at this gain"}
    with TemporaryDirectory(prefix="v5-flat-") as directory:
        signal = np.memmap(Path(directory) / "signal", mode="w+", dtype=np.float32, shape=(side, side, 3))
        codes = np.memmap(Path(directory) / "codes", mode="w+", dtype=np.uint16, shape=(side, side))
        samples = []
        try:
            for fraction in (0, 0.005, 0.01, 0.02, 0.04, 0.08):
                signal[:] = fraction
                BayerSensor(camera).process(signal, codes)
                values = codes.astype(np.float64) - camera.black_level
                samples.append({"mean_dn": float(values.mean()), "variance_dn": float(values.var(ddof=1))})
        finally:
            codes._mmap.close()
            signal._mmap.close()
    dark = samples[0]["variance_dn"]
    means = np.array([sample["mean_dn"] for sample in samples[1:]])
    variances = np.array([sample["variance_dn"] for sample in samples[1:]])
    measured_slope = float(np.dot(means, variances - dark) / np.dot(means, means))
    if sigma:
        bins = np.arange(-int(np.ceil(9 * sigma)) - 1, int(np.ceil(9 * sigma)) + 2)
        expected_dark = float(np.sum(bins**2 * (ndtr((bins + 0.5) / sigma) - ndtr((bins - 0.5) / sigma))))
    else:
        expected_dark = 0.0
    slope_error = abs(measured_slope / scale - 1) if camera.shot_noise else None
    dark_error = abs(dark - expected_dark) / max(expected_dark, 0.05)
    passed = dark_error <= tolerance and (slope_error is None or slope_error <= tolerance)
    return {
        "status": "passed" if passed else "failed",
        "scope": "baseline simulator self-consistency only",
        "samples": samples,
        "measured_slope_dn": measured_slope,
        "expected_slope_dn": scale if camera.shot_noise else 0,
        "dark_variance_dn": dark,
        "expected_rounded_dark_variance_dn": expected_dark,
        "slope_relative_error": slope_error,
        "dark_relative_error": dark_error,
        "relative_tolerance": tolerance,
        "limitations": "No physical calibration, body-map, dark-current or nonlinear-response validation",
    }
