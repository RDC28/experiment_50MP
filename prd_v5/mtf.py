"""Bounded sampled-edge response, with explicit caller measurement evidence."""

from numbers import Real

import numpy as np


def edge_mtf(profile, *, sample_spacing_px=1.0, evidence=None):
    """Measure the normalized FFT magnitude of a first-difference edge profile.

    Supply a uniformly sampled linear-light ESF normal to a straight isolated
    edge, with both plateaus included. Evidence must describe extraction and
    linearization. This is not ISO slanted-edge SFR or calibrated optical MTF.
    No smoothing, window, derivative correction or implicit resampling is used.
    """
    report = {
        "status": "not measured",
        "method": "sampled ESF first difference; unwindowed FFT magnitude normalized at DC",
        "units": "cycles per native pixel",
        "limitations": "Caller-attested edge/linearization; finite-profile sampled response, not lens MTF or truth",
    }
    if profile is None:
        return {**report, "reason": "No measured edge profile supplied"}
    if evidence is None:
        return {**report, "reason": "No edge extraction/linearization evidence supplied"}
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"edge", "linearization"}
        or any(not isinstance(v, str) or not v.strip() for v in evidence.values())
    ):
        raise ValueError("evidence requires nonempty edge and linearization strings")
    if (
        isinstance(sample_spacing_px, (bool, np.bool_))
        or not isinstance(sample_spacing_px, Real)
        or not np.isfinite(sample_spacing_px)
        or sample_spacing_px <= 0
    ):
        raise ValueError("sample_spacing_px must be finite and positive")
    samples = np.asarray(profile)
    if samples.ndim != 1 or not 16 <= samples.size <= 65536 or samples.dtype.kind not in "uif":
        raise ValueError("profile must contain 16 to 65536 real numeric samples")
    samples = samples.astype(np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError("profile must be finite")
    scale = max(1.0, float(np.max(np.abs(samples))))
    samples /= scale
    contrast = float(samples[-1] - samples[0])
    if abs(contrast) <= 1e-12:
        return {**report, "reason": "No resolvable endpoint edge contrast"}
    lsf = np.diff(samples) / contrast
    # Padding refines crossing interpolation but adds no spatial information.
    fft_size = 1 << (4 * lsf.size - 1).bit_length()
    response = np.abs(np.fft.rfft(lsf, n=fft_size))
    response /= response[0]
    frequencies = np.fft.rfftfreq(fft_size, d=float(sample_spacing_px))
    # Oversampling never extends the reported native-pixel Nyquist limit.
    keep = frequencies <= min(0.5, 0.5 / sample_spacing_px)
    frequencies, response = frequencies[keep], response[keep]

    def crossing(level):
        indices = np.flatnonzero(response <= level)
        if not indices.size:
            return {"frequency": None, "reason": "No crossing within sampled/native Nyquist limit"}
        i = int(indices[0])
        fraction = (response[i - 1] - level) / (response[i - 1] - response[i])
        return {"frequency": float(frequencies[i - 1] + fraction * (frequencies[i] - frequencies[i - 1]))}

    return {
        **report,
        "status": "measured sampled edge response",
        "evidence": dict(evidence),
        "samples": samples.size,
        "sample_spacing_px": float(sample_spacing_px),
        "normalized_endpoint_contrast": contrast,
        "profile_scale": scale,
        "frequencies": frequencies.tolist(),
        "response": response.tolist(),
        "mtf50": crossing(0.5),
        "mtf10": crossing(0.1),
    }


def tiff_edge_mtf(path, *, region=None, normal_axis="x", edge_evidence=None):
    """Average a caller-selected axis-aligned edge in an encoded sRGB TIFF.

    Region is (top, left, bottom, right), in native pixels. The caller must
    attest an isolated straight edge normal to x or y, with both plateaus.
    Decode sRGB before Rec.709 luminance and averaging. No alignment, edge
    detection, resampling or slanted-edge correction is inferred.
    """
    import tifffile

    from prd_v5.comparison import validate_region

    validate_region(region)
    if normal_axis not in ("x", "y"):
        raise ValueError("normal_axis must be x or y")
    if edge_evidence is not None and (not isinstance(edge_evidence, str) or not edge_evidence.strip()):
        raise ValueError("edge_evidence must be a nonempty string")
    if region is None or edge_evidence is None:
        return {
            "status": "not measured",
            "reason": "No caller-selected edge region and edge evidence supplied",
        }
    array = tifffile.memmap(path, mode="r")
    try:
        if array.dtype != np.uint16 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Edge extraction requires uint16 encoded sRGB RGB TIFF")
        top, left, bottom, right = region
        if bottom > array.shape[0] or right > array.shape[1]:
            raise ValueError("region exceeds TIFF raster")
        length = right - left if normal_axis == "x" else bottom - top
        if not 16 <= length <= 65536 or (bottom - top) * (right - left) > 1048576:
            raise ValueError("edge region requires 16 to 65536 normal samples and at most 1048576 pixels")
        profile = np.zeros(length, dtype=np.float64)
        # Bound decoded working memory independently of the selected rectangle.
        for row in range(top, bottom):
            rgb = array[row, left:right].astype(np.float64) / 65535
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
            if normal_axis == "x":
                profile += luminance
            else:
                profile[row - top] = luminance.mean()
        if normal_axis == "x":
            profile /= bottom - top
        result = edge_mtf(
            profile,
            evidence={
                "edge": edge_evidence,
                "linearization": "Caller-declared encoded sRGB; inverse sRGB transfer then Rec.709 luminance",
            },
        )
        return {
            **result,
            "region": list(region),
            "normal_axis": normal_axis,
            "extraction": "Native-pixel axis-aligned averaging; no slanted-edge correction or automatic selection",
        }
    finally:
        array._mmap.close()
