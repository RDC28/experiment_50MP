"""Realism metrics: the synthetic DNG measured side by side with a real camera raw.

The validation noise gates fit the simulator's own mosaic against the NoiseProfile it wrote, so they
prove self-consistency only. This module measures what a real sensor looks like and the synthetic
DNG the same way, with the same code, so the two can be compared: black-subtracted histograms,
noise against signal (photon transfer), the dark read-noise tail, row/column banding and its power
spectrum, hot and dead photosites, the white balance and the colour matrix.

Every number is a property of one frame. A single reference raw supports per-frame statistics
(banding, tails, defect density) but not a calibration: photon transfer comes from the flattest
blocks of an ordinary scene, not from flat fields. The comparison is labelled realism, it is report
only, and no gate depends on it. Reading the reference file is the caller's choice; nothing here
copies it into the run tree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REALISM_JSON = "realism_comparison.json"
REALISM_FIGURE = "realism_comparison.png"
# Histogram bins in EV below the clipping point of the black-subtracted range.
HISTOGRAM_EDGES_EV = np.arange(-14.0, 0.25, 0.25)
_GAUSSIAN_TAIL = {3: 2.6998e-3, 4: 6.334e-5, 5: 5.733e-7}
_D65_XYZ = np.array([0.95047, 1.0, 1.08883])


def _plane_names(pattern: np.ndarray, colours: str) -> dict[tuple[int, int], str]:
    """Name each 2x2 CFA position R, G1, G2 or B from LibRaw's raw_pattern and color_desc."""
    names, greens = {}, 0
    for row in range(2):
        for column in range(2):
            colour = colours[int(pattern[row, column])]
            if colour == "G":
                greens += 1
                colour = f"G{greens}"
            names[(row, column)] = colour
    if sorted(names.values()) != ["B", "G1", "G2", "R"]:
        raise ValueError(f"Realism metrics need an RGGB-family Bayer mosaic, got {colours!r} {pattern.tolist()}")
    return names


def _block_size(plane: np.ndarray) -> int:
    """32 plane pixels on a full frame; smaller on tiny test mosaics so there are enough blocks."""
    return int(min(32, max(4, min(plane.shape) // 16)))


def _detrended_blocks(plane: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and plane-detrended variance of every block, so smooth scene gradients do not count as noise."""
    height, width = (plane.shape[0] // block) * block, (plane.shape[1] // block) * block
    blocks = plane[:height, :width].reshape(height // block, block, width // block, block).astype(np.float32)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block, block)
    means = blocks.mean(axis=(1, 2))
    axis = np.arange(block, dtype=np.float32) - (block - 1) / 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    centred = blocks - means[:, None, None]
    slope_y = (centred * yy).sum(axis=(1, 2)) / (yy**2).sum()
    slope_x = (centred * xx).sum(axis=(1, 2)) / (xx**2).sum()
    residual = centred - slope_y[:, None, None] * yy - slope_x[:, None, None] * xx
    variance = residual.var(axis=(1, 2)) * block**2 / (block**2 - 3)
    return means, variance


def _photon_transfer(
    means: np.ndarray, variance: np.ndarray, black: float, white: float, min_blocks: int
) -> tuple[list[dict[str, float]], dict[str, float] | None]:
    """Lower envelope of block variance against signal, and its linear fit variance = k * signal + v0."""
    signal = means - black
    edges = np.concatenate(([-np.inf, 0.0], np.geomspace(4, max(white - black, 8), 40)))
    index = np.digitize(signal, edges) - 1
    envelope = []
    for bin_index in range(len(edges) - 1):
        selected = index == bin_index
        if np.count_nonzero(selected) >= min_blocks:
            floor = float(np.quantile(variance[selected], 0.05))
            lowest = selected & (variance <= floor)
            envelope.append(
                {
                    "signal_dn": float(signal[lowest].mean()),
                    "variance_dn2": floor,
                    "blocks": int(np.count_nonzero(selected)),
                }
            )
    usable = [point for point in envelope if 0 < point["signal_dn"] < 0.7 * (white - black)]
    if len(usable) < 4:
        return envelope, None
    xs = np.array([point["signal_dn"] for point in usable])
    ys = np.array([point["variance_dn2"] for point in usable])
    weights = np.sqrt([point["blocks"] for point in usable])
    design = np.vstack([xs, np.ones_like(xs)]).T * weights[:, None]
    slope, offset = np.linalg.lstsq(design, ys * weights, rcond=None)[0]
    fit = {"dn2_per_dn": float(slope), "read_variance_dn2": float(offset), "points": len(usable)}
    if slope > 0:
        # Electrons make two sensors with different ADC ranges comparable.
        fit["conversion_gain_e_per_dn"] = float(1 / slope)
        fit["full_well_e"] = float((white - black) / slope)
        fit["read_noise_e"] = float(math.sqrt(max(offset, 0.0)) / slope)
    return envelope, fit


def _detilt(tile: np.ndarray) -> np.ndarray:
    """``tile`` minus its least-squares plane a + b*y + c*x."""
    tile = tile.astype(np.float64)
    yy, xx = np.meshgrid(*(np.arange(n) - (n - 1) / 2 for n in tile.shape), indexing="ij")
    centred = tile - tile.mean()
    return centred - (centred * yy).sum() / (yy**2).sum() * yy - (centred * xx).sum() / (xx**2).sum() * xx


def _dark_statistics(plane: np.ndarray, means: np.ndarray, block: int, black: float) -> tuple[dict, np.ndarray]:
    """Read-noise tail and row/column banding in the darkest blocks of the frame."""
    blocks_per_row = (plane.shape[1] // block) or 1
    darkest = np.argsort(means)[: max(min(200, means.size), int(0.002 * means.size))]
    values, noise, row_deviation, column_deviation = [], [], [], []
    for flat_index in darkest:
        by, bx = divmod(int(flat_index), blocks_per_row)
        tile = plane[by * block : (by + 1) * block, bx * block : (bx + 1) * block].astype(np.float32) - black
        values.append(tile.ravel())
        # Noise is measured about each block's own fitted plane, so scene signal is not counted as noise.
        flat = _detilt(tile)
        noise.append(flat.ravel())
        row_deviation.append(flat.mean(axis=1))
        column_deviation.append(flat.mean(axis=0))
    dark, residual = np.concatenate(values), np.concatenate(noise)
    std = float(residual.std())
    z = residual / (std or 1.0)
    sigma_mad = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    rows, columns = np.asarray(row_deviation), np.asarray(column_deviation)
    # Line means over `block` samples carry std**2 / block of white noise; only the excess is banding.
    floor = std**2 / block
    statistics = {
        "blocks": int(darkest.size),
        "mean_above_black_dn": float(dark.mean()),
        "sigma_mad_dn": sigma_mad,
        "std_dn": std,
        "excess_kurtosis": float(np.mean(z**4) - 3),
        "positive_tail_beyond_4sigma": float(np.mean(z > 4)),
        "negative_tail_beyond_4sigma": float(np.mean(z < -4)),
        "gaussian_tail_beyond_4sigma_one_side": _GAUSSIAN_TAIL[4] / 2,
        "fraction_below_black": float(np.mean(dark < 0)),
        "row_banding_rms_dn": float(math.sqrt(max(rows.var() - floor, 0.0))),
        "column_banding_rms_dn": float(math.sqrt(max(columns.var() - floor, 0.0))),
    }
    return statistics, dark


_PSD_LOW_BINS = 2


def _banding_psd(plane: np.ndarray, black: float) -> dict[str, Any]:
    """2D power spectrum of the darkest square window: power on the axes over power off them.

    Row banding concentrates power on the vertical-frequency axis (kx = 0), column banding on the
    horizontal one. White noise gives ratios near 1.
    """
    side = int(min(256, min(plane.shape) // 16 * 16))
    if side < 16:
        return {}
    step = max(1, side // 16)
    best, origin = math.inf, (0, 0)
    for top in range(0, plane.shape[0] - side + 1, step * 4):
        for left in range(0, plane.shape[1] - side + 1, step * 4):
            level = float(plane[top : top + side : step, left : left + side : step].mean())
            if level < best:
                best, origin = level, (top, left)
    top, left = origin
    window = plane[top : top + side, left : left + side].astype(np.float64) - black
    # A single hot photosite has a flat spectrum that would swamp the banding lines; clip outliers first.
    median = float(np.median(window))
    spread = 1.4826 * float(np.median(np.abs(window - median))) or 1.0
    window = np.clip(window, median - 8 * spread, median + 8 * spread)
    level = float(window.mean())
    # Removing the fitted plane keeps the scene's own gradient from putting power on the axes.
    taper = np.hanning(side)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(_detilt(window) * taper[:, None] * taper[None, :])) ** 2)
    centre = side // 2
    # The lowest few frequencies hold residual scene shading and the taper's leakage, not banding.
    low = slice(centre - _PSD_LOW_BINS, centre + _PSD_LOW_BINS + 1)
    vertical, horizontal = spectrum[:, centre].copy(), spectrum[centre, :].copy()
    vertical[low] = horizontal[low] = np.nan
    off_axis = spectrum.copy()
    off_axis[:, centre] = off_axis[centre, :] = np.nan
    off_axis[low, low] = np.nan
    off_mean = float(np.nanmean(off_axis))
    return {
        "window_origin_plane_px": [int(top), int(left)],
        "window_side": side,
        "window_mean_above_black_dn": level,
        "row_axis_power_over_offaxis": float(np.nanmean(vertical) / off_mean) if off_mean else math.nan,
        "column_axis_power_over_offaxis": float(np.nanmean(horizontal) / off_mean) if off_mean else math.nan,
    }


def _defects(plane: np.ndarray, black: float, fit: Mapping[str, float] | None) -> dict[str, Any]:
    """Isolated hot and dead photosites: far above (or below, near black) all eight neighbours."""
    from scipy import ndimage

    values = plane.astype(np.float32)
    footprint = np.ones((3, 3), bool)
    footprint[1, 1] = False
    highest = ndimage.maximum_filter(values, footprint=footprint, mode="nearest")
    lowest = ndimage.minimum_filter(values, footprint=footprint, mode="nearest")
    median = ndimage.median_filter(values, footprint=footprint, mode="nearest")
    slope = fit["dn2_per_dn"] if fit else 0.3
    offset = fit["read_variance_dn2"] if fit else 1.0
    local_sigma = np.sqrt(np.maximum(offset + max(slope, 0.0) * np.maximum(median - black, 0), 1.0))
    margin = np.maximum(8 * local_sigma, 40)
    hot = (values - highest) > margin
    dead = ((lowest - values) > margin) & (values <= black + 16)
    return {
        "hot_count": int(hot.sum()),
        "hot_fraction": float(hot.mean()),
        "dead_count": int(dead.sum()),
        "dead_fraction": float(dead.mean()),
    }


def measure_mosaic(
    mosaic: np.ndarray,
    *,
    black: float,
    white: float,
    pattern: np.ndarray,
    colours: str = "RGBG",
) -> dict[str, Any]:
    """Per-plane realism statistics of one Bayer mosaic in its own code units.

    ``pattern`` is LibRaw's 2x2 raw_pattern and ``colours`` its color_desc. ``white`` is the
    clipping code. Histograms are black-subtracted and expressed in EV below the clipping range,
    so raws with different black and white levels share one axis.
    """
    mosaic = np.asarray(mosaic)
    if mosaic.ndim != 2 or mosaic.shape[0] < 4 or mosaic.shape[1] < 4:
        raise ValueError("Expected a 2D Bayer mosaic of at least 4 x 4 photosites")
    if not white > black:
        raise ValueError("White level must exceed the black level")
    names = _plane_names(np.asarray(pattern), colours)
    span = white - black
    planes: dict[str, Any] = {}
    for (row, column), name in sorted(names.items()):
        plane = mosaic[row::2, column::2]
        signal = plane.astype(np.float64) - black
        positive = signal[signal > 0]
        ev = np.log2(np.minimum(positive, span) / span)
        counts, _ = np.histogram(ev, bins=HISTOGRAM_EDGES_EV)
        block = _block_size(plane)
        means, variance = _detrended_blocks(plane, block)
        envelope, fit = _photon_transfer(means, variance, black, white, min_blocks=max(5, means.size // 400))
        dark, _ = _dark_statistics(plane, means, block, black)
        median = float(np.median(signal))
        planes[name] = {
            "histogram_counts": counts.tolist(),
            "fraction_at_or_below_black": float(np.mean(signal <= 0)),
            "fraction_clipped": float(np.mean(signal >= span)),
            "median_ev_below_clip": float(math.log2(median / span)) if median > 0 else None,
            "p99_ev_below_clip": float(math.log2(max(np.quantile(signal, 0.99), 1e-9) / span)),
            "photon_transfer_envelope": envelope,
            "photon_transfer_fit": fit,
            "darkest_blocks": dark,
            "banding_psd": _banding_psd(plane, black),
            "defects": _defects(plane, black, fit),
        }
    return {
        "shape": list(mosaic.shape),
        "black": float(black),
        "white": float(white),
        "histogram_edges_ev": HISTOGRAM_EDGES_EV.tolist(),
        "planes": planes,
    }


def _normalised_neutral(values: Any) -> list[float] | None:
    values = np.asarray(values, dtype=float)[:3]
    if values.size != 3 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return None
    return (values / values[1]).tolist()


def measure_raw(path: str | Path) -> dict[str, Any]:
    """Open any LibRaw-readable raw (a camera ARW or the synthetic DNG) and measure it.

    The visible area is measured; LibRaw's black and clipping levels, as-shot white balance and
    XYZ-to-camera matrix are what a raw developer would use.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.copy()
        black = float(np.mean(raw.black_level_per_channel))
        # Sony raws clip below the container white level (ARW: 15360 of 16383); prefer the camera's.
        camera_white = [value for value in (raw.camera_white_level_per_channel or ()) if value > 0]
        white = float(min(camera_white)) if camera_white else float(raw.white_level)
        pattern = np.asarray(raw.raw_pattern)
        colours = raw.color_desc.decode()
        camera_wb = np.asarray(raw.camera_whitebalance, dtype=float)[:3]
        xyz_to_camera = np.asarray(raw.rgb_xyz_matrix, dtype=float)[:3]
    measured = measure_mosaic(mosaic, black=black, white=white, pattern=pattern, colours=colours)
    del mosaic
    # LibRaw's as-shot multipliers are reciprocal to the camera neutral (DNG AsShotNeutral).
    neutral = _normalised_neutral(1 / camera_wb) if np.all(camera_wb > 0) else None
    d65_response = xyz_to_camera @ _D65_XYZ
    measured.update(
        file=Path(path).name,
        as_shot_neutral=neutral,
        xyz_to_camera=xyz_to_camera.tolist() if np.all(np.isfinite(xyz_to_camera)) else None,
        d65_camera_neutral=_normalised_neutral(d65_response),
    )
    return measured


def _green(planes: Mapping[str, Any], read: Any) -> float | None:
    values = [read(planes[name]) for name in ("G1", "G2")]
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


# (label, unit, reader of one plane's statistics); green planes are averaged.
_PLANE_METRICS: tuple[tuple[str, str, Any], ...] = (
    ("Median signal", "EV below clip", lambda p: p["median_ev_below_clip"]),
    ("99th percentile signal", "EV below clip", lambda p: p["p99_ev_below_clip"]),
    ("Clipped photosites", "fraction", lambda p: p["fraction_clipped"]),
    ("At or below black", "fraction", lambda p: p["fraction_at_or_below_black"]),
    ("Full well (photon transfer)", "e-", lambda p: (p["photon_transfer_fit"] or {}).get("full_well_e")),
    ("Read noise (photon transfer)", "e-", lambda p: (p["photon_transfer_fit"] or {}).get("read_noise_e")),
    ("Dark read noise sigma (MAD)", "DN", lambda p: p["darkest_blocks"]["sigma_mad_dn"]),
    ("Dark positive tail > 4 sigma", "fraction", lambda p: p["darkest_blocks"]["positive_tail_beyond_4sigma"]),
    ("Dark excess kurtosis", "", lambda p: p["darkest_blocks"]["excess_kurtosis"]),
    (
        "Row banding / dark sigma",
        "ratio",
        lambda p: p["darkest_blocks"]["row_banding_rms_dn"] / (p["darkest_blocks"]["std_dn"] or math.nan),
    ),
    (
        "Column banding / dark sigma",
        "ratio",
        lambda p: p["darkest_blocks"]["column_banding_rms_dn"] / (p["darkest_blocks"]["std_dn"] or math.nan),
    ),
    ("PSD row-axis power / off-axis", "ratio", lambda p: p["banding_psd"].get("row_axis_power_over_offaxis")),
    ("PSD column-axis power / off-axis", "ratio", lambda p: p["banding_psd"].get("column_axis_power_over_offaxis")),
    ("Hot photosites", "fraction", lambda p: p["defects"]["hot_fraction"]),
    ("Dead photosites", "fraction", lambda p: p["defects"]["dead_fraction"]),
)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def compare(reference: Mapping[str, Any], synthetic: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Side-by-side rows: metric, plane, unit, reference value, synthetic value, synthetic/reference."""
    rows = []

    def add(metric: str, plane: str, unit: str, ref: Any, syn: Any) -> None:
        ref, syn = _number(ref), _number(syn)
        ratio = syn / ref if ref not in (None, 0.0) and syn is not None else None
        rows.append(
            {"metric": metric, "plane": plane, "unit": unit, "reference": ref, "synthetic": syn, "ratio": ratio}
        )

    for metric, unit, read in _PLANE_METRICS:
        for plane in ("R", "G", "B"):
            if plane == "G":
                ref, syn = _green(reference["planes"], read), _green(synthetic["planes"], read)
            else:
                ref, syn = read(reference["planes"][plane]), read(synthetic["planes"][plane])
            add(metric, plane, unit, ref, syn)
    for key, metric in (("as_shot_neutral", "AsShotNeutral"), ("d65_camera_neutral", "D65 camera neutral")):
        for index, plane in ((0, "R"), (2, "B")):
            ref, syn = reference.get(key), synthetic.get(key)
            add(metric, plane, "G = 1", ref[index] if ref else None, syn[index] if syn else None)
    return rows


def write_realism(
    root: str | Path, reference_raw: str | Path, *, synthetic_dng: str | Path | None = None
) -> dict[str, Any]:
    """Measure the reference raw and the run's DNG; write ``validation/realism_comparison.{json,png}``.

    Returns the written document. The reference file itself is never copied into the run tree.
    """
    root = Path(root)
    synthetic_dng = Path(synthetic_dng) if synthetic_dng else root / "raw" / "synthetic_50mp.dng"
    reference = measure_raw(reference_raw)
    synthetic = measure_raw(synthetic_dng)
    document = {
        "label": "Realism: synthetic DNG against one real reference raw (report only, no gate)",
        "caveat": (
            "Single-frame statistics of two different scenes. Photon transfer uses the flattest blocks of "
            "ordinary content, not flat fields; scene content (exposure, texture, dark area) differs between "
            "the two files, so read the ratios as realism indicators, not calibration errors."
        ),
        "reference": reference,
        "synthetic": synthetic,
        "rows": compare(reference, synthetic),
    }
    out = root / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / REALISM_JSON).write_text(json.dumps(_finite(document), indent=1, allow_nan=False) + "\n")
    write_realism_figure(document, out / REALISM_FIGURE)
    return document


def _finite(value: Any) -> Any:
    """JSON-safe copy: NumPy scalars become Python numbers and non-finite numbers become null."""
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_realism_figure(document: Mapping[str, Any], path: Path) -> Path:
    """Black-subtracted histograms and noise against signal, reference and synthetic overlaid."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figure, (histograms, transfer) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    centres = (HISTOGRAM_EDGES_EV[:-1] + HISTOGRAM_EDGES_EV[1:]) / 2
    colours = {"R": "#c0392b", "G1": "#27ae60", "B": "#2c6fbb"}
    for source, style in (("reference", "-"), ("synthetic", "--")):
        planes = document[source]["planes"]
        for name, colour in colours.items():
            counts = np.asarray(planes[name]["histogram_counts"], dtype=float)
            total = counts.sum() or 1.0
            histograms.plot(centres, counts / total, style, color=colour, lw=1.2, label=f"{source} {name}")
            envelope = planes[name]["photon_transfer_envelope"]
            points = [(p["signal_dn"], p["variance_dn2"]) for p in envelope if p["signal_dn"] > 0]
            if points:
                span = document[source]["white"] - document[source]["black"]
                xs, ys = np.array(points).T
                # Normalised to the clipping range so two ADC ranges share the axes.
                transfer.loglog(xs / span, ys / span**2, style, color=colour, lw=1.2, label=f"{source} {name}")
    histograms.set(xlabel="EV below clip (black-subtracted)", ylabel="fraction of photosites", yscale="log")
    histograms.set_title("Black-subtracted histograms", fontsize=10)
    transfer.set(xlabel="signal / clip range", ylabel="variance / clip range^2")
    transfer.set_title("Noise against signal (flattest blocks)", fontsize=10)
    for axes in (histograms, transfer):
        axes.grid(alpha=0.3)
        axes.legend(fontsize=7)
    figure.suptitle("Realism: reference raw (solid) against the synthetic DNG (dashed)", fontsize=11)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
