"""PRD_v3 validation figures written to ``validation/`` from a finished ``ValidationRun``.

Every figure reads only downsampled products the validators already keep in memory, or
bounded crops of memory-mapped run arrays, so a 50 MP run never allocates a full frame here.
Figures render with matplotlib's non-interactive Agg canvas and are capped at ``dpi`` x size.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from synthraw.validation import _SRGB_TO_XYZ, _matrix, _srgb_encode, libraw_camera_to_srgb

if TYPE_CHECKING:
    from synthraw.validation_runner import ValidationRun

FIGURE_NAMES = (
    "roundtrip_difference.png",
    "cfa_planes.png",
    "mean_variance_curve.png",
    "noise_histograms.png",
    "noise_psd.png",
    "colour_validation.png",
    "wb_contact_sheet.png",
    "exposure_contact_sheet.png",
    "demosaic_comparison.png",
)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _subsample(array: np.ndarray, max_side: int, *, multiple: int = 1) -> np.ndarray:
    """Strided view at most ``max_side`` per side; the stride stays a multiple of the CFA period."""
    step = max(1, math.ceil(max(array.shape[:2]) / max_side))
    step = math.ceil(step / multiple) * multiple
    return np.asarray(array[::step, ::step])


def _centre_crop(array: np.ndarray, side: int, *, multiple: int = 2) -> np.ndarray:
    height, width = array.shape[:2]
    rows, cols = min(side, height) // multiple * multiple, min(side, width) // multiple * multiple
    top, left = (height - rows) // 2 // multiple * multiple, (width - cols) // 2 // multiple * multiple
    return np.asarray(array[top : top + rows, left : left + cols], dtype=np.float64)


def _code_bins(values: np.ndarray, max_bins: int = 60) -> np.ndarray:
    """Histogram edges on whole ADC code steps, so integer codes never alias into a comb."""
    unique = np.unique(values)
    if len(unique) < 2:
        return np.array([unique[0] - 0.5, unique[0] + 0.5]) if len(unique) else np.array([0.0, 1.0])
    step = float(np.diff(unique).min())
    width = step * math.ceil((unique[-1] - unique[0]) / step / max_bins + 1e-9)
    return np.arange(unique[0] - step / 2, unique[-1] + width, width)


def _placeholder(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=9, transform=ax.transAxes)
    ax.set_axis_off()


def _contact_sheet(plt: Any, thumbs: Mapping[str, np.ndarray], title: str, columns: int) -> Any:
    if not thumbs:
        figure, ax = plt.subplots(figsize=(4, 2), layout="constrained")
        _placeholder(ax, f"{title}: not rendered")
        return figure
    rows = math.ceil(len(thumbs) / columns)
    columns = min(columns, len(thumbs))
    first = next(iter(thumbs.values()))
    aspect = first.shape[0] / max(first.shape[1], 1)
    size = (2.8 * columns, (2.8 * aspect + 0.35) * rows + 0.4)
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False, layout="constrained")
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (label, thumb) in zip(axes.ravel(), thumbs.items(), strict=False):
        ax.imshow(np.clip(thumb, 0, 1), interpolation="nearest")
        ax.set_title(label, fontsize=9)
    figure.suptitle(title, fontsize=10)
    return figure


class ValidationFigureWriter:
    """Write the PRD validation figures (``FIGURE_NAMES``) into ``<run>/validation/``."""

    def __init__(self, *, dpi: int = 110, max_side: int = 512, psd_side: int = 256) -> None:
        self.dpi = dpi
        self.max_side = max_side
        self.psd_side = psd_side

    def write(self, run: ValidationRun) -> dict[str, Path]:
        plt = _pyplot()
        root = Path(run.root)
        out = root / "validation"
        out.mkdir(parents=True, exist_ok=True)
        groups = json.loads((root / "metadata" / "metadata.json").read_text())["groups"]
        phase = next(r["value"] for r in groups["sensor_profile"] if r["field"] == "applied_cfa_phase")
        builders: dict[str, Callable[[], Any]] = {
            "roundtrip_difference.png": lambda: self._roundtrip(plt, run),
            "cfa_planes.png": lambda: self._cfa_planes(plt, root, phase),
            "mean_variance_curve.png": lambda: self._mean_variance(plt, run),
            "noise_histograms.png": lambda: self._noise_histograms(plt, run),
            "noise_psd.png": lambda: self._noise_psd(plt, root, phase),
            "colour_validation.png": lambda: self._colour(plt, root),
            "wb_contact_sheet.png": lambda: _contact_sheet(
                plt, run.editability_thumbs.get("white_balance", {}), "White balance: cool / neutral / warm", 3
            ),
            "exposure_contact_sheet.png": lambda: _contact_sheet(
                plt, run.editability_thumbs.get("exposure", {}), "Exposure -4..+4 EV (linear LibRaw render)", 5
            ),
            "demosaic_comparison.png": lambda: self._demosaic(plt, run),
        }
        written = {}
        for name, build in builders.items():
            figure = build()
            try:
                figure.savefig(out / name, dpi=self.dpi, bbox_inches="tight")
            finally:
                plt.close(figure)
            written[name] = out / name
        return written

    def _roundtrip(self, plt: Any, run: ValidationRun) -> Any:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        result = run.roundtrip
        if result is None:
            figure, axes = plt.subplots(1, 4, figsize=(12, 1.6), layout="constrained")
            for ax in axes:
                _placeholder(ax, "round trip not measured (no developed TIFF)")
            return figure
        delta = _subsample(result.difference, self.max_side)
        supported = _subsample(result.supported, self.max_side).astype(bool)
        # Panel boxes follow the image aspect so the row carries no empty band above or below.
        aspect = delta.shape[0] / max(delta.shape[1], 1)
        panel_width = 3.2 / max(aspect, 1.0)
        size = (4 * panel_width + 1.0, panel_width * aspect + 0.8)
        figure, axes = plt.subplots(1, 4, figsize=size, layout="constrained")
        figure.get_layout_engine().set(wspace=0.06)
        panels = (
            ("source-derived target", _subsample(result.reference_srgb, self.max_side)),
            ("developed, aligned", _subsample(result.developed_srgb, self.max_side)),
        )
        for ax, (title, image) in zip(axes[:2], panels, strict=True):
            ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
            ax.set_title(title, fontsize=9)
        shown = axes[2].imshow(delta, cmap="magma", vmin=0, vmax=max(5.0, float(np.percentile(delta, 99))))
        axes[2].set_title("deltaE00 map", fontsize=9)
        # An inset in axes coordinates tracks the drawn image box, so the bar is exactly the panel height.
        colourbar = figure.colorbar(shown, cax=axes[2].inset_axes((1.03, 0.0, 0.04, 1.0)))
        colourbar.set_label("deltaE00", fontsize=8)
        colourbar.ax.tick_params(labelsize=7)
        colours = ("#e8833a", "#3a78c2")
        axes[3].imshow(supported, cmap=ListedColormap(colours), vmin=0, vmax=1, interpolation="nearest")
        axes[3].set_title("source-supported region (scored)", fontsize=9)
        share = float(supported.mean())
        axes[3].legend(
            handles=[
                Patch(color=colours[1], label=f"source-supported {share:.0%}"),
                Patch(color=colours[0], label=f"generated / excluded {1 - share:.0%}"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=7,
            frameon=False,
            handlelength=1.2,
            columnspacing=1.0,
        )
        for ax in axes:
            ax.set_axis_off()
        return figure

    def _cfa_planes(self, plt: Any, root: Path, phase: str) -> Any:
        mosaic = np.load(root / "raw" / "sensor_mosaic_uint16.npy", mmap_mode="r")
        tile = _subsample(mosaic, 2 * self.max_side, multiple=2).astype(np.float64)
        figure, axes = plt.subplots(1, 6, figsize=(18, 3.2), layout="constrained")
        axes[0].imshow(tile, cmap="gray", interpolation="nearest")
        axes[0].set_title(f"mosaic ({phase})", fontsize=9)
        zoom = _centre_crop(mosaic, 16)
        axes[1].imshow(zoom, cmap="gray", interpolation="nearest")
        axes[1].set_title("16x16 centre photosites", fontsize=9)
        for index, ax in enumerate(axes[2:]):
            row, col = divmod(index, 2)
            plane = tile[row::2, col::2]
            ax.imshow(plane, cmap="gray", interpolation="nearest")
            ax.set_title(f"plane ({row},{col}) = {phase[2 * row + col]}", fontsize=9)
        for ax in axes:
            ax.set_axis_off()
        return figure

    def _mean_variance(self, plt: Any, run: ValidationRun) -> Any:
        fit = run.noise_fit
        figure, (curve, residual) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        if fit is None:
            _placeholder(curve, "flat-field noise test not run")
            _placeholder(residual, "")
            return figure
        line = np.linspace(0, float(fit.means.max()), 100)
        curve.plot(fit.means, fit.variances, "o", label="measured flats")
        curve.plot(line, fit.slope * line + fit.offset, "-", label=f"fit S={fit.slope:.3g}, O={fit.offset:.3g}")
        curve.set_xlabel("mean (normalised DNG units)")
        curve.set_ylabel("temporal variance")
        curve.set_title(f"Var = S*mean + O, R^2 = {fit.r2:.5f}", fontsize=9)
        curve.legend(fontsize=8)
        relative = fit.residuals / np.maximum(fit.variances, np.finfo(float).tiny)
        residual.axhline(0, color="0.6", linewidth=0.8)
        residual.plot(fit.means, relative, "o-")
        # Symmetric log keeps the zero-signal flat at its true mean instead of clamping it.
        residual.set_xscale("symlog", linthresh=max(float(np.min(fit.means[fit.means > 0], initial=1e-4)), 1e-6))
        residual.set_xlabel("mean (normalised DNG units, symlog)")
        residual.set_ylabel("residual / variance")
        residual.set_title("relative fit residuals", fontsize=9)
        return figure

    def _noise_histograms(self, plt: Any, run: ValidationRun) -> Any:
        fit = run.noise_fit
        if fit is None or not fit.samples:
            figure, ax = plt.subplots(figsize=(4, 2), layout="constrained")
            _placeholder(ax, "flat-field noise test not run")
            return figure
        levels = list(fit.samples)
        columns = min(5, len(levels))
        rows = math.ceil(len(levels) / columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(3 * columns, 2.4 * rows), squeeze=False, layout="constrained"
        )
        for ax in axes.ravel():
            ax.set_axis_off()
        for ax, level in zip(axes.ravel(), levels, strict=False):
            values = fit.samples[level]
            ax.set_axis_on()
            ax.hist(values, bins=_code_bins(values), color="tab:blue")
            mean = float(values.mean())
            ax.set_title(f"flat {level:g} FW: mean {0.0 if abs(mean) < 1e-12 else mean:.3g}", fontsize=8)
            ax.tick_params(labelsize=7)
        figure.suptitle("Flat-field code histograms (one realisation per level)", fontsize=10)
        return figure

    def _noise_psd(self, plt: Any, root: Path, phase: str) -> Any:
        inter = root / "intermediates"
        noisy = _centre_crop(np.load(inter / "10_noisy_cfa.npy", mmap_mode="r"), self.psd_side)
        clean = _centre_crop(np.load(inter / "09_clean_cfa.npy", mmap_mode="r"), self.psd_side)
        residual = noisy - clean
        figure, (image, radial) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        for offset, name in enumerate(("(0,0)", "(0,1)", "(1,0)", "(1,1)")):
            row, col = divmod(offset, 2)
            plane = residual[row::2, col::2]
            plane = plane - plane.mean()
            power = np.abs(np.fft.fftshift(np.fft.fft2(plane))) ** 2 / max(plane.size, 1)
            centre = np.array(power.shape) // 2
            if offset == 0:
                # The DC bin is zero after mean removal; show it at the median so it does not set the scale.
                shown_power = power.copy()
                shown_power[tuple(centre)] = np.median(power)
                logged = np.log10(np.maximum(shown_power, 1e-12))
                shown = image.imshow(logged, cmap="viridis", vmin=np.percentile(logged, 1), vmax=logged.max())
                figure.colorbar(shown, ax=image, fraction=0.046, label="log10 power")
            yy, xx = np.indices(power.shape)
            radius = np.hypot(yy - centre[0], xx - centre[1]).astype(int)
            profile = np.bincount(radius.ravel(), power.ravel()) / np.maximum(np.bincount(radius.ravel()), 1)
            radial.semilogy(profile[1 : min(centre) + 1], label=f"{name} {phase[offset]}")
        image.set_title("log10 PSD of noise residual, plane (0,0)", fontsize=9)
        image.set_axis_off()
        radial.set_xlabel("radial frequency bin")
        radial.set_ylabel("power (electrons^2)")
        radial.set_title("radially averaged PSD (flat = white noise)", fontsize=9)
        radial.legend(fontsize=8)
        return figure

    def _colour(self, plt: Any, root: Path) -> Any:
        from synthraw.validation_runner import load_dng_records

        records = load_dng_records(root / "metadata" / "dng_metadata.json")
        figure, axes = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
        for ax, name in zip(axes[:2], ("ColorMatrix1", "ColorMatrix2"), strict=True):
            matrix = _matrix(records, name)
            ax.imshow(matrix, cmap="coolwarm", vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())
            for (row, col), value in np.ndenumerate(matrix):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"{name} (XYZ->camera), cond {np.linalg.cond(matrix):.1f}", fontsize=9)
            ax.set_xticks(range(3), ["X", "Y", "Z"])
            ax.set_yticks(range(3), ["R", "G", "B"])
        neutral = np.asarray(records["AsShotNeutral"].value, dtype=float)
        axes[2].bar(["R", "G", "B"], neutral, color=["tab:red", "tab:green", "tab:blue"])
        axes[2].set_title("AsShotNeutral (camera neutral)", fontsize=9)
        camera = np.load(root / "intermediates" / "05_camera_native_rgb.npy", mmap_mode="r")
        samples = _subsample(camera, 64).reshape(-1, 3).astype(np.float64)
        srgb = (samples / neutral) @ libraw_camera_to_srgb(_matrix(records, "ColorMatrix2")).T
        xyz = srgb @ _SRGB_TO_XYZ.T
        total = xyz.sum(axis=1)
        keep = total > 1e-6
        chroma = xyz[keep, :2] / total[keep, None]
        colours = _srgb_encode(srgb[keep] / max(float(srgb.max()), 1e-6))
        axes[3].scatter(chroma[:, 0], chroma[:, 1], s=6, c=colours)
        primaries = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
        axes[3].plot(primaries[:, 0], primaries[:, 1], "k-", linewidth=0.8, label="sRGB gamut")
        axes[3].plot(0.3127, 0.3290, "k+", markersize=10, label="D65")
        axes[3].set_xlim(0, 0.8)
        axes[3].set_ylim(0, 0.9)
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        axes[3].set_title("camera-native pixels via DNG profile", fontsize=9)
        axes[3].legend(fontsize=7)
        return figure

    def _demosaic(self, plt: Any, run: ValidationRun) -> Any:
        thumbs = {name: _srgb_encode(thumb) for name, thumb in run.demosaic_thumbs.items()}
        if "AHD" in run.demosaic_thumbs:
            for name, thumb in run.demosaic_thumbs.items():
                if name != "AHD":
                    diff = np.abs(thumb - run.demosaic_thumbs["AHD"]).mean(axis=-1, keepdims=True)
                    thumbs[f"|{name} - AHD| x10"] = np.repeat(np.clip(diff * 10, 0, 1), 3, axis=-1)
        return _contact_sheet(plt, thumbs, "Demosaic comparison (LibRaw, linear, sRGB-encoded preview)", 5)
