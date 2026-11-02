"""The frozen max preset: render RGB to the three 16-bit RAW outputs (sections 5.1, 5.2, 5.4, 9.3).

The pipeline sees only the 8-bit render and camera-model constants. The neutral is the section 5.3
prior unless a caller passes another one for a labelled diagnostic (oracle rungs, ablation rows).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from prd_v6 import colour, constants, resample, sr


def camera_matrix() -> np.ndarray:
    from synthraw.camera_profile import ILCE1M2_XYZ_TO_CAMERA_D65

    return np.asarray(ILCE1M2_XYZ_TO_CAMERA_D65, dtype=np.float64)


@dataclass(frozen=True)
class Preset:
    """Frozen from the section 5.1 reasoning; a run measures it but never changes it (M3)."""

    name: str = "max"
    sr_model: str = "HAT-L x4"
    self_ensemble: bool = True
    tile: int = constants.SR_TILE
    overlap: int = constants.SR_OVERLAP
    back_projection_iterations: int = constants.BACK_PROJECTION_ITERATIONS
    illuminant: str = "prior"
    noise_50mp: bool = True
    noise_seed: int = constants.NOISE_SEED
    output_scale: str = "16-bit (14-bit levels x4)"
    precision: str = "float32"

    def as_dict(self) -> dict:
        return asdict(self)


MAX = Preset()


def decode(encoded_uint8: np.ndarray) -> np.ndarray:
    """8-bit sRGB render to linear light, exact transfer function."""
    return colour.srgb_decode(np.asarray(encoded_uint8, dtype=np.float32) / 255.0)


def to_camera(linear: np.ndarray, neutral) -> np.ndarray:
    """Linear sRGB to black-subtracted camera RGB in DN16 (render white = sensor clip)."""
    return colour.linear_srgb_to_camera(linear, neutral, camera_matrix(), constants.SIGNAL_16)


def quantize(black_subtracted: np.ndarray) -> np.ndarray:
    """Stored uint16 samples: round, add black 2048, clip to [0, 61440]."""
    stored = np.rint(np.asarray(black_subtracted, dtype=np.float32)) + constants.BLACK_16
    return np.clip(stored, 0, constants.WHITE_16).astype(np.uint16)


def quantize_clean(black_subtracted: np.ndarray) -> np.ndarray:
    """A noise-free output cannot go below black: clip the signal to [0, white] first."""
    return quantize(np.clip(black_subtracted, 0.0, constants.SIGNAL_16))


def stored_to_signal(stored: np.ndarray, black=constants.BLACK_16) -> np.ndarray:
    return np.asarray(stored, dtype=np.float32) - np.asarray(black, dtype=np.float32)


def mosaic(camera_rgb: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Sample camera RGB at the CFA: one value per photosite at the reference phase."""
    return np.take_along_axis(camera_rgb, layout[..., None].astype(np.intp), axis=2)[..., 0]


def add_noise(signal16: np.ndarray, seed: int) -> np.ndarray:
    """Shot and read noise at the measured A1 II values, in DN16 (section 5.1)."""
    rng = np.random.default_rng(seed)
    gain = constants.GAIN_E_PER_DN14
    electrons = np.minimum(np.maximum(signal16, 0.0) / constants.SCALE_16 * gain, constants.FULL_WELL_E)
    noisy = rng.poisson(electrons).astype(np.float32)
    noisy += rng.normal(0.0, constants.READ_NOISE_E, size=noisy.shape).astype(np.float32)
    return noisy / gain * constants.SCALE_16


def upscale_linear(
    encoded_x4: np.ndarray,
    source_linear: np.ndarray,
    out_hw: tuple[int, int],
    *,
    back_projection_iterations: int,
    source_clipped: np.ndarray | None,
) -> np.ndarray:
    """SR output (encoded, x4) to linear light on the output grid, then back-projection."""
    linear = colour.srgb_decode(np.clip(encoded_x4, 0.0, 1.0))
    linear = resample.to_grid(linear, out_hw)
    if back_projection_iterations:
        linear = resample.back_project(
            linear, source_linear, iterations=back_projection_iterations, source_clipped=source_clipped
        )
    return linear


def run_sr(encoded_uint8: np.ndarray, model: sr.Model, preset: Preset = MAX, progress=None) -> sr.Upscaled:
    encoded = np.asarray(encoded_uint8, dtype=np.float32) / 255.0
    return sr.upscale(
        encoded,
        model,
        tile=preset.tile,
        overlap=preset.overlap,
        ensemble=preset.self_ensemble,
        progress=progress,
    )


def linear_output(
    render_uint8: np.ndarray,
    out_hw: tuple[int, int],
    *,
    model: sr.Model | None,
    preset: Preset = MAX,
    upscaled: sr.Upscaled | None = None,
    progress=None,
) -> np.ndarray:
    """Linear sRGB on an output grid from a render of any size (section 5.2 branches).

    At or below the source size the render is area-downsampled (unchanged when equal). Larger grids go
    through SR x4, then area or Lanczos to the grid, then back-projection. `upscaled` reuses an SR
    result computed once for the same render (section 9.1).
    """
    source = decode(render_uint8)
    if out_hw[0] <= source.shape[0] and out_hw[1] <= source.shape[1]:
        return resample.area(source, out_hw)
    if upscaled is None:
        if model is None:
            raise RuntimeError("a larger output needs the SR model")
        upscaled = run_sr(render_uint8, model, preset, progress)
    return upscale_linear(
        upscaled.ensemble,
        source,
        out_hw,
        back_projection_iterations=preset.back_projection_iterations,
        source_clipped=resample.source_clipped(render_uint8),
    )


def linear_raw_samples(linear: np.ndarray, neutral) -> np.ndarray:
    """2K and 4K outputs: three camera-RGB samples per pixel, no added noise."""
    return quantize_clean(to_camera(linear, neutral))


def cfa_samples(linear: np.ndarray, neutral, layout: np.ndarray, *, noise_seed: int | None):
    """50 MP output: (clean, noisy) CFA samples; noisy is None when noise_seed is None."""
    signal = mosaic(to_camera(linear, neutral), layout)
    clean = quantize_clean(signal)
    noisy = None if noise_seed is None else quantize(add_noise(np.clip(signal, 0, constants.SIGNAL_16), noise_seed))
    return clean, noisy


def develop_linear_raw(samples: np.ndarray, neutral) -> np.ndarray:
    """Encoded sRGB of a LinearRaw output, the exact inverse of `to_camera` (developed TIFF)."""
    camera = stored_to_signal(samples)
    linear = colour.camera_to_linear_srgb(camera, neutral, camera_matrix(), constants.SIGNAL_16)
    return colour.srgb_encode(linear)


def develop_cfa_dng(path) -> np.ndarray:
    """Encoded sRGB of a CFA DNG through LibRaw: as-shot neutral, AHD, linear, then the sRGB transfer."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(1, 1),
            no_auto_bright=True,
            highlight_mode=rawpy.HighlightMode.Clip,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            output_bps=16,
            adjust_maximum_thr=0,
            user_flip=0,
        )
    return colour.srgb_encode(rgb.astype(np.float32) / 65535.0)
