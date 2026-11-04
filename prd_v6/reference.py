"""Benchmark construction from the reference ARW: renders R1 and R2, references at each size (6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prd_v6 import colour, constants, resample

PLANE_OF_COLOUR = {"R": 0, "G": 1, "B": 2}


class UnsuitableReference(RuntimeError):
    """The reference frame cannot support the gate (section 6.2): the run is INVALID."""


@dataclass
class Reference:
    raw_visible: np.ndarray  # uint16 (H, W), raw DN14 with black
    layout: np.ndarray  # int8 (H, W) plane index 0 R, 1 G, 2 B
    pattern: str  # 2x2 phase of the visible area, e.g. "RGGB"
    black: np.ndarray  # float32 (H, W)-broadcastable per-photosite black, DN14
    libraw: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.raw_visible.shape)

    def signal16(self) -> np.ndarray:
        """Black-subtracted reference on the 16-bit scale (50 MP reference x)."""
        return (self.raw_visible.astype(np.float32) - self.black) * constants.SCALE_16

    def clipped(self) -> np.ndarray:
        """Sensor-clipped photosites: raw >= 0.98 x 15360 before black subtraction."""
        return self.raw_visible >= constants.CLIPPED_FRACTION_OF_CLIP * constants.CLIP_14

    def oracle_neutral(self) -> np.ndarray:
        return colour.neutral_from_multipliers(self.libraw["camera_whitebalance"])


def active_window(raw) -> tuple[slice, slice]:
    """Rows and columns of the camera's active area inside rawpy's visible array.

    LibRaw reports the Sony A1 II's full 8672 x 5784 array as visible (margins 0) and gives the
    8640 x 5760 active area, the DNG default crop, as crop_*_margin and crop_width/height. Every
    array decoded from the file (raw, colours, postprocess output) is cut with this one window, and
    the noise gate's `synthraw.realism.measure_raw` measures the same window. The origin must be even
    so the CFA phase at the crop origin is the sensor's own.
    """
    from synthraw.realism import active_area

    try:
        return active_area(raw)
    except ValueError as error:
        raise UnsuitableReference(str(error)) from error


def load(arw: str | Path) -> Reference:
    import rawpy

    with rawpy.imread(str(arw)) as raw:
        window = active_window(raw)
        visible = np.array(raw.raw_image_visible[window], dtype=np.uint16, copy=True)
        colours = np.asarray(raw.raw_colors_visible)[window]
        desc = raw.color_desc.decode()
        black_per_channel = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        letters = np.array([desc[i] for i in range(len(desc))])
        layout = np.vectorize(PLANE_OF_COLOUR.get)(letters[colours]).astype(np.int8)
        pattern = "".join(letters[colours[:2, :2].reshape(-1)])
        black = black_per_channel[colours]
        info = {
            "libraw_version": ".".join(str(v) for v in rawpy.libraw_version),
            "rawpy_version": rawpy.__version__,
            "raw_size": [int(raw.sizes.raw_height), int(raw.sizes.raw_width)],
            "full_visible_size": [int(raw.sizes.iheight), int(raw.sizes.iwidth)],
            "active_origin": [int(raw.sizes.crop_top_margin), int(raw.sizes.crop_left_margin)],
            "visible_size": list(visible.shape),
            "color_desc": desc,
            "cfa_phase": pattern,
            "black_level_per_channel": [int(v) for v in raw.black_level_per_channel],
            "white_level": int(raw.white_level),
            "camera_white_level_per_channel": [int(v) for v in (raw.camera_white_level_per_channel or [])],
            "camera_whitebalance": [float(v) for v in raw.camera_whitebalance],
            "daylight_whitebalance": [float(v) for v in raw.daylight_whitebalance],
            "rgb_xyz_matrix": np.asarray(raw.rgb_xyz_matrix, dtype=float)[:3].tolist(),
        }
    if any(c not in "RGB" for c in pattern):
        raise UnsuitableReference(f"unsupported CFA {desc} phase {pattern}")
    return Reference(visible, layout, pattern, black, info)


def _postprocess(arw: str | Path, **options) -> tuple[np.ndarray, float]:
    import rawpy

    with rawpy.imread(str(arw)) as raw:
        out = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            highlight_mode=rawpy.HighlightMode.Clip,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            output_bps=16,
            adjust_maximum_thr=0,
            user_flip=0,
            **options,
        )[active_window(raw)]
        span = float(raw.white_level) - float(np.mean(raw.black_level_per_channel))
    return out, span


def render_linear(arw: str | Path) -> np.ndarray:
    """R1 step 1: camera white balance, sRGB primaries, linear, float32 in [0, 1]."""
    import rawpy

    out, _ = _postprocess(arw, use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB)
    return out.astype(np.float32) / 65535.0


def camera_rgb16(arw: str | Path) -> np.ndarray:
    """Camera-space AHD decode with unit white balance, back on the black-subtracted DN16 scale."""
    import rawpy

    out, span = _postprocess(arw, user_wb=[1.0, 1.0, 1.0, 1.0], output_color=rawpy.ColorSpace.raw)
    return out.astype(np.float32) * np.float32(span / 65535.0 * constants.SCALE_16)


def r1(linear_full: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """8-bit R1 render on a grid: area-downsample in linear light, exact sRGB, quantize."""
    return colour.to_uint8(colour.srgb_encode(resample.area(linear_full, out_hw)))


def r2(linear_full: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """R2 before JPEG: R1's linear image through the filmic curve, then the sRGB transfer."""
    return colour.to_uint8(colour.srgb_encode(colour.filmic(resample.area(linear_full, out_hw))))


def save_png(path: Path, encoded_uint8: np.ndarray, icc: bytes) -> np.ndarray:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded_uint8, "RGB").save(path, icc_profile=icc)
    return read_rgb8(path)


def save_jpeg(path: Path, encoded_uint8: np.ndarray, icc: bytes, quality: int = 90) -> np.ndarray:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded_uint8, "RGB").save(path, quality=quality, icc_profile=icc, subsampling=0)
    return read_rgb8(path)


def read_rgb8(path: Path) -> np.ndarray:
    """What the pipeline sees: the decoded file, never the in-memory render."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def binned_reference(camera16: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    return resample.area(camera16, out_hw)


def mask_50mp(reference: Reference) -> tuple[np.ndarray, dict]:
    border = constants.BORDER_PHOTOSITES
    inner = np.zeros(reference.shape, dtype=bool)
    inner[border:-border, border:-border] = True
    clipped = reference.clipped()
    fraction = float(clipped[inner].mean())
    return inner & ~clipped, {"border": border, "clipped_fraction": fraction}


def mask_binned(clipped_full: np.ndarray, out_hw: tuple[int, int], ratio: float) -> np.ndarray:
    """Output pixels with any sensor-clipped photosite in their footprint, and the border, are masked."""
    border = constants.border_pixels(ratio)
    touched = resample.area(clipped_full.astype(np.float32), out_hw) > 0
    inner = np.zeros(out_hw, dtype=bool)
    inner[border:-border, border:-border] = True
    return inner & ~touched


def check_suitable(stats: dict) -> None:
    if stats["clipped_fraction"] > constants.MAX_CLIPPED_FRACTION:
        raise UnsuitableReference(
            f"sensor-clipped photosites are {stats['clipped_fraction']:.2%} of the visible area excluding the "
            f"border (limit {constants.MAX_CLIPPED_FRACTION:.0%})"
        )
