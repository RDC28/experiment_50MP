"""Identical evaluation crops across baselines (PRD sections 9.2, 26.1 and 27.3).

Crops are defined once in source pixels and mapped to every output with the
same per-axis scale, so each method is judged on exactly the same scene region.
Automatic selection takes the centre plus the most detailed non-overlapping
windows (highest mean gradient magnitude); manual crops replace it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .quality_metrics import luma

_GUTTER = 8
_HEADER = 30
_DIFF_GAIN = 8


@dataclass(frozen=True)
class Crop:
    name: str
    x: int
    y: int
    width: int
    height: int

    def output_box(self, source_size: tuple[int, int], output_size: tuple[int, int]) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of this crop in an output of ``output_size`` (width, height)."""
        sx = output_size[0] / source_size[0]
        sy = output_size[1] / source_size[1]
        return (
            round(self.x * sx),
            round(self.y * sy),
            round((self.x + self.width) * sx),
            round((self.y + self.height) * sy),
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "width": self.width, "height": self.height}


def select_crops(encoded: np.ndarray, size: int, count: int, manual=()) -> list[Crop]:
    """Choose evaluation crops on a source image (sRGB-encoded float, (H, W, 3))."""
    height, width = encoded.shape[:2]
    if manual:
        crops = []
        for index, (x, y, w, h) in enumerate(manual, start=1):
            if x + w > width or y + h > height:
                raise ValueError(f"manual crop {[x, y, w, h]} exceeds the {width}x{height} source")
            crops.append(Crop(f"manual_{index}", int(x), int(y), int(w), int(h)))
        return crops

    size = min(size, width, height)
    crops = [Crop("center", (width - size) // 2, (height - size) // 2, size, size)]
    if count == 1:
        return crops

    y_luma = luma(encoded)
    gradient = np.hypot(cv2.Sobel(y_luma, cv2.CV_32F, 1, 0), cv2.Sobel(y_luma, cv2.CV_32F, 0, 1))
    integral = cv2.integral(gradient.astype(np.float64))
    stride = max(size // 2, 1)
    xs = sorted(set(range(0, width - size + 1, stride)) | {width - size})
    ys = sorted(set(range(0, height - size + 1, stride)) | {height - size})
    candidates = []
    for y in ys:
        for x in xs:
            energy = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
            candidates.append((-energy, y, x))
    candidates.sort()
    for _, y, x in candidates:
        if len(crops) >= count:
            break
        if all(_disjoint(x, y, size, c) for c in crops):
            crops.append(Crop(f"detail_{len(crops)}", x, y, size, size))
    return crops


def crop_panel(source_tile: np.ndarray, tiles: dict[str, np.ndarray]) -> np.ndarray:
    """Side-by-side uint8 panel: source (nearest-neighbour), each method, and their difference."""
    names = list(tiles)
    tile_h, tile_w = tiles[names[0]].shape[:2]
    columns = [("source (nearest)", cv2.resize(source_tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST))]
    columns += [(name, tiles[name]) for name in names]
    if len(names) >= 2:
        a, b = names[0], names[1]
        diff = np.abs(tiles[a].astype(np.int16) - tiles[b].astype(np.int16)).max(axis=2)
        heat = cv2.applyColorMap(np.clip(diff * _DIFF_GAIN, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        columns.append((f"|{a} - {b}| x{_DIFF_GAIN}", cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)))

    width = len(columns) * tile_w + (len(columns) - 1) * _GUTTER
    panel = np.full((tile_h + _HEADER, width, 3), 24, dtype=np.uint8)
    for index, (label, tile) in enumerate(columns):
        x = index * (tile_w + _GUTTER)
        panel[_HEADER:, x : x + tile_w] = tile
        scale = 0.5 if tile_w >= 200 else 0.35
        cv2.putText(
            panel, label, (x + 6, _HEADER - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA
        )
    return panel


def _disjoint(x: int, y: int, size: int, crop: Crop) -> bool:
    return x + size <= crop.x or crop.x + crop.width <= x or y + size <= crop.y or crop.y + crop.height <= y
