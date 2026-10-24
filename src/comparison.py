"""Identical evaluation crops across baselines (PRD sections 9.2, 26.1 and 27.3).

Crops are defined once in source pixels and mapped to every output with the
same per-axis scale, so each method is judged on exactly the same scene region.
Automatic selection always starts with the main subject (``subjects.analyse``):
the primary face (eyes, then nose and mouth when the face is larger than a
crop), else the centre of interest, else the centre. The remaining crops are
the most detailed windows (highest mean gradient magnitude), spread apart where
the image allows, and at least one is kept whenever more than one crop is
requested. Manual crops replace the automatic selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .quality_metrics import luma
from .subjects import SubjectAnalysis, analyse

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
    label: str = ""

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
        return {
            "name": self.name,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def select_crops(
    encoded: np.ndarray, size: int, count: int, manual=(), subjects: SubjectAnalysis | None = None
) -> list[Crop]:
    """Choose evaluation crops on a source image (sRGB-encoded float, (H, W, 3)).

    ``subjects`` is the analysis of the same image; it is computed here when not given.
    """
    height, width = encoded.shape[:2]
    if manual:
        crops = []
        for index, (x, y, w, h) in enumerate(manual, start=1):
            if x + w > width or y + h > height:
                raise ValueError(f"manual crop {[x, y, w, h]} exceeds the {width}x{height} source")
            crops.append(Crop(f"manual_{index}", int(x), int(y), int(w), int(h), "manual crop from the configuration"))
        return crops

    size = min(size, width, height)
    subjects = subjects or analyse(encoded)
    crops = _subject_crops(subjects, size, width, height, limit=1 if count == 1 else count - 1)
    if len(crops) >= count:
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
    # First pass keeps half a crop of space around every crop so detail crops show different parts of the
    # scene; the second pass only requires them not to overlap.
    for gap in (size // 2, 0):
        for _, y, x in candidates:
            if len(crops) >= count:
                return crops
            if all(_disjoint(x, y, size, c, gap) for c in crops):
                index = sum(c.name.startswith("detail_") for c in crops) + 1
                crops.append(Crop(f"detail_{index}", x, y, size, size, f"detail: most detailed region {index}"))
    return crops


def _subject_crops(subjects: SubjectAnalysis, size: int, width: int, height: int, limit: int) -> list[Crop]:
    """Crops of the main subject: the primary face, else the centre of interest, else the centre."""
    face = subjects.primary_face
    if face is not None:
        cx = face.x + face.width / 2
        if max(face.width, face.height) <= size:
            return [_centred("face", "primary face", cx, face.y + face.height / 2, size, width, height)]
        # A face larger than a crop: the eyes carry identity (PRD 11.1, 26.2), then nose and mouth below them.
        eyes = _centred("face_eyes", "primary face: eyes", cx, face.y + 0.4 * face.height, size, width, height)
        crops = [eyes]
        if limit >= 2 and face.height >= 1.5 * size and eyes.y + 2 * size <= height:
            crops.append(Crop("face_nose_mouth", eyes.x, eyes.y + size, size, size, "primary face: nose and mouth"))
        return crops
    if subjects.center_of_interest is not None:
        x, y = subjects.center_of_interest
        return [_centred("subject", "centre of interest (saliency)", x, y, size, width, height)]
    return [Crop("center", (width - size) // 2, (height - size) // 2, size, size, "image centre")]


def _centred(name: str, label: str, cx: float, cy: float, size: int, width: int, height: int) -> Crop:
    x = min(max(0, round(cx - size / 2)), width - size)
    y = min(max(0, round(cy - size / 2)), height - size)
    return Crop(name, x, y, size, size, label)


def crop_panel(
    source_tile: np.ndarray,
    tiles: dict[str, np.ndarray],
    difference: tuple[str, str] | None = None,
    *,
    source_label: str = "source (nearest)",
) -> np.ndarray:
    """Side-by-side uint8 panel: source (nearest-neighbour), each method, and the difference of two methods.

    ``difference`` names the pair to compare; by default the first two methods. ``source_label`` titles the
    first column.
    """
    names = list(tiles)
    tile_h, tile_w = tiles[names[0]].shape[:2]
    columns = [(source_label, cv2.resize(source_tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST))]
    columns += [(name, tiles[name]) for name in names]
    if difference or len(names) >= 2:
        a, b = difference or (names[0], names[1])
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


def _disjoint(x: int, y: int, size: int, crop: Crop, gap: int = 0) -> bool:
    return (
        x + size + gap <= crop.x
        or crop.x + crop.width + gap <= x
        or y + size + gap <= crop.y
        or crop.y + crop.height + gap <= y
    )
