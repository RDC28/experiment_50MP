"""PRD 18.3 review packaging for Notebook 01, without identity claims.

Native face crops retain their actual source/output sampling. Contact sheets use
nearest-neighbour fitting with the scale labelled; they are review aids, not
masters. Names describe roles rather than claiming that a reduced quick run is
4K or 50 MP. All returned records are JSON-serializable manifest entries.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import color_management as cm
from .comparison import Crop
from .exporters import write_review_image
from .face_preservation import FaceTierConfig, face_tier
from .human_pipeline import STAND_IN_DESCRIPTION, _validate_image
from .human_reconstruction import HumanReconstruction
from .region_masks import mask_overlay

_CELL = 320
_GAP = 8


def _display(linear):
    return np.round(np.clip(cm.linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8)


def _sheet(tiles, title, *, columns=3):
    """Compact row-major sheet; each crop fits the same apparent-size cell.

    Nearest-neighbour display preserves visible sampling differences. Native
    crop files remain available for 100% review. Row heights follow the actual
    fitted images rather than reserving square space for landscape previews.
    """
    if not tiles:
        tiles = [("No observed faces", np.full((32, 32, 3), 24, np.uint8))]
    columns = min(columns, len(tiles))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def text_height(text):
        box = probe.multiline_textbbox((0, 0), text)
        return box[3] + _GAP * 2

    width = columns * _CELL + (columns - 1) * _GAP
    title_lines = []
    for line in title.split("\n"):
        current = ""
        for word in line.split():
            candidate = f"{current} {word}".strip()
            if current and probe.textlength(candidate) > width - 2 * _GAP:
                title_lines.append(current)
                current = word
            else:
                current = candidate
        title_lines.append(current)
    title = "\n".join(title_lines)
    title_height = text_height(title)
    rows = []
    for start in range(0, len(tiles), columns):
        row = []
        for label, tile in tiles[start : start + columns]:
            h, w = tile.shape[:2]
            scale = min(_CELL / w, _CELL / h)
            text = f"{label}\n{w}x{h} pixels; display x{scale:.3g} nearest"
            preview = Image.fromarray(tile).resize(
                (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.NEAREST
            )
            row.append((text, preview))
        band = max(text_height(text) for text, _ in row)
        rows.append((row, band, max(preview.height for _, preview in row)))
    height = title_height + sum(band + h + _GAP for _, band, h in rows)
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((_GAP, _GAP), title, fill="white")
    y = title_height
    for row, band, h in rows:
        for column, (text, preview) in enumerate(row):
            x = column * (_CELL + _GAP)
            draw.multiline_text((x + _GAP, y + _GAP), text, fill="white")
            canvas.paste(preview, (x + (_CELL - preview.width) // 2, y + band))
        y += band + h + _GAP
    return np.asarray(canvas)


def write_human_comparisons(
    source: np.ndarray,
    result: HumanReconstruction,
    figures_dir: str | Path,
    *,
    tier_config: FaceTierConfig | None = None,
) -> dict:
    """Write mandatory comparisons, per-region overlays and native face crops.

    Pass the reconstruction's tier configuration so background selections agree
    with identity reporting. Background comparisons select non-primary tier-3
    annotations. Empty selections are explicitly labelled, never silently lost.
    """
    _validate_image(source)
    size = (source.shape[1], source.shape[0])
    for face in result.detections.faces:
        face.validate_bounds(*size)
    arrays = {"original": source, **result.candidates, "final": result.final}
    for image in arrays.values():
        _validate_image(image)
    target_shape = result.final.shape
    if any(result.candidates[key].shape != target_shape for key in ("lanczos", "conservative", "balanced")):
        raise ValueError("comparison candidates must share final dimensions")
    root = Path(figures_dir)
    root.mkdir(parents=True, exist_ok=True)
    figures = {}
    crops = []
    small_tiles = []
    face_tiles = {}
    labels = {
        "original": "Original source face crops",
        "lanczos": "Lanczos interpolated face crops",
        "conservative": "Conservative SR stand-in crops",
        "balanced": "Balanced SR stand-in face crops",
        "final": "Final identity-preserving face crops",
    }

    def save(name, pixels, description):
        path = write_review_image(root / f"{name}.png", pixels)
        return {"path": str(path), "classification": "Packaging", "description": description}

    for method, image in arrays.items():
        tiles = []
        for index, face in enumerate(result.detections.faces):
            crop = Crop(f"face_{index}", face.x, face.y, face.width, face.height)
            box = crop.output_box(size, (image.shape[1], image.shape[0]))
            x0, y0, x1, y1 = box
            tile = _display(image[y0:y1, x0:x1])
            tier = face_tier(face, tier_config)
            label = f"Face {index}; source tier {tier}; {method}"
            entry = save(f"face_{index}_{method}", tile, label)
            crops.append({**entry, "face_index": index, "tier": tier, "method": method, "box": list(box)})
            tiles.append((label, tile))
            face_tiles[index, method] = (label, tile)
            if tier == 3 and index != result.detections.primary_index:
                small_tiles.append((label, tile))
        figures[f"{method}_face_crops"] = save(
            f"{method}_face_crops",
            _sheet(tiles, labels[method] + "\nSource observations; no identity proof"),
            labels[method],
        )
    methods = ("original", "lanczos", "conservative", "balanced", "final")
    figures["face_comparison_grid"] = save(
        "face_comparison_grid",
        _sheet(
            [face_tiles[index, method] for index in range(len(result.detections.faces)) for method in methods],
            "Face comparison: rows = faces; columns = original, Lanczos, conservative, balanced, final\n"
            "Matched apparent subject size; nearest display; observations are not identity proof",
            columns=5,
        ),
        "Matched apparent-size face grid; native crops retained for 100% review",
    )
    overview = []
    for method in ("original", "lanczos", "final"):
        image = arrays[method]
        # Resize the linear image before encoding to avoid a second full-size display buffer.
        h, w = image.shape[:2]
        scale = min(1, _CELL / max(h, w))
        preview = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        overview.append((f"{method}; native {w}x{h}; fitted overview", _display(preview)))
    figures["crowd_overview"] = save(
        "crowd_overview", _sheet(overview, "Full crowd overview"), "Source, Lanczos and final"
    )
    figures["small_background_faces"] = save(
        "small_background_faces",
        _sheet(small_tiles, "Small background-face comparisons\nSource tier 3; no identity claims"),
        "All candidates and final for non-primary tier-3 observations",
    )
    overlays = [(name, mask_overlay(source, mask)) for name, mask in result.masks.masks.items()]
    figures["mask_overlay"] = save(
        "mask_overlay",
        _sheet(overlays, "Estimated region masks\nMagenta = weight; not segmentation truth"),
        "Source-scale estimated masks; magenta opacity is proportional to weight",
    )
    return {"figures": figures, "face_crops": crops, "notice": STAND_IN_DESCRIPTION}
