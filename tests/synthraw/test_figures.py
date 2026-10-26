"""PRD_v3 validation figures written from a tiny end-to-end validation run."""

import dataclasses
import shutil
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.figures import FIGURE_NAMES, ValidationFigureWriter, _centre_crop, _subsample
from synthraw.pipeline import run_pipeline
from synthraw.validation_runner import ValidationRunner


@pytest.fixture(scope="module")
def validated(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("figures_run")
    rows, cols = np.mgrid[0:20, 0:48]
    pixels = np.stack([255 * cols / 47, 255 * rows / 19, 128 + 100 * np.sin(cols / 6)], axis=-1)
    source = tmp_path / "source.png"
    Image.fromarray(np.clip(np.round(pixels), 0, 255).astype(np.uint8)).save(source)
    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    root = run_pipeline(
        source, config, tmp_path / "runs", timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=96
    ).root
    return ValidationRunner().run(root)


def _image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def test_every_prd_figure_is_written_bounded_and_not_blank(validated):
    written = ValidationFigureWriter().write(validated)
    assert tuple(written) == FIGURE_NAMES
    for name, path in written.items():
        assert path == validated.root / "validation" / name and path.is_file()
        pixels = _image(path)
        # Downsampled for size: a figure stays a few hundred KB, never a full-resolution frame.
        assert max(pixels.shape[:2]) <= 2400 and path.stat().st_size < 1_000_000, name
        assert pixels.std() > 5, f"{name} looks blank"


def test_missing_products_render_labelled_placeholders(validated, tmp_path):
    root = shutil.copytree(validated.root, tmp_path / "copy")
    bare = dataclasses.replace(
        validated, root=root, noise_fit=None, roundtrip=None, demosaic_thumbs={}, editability_thumbs={}
    )
    written = ValidationFigureWriter().write(bare)
    assert set(written) == set(FIGURE_NAMES)
    assert all(path.is_file() and path.stat().st_size > 0 for path in written.values())


def test_subsampling_keeps_the_cfa_phase_and_bounds_size():
    mosaic = np.arange(1000 * 600).reshape(1000, 600)
    tile = _subsample(mosaic, 128, multiple=2)
    assert max(tile.shape) <= 128
    step = tile[0, 1] - tile[0, 0]
    assert step % 2 == 0, "an odd stride would mix CFA colours in each plane"
    crop = _centre_crop(mosaic, 16)
    assert crop.shape == (16, 16)
    top, left = divmod(int(crop[0, 0]), 600)
    assert top % 2 == 0 and left % 2 == 0


def test_roundtrip_figure_is_tight_and_labels_supported_versus_generated(validated, tmp_path):
    root = shutil.copytree(validated.root, tmp_path / "copy")
    path = ValidationFigureWriter().write(dataclasses.replace(validated, root=root))["roundtrip_difference.png"]
    pixels = _image(path).astype(int)
    height, width = pixels.shape[:2]
    # Four 40x96 panels in one row: no empty band above or below the images.
    assert height / width < 0.2, (height, width)
    rows_with_ink = np.flatnonzero((np.abs(pixels - 255).sum(axis=-1) > 30).any(axis=1))
    assert rows_with_ink[0] < 0.1 * height and rows_with_ink[-1] > 0.9 * height
    # The supported-region panel uses two distinct legend colours, not a thin black outline.
    for colour in ((0x3A, 0x78, 0xC2), (0xE8, 0x83, 0x3A)):
        assert (np.abs(pixels - colour).sum(axis=-1) < 12).sum() > 200, colour
