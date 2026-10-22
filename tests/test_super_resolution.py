"""Tiling, blending, seam detection, memory fallback and hallucination controls of Phase 2 SR.

The network is replaced by small local numpy models so these tests are fast and need neither PyTorch
nor weights; ``test_sr_models.py`` covers the real network.
"""

import cv2
import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import super_resolution as sr
from synthetic_raw_lab.sr_models import DeviceOutOfMemory
from synthetic_raw_lab.test_images import make_test_image

SRC_W, SRC_H = 300, 200
OUT_W, OUT_H = 737, 491  # non-integer scale (about 2.46), like a 4K source to 50 MP


class LocalModel:
    """x4 'network' with a 5x5 receptive field and zero padding, so tile edges differ from the interior."""

    scale = 4

    def __init__(self, device="cpu"):
        self.device = device
        self.calls = []

    def __call__(self, encoded):
        self.calls.append(encoded.shape[:2])
        blurred = np.stack([ndimage.uniform_filter(encoded[..., c], 5, mode="constant") for c in range(3)], -1)
        sharpened = np.clip(encoded + 0.5 * (encoded - blurred), 0, 1)
        h, w = encoded.shape[:2]
        return cv2.resize(sharpened, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    def fallback(self):
        return None


class ContextModel(LocalModel):
    """Output depends on the whole tile (its mean), like the global context real networks use."""

    def __call__(self, encoded):
        return np.clip(super().__call__(encoded) + 0.3 * (encoded.mean() - 0.4), 0, 1)


class PointModel(ContextModel):
    """Pixel replication plus the tile-mean term: no border effects of its own."""

    def __call__(self, encoded):
        return np.clip(np.repeat(np.repeat(encoded, 4, 0), 4, 1) + 0.3 * (encoded.mean() - 0.4), 0, 1)


class WindowModel(LocalModel):
    """Like window attention: mixes each 8 x 8 block counted from the tile origin (SwinIR's window grid)."""

    alignment = 8

    def __call__(self, encoded):
        h, w = encoded.shape[:2]
        padded = np.pad(encoded, ((0, -h % 8), (0, -w % 8), (0, 0)), mode="reflect")
        blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8, 3).mean(axis=(1, 3))
        means = np.repeat(np.repeat(blocks, 8, 0), 8, 1)[:h, :w]
        return super().__call__(np.clip(encoded + 0.5 * (encoded - means), 0, 1))


class OutOfMemoryModel(LocalModel):
    """Raises DeviceOutOfMemory for tiles larger than ``limit`` pixels per side."""

    def __init__(self, limit, device="gpu", fallback_model=None):
        super().__init__(device)
        self.limit = limit
        self.fallback_model = fallback_model
        self.releases = 0

    def release_cache(self):
        self.releases += 1

    def __call__(self, encoded):
        if max(encoded.shape[:2]) > self.limit:
            raise DeviceOutOfMemory(f"{self.device}: simulated out of memory")
        return super().__call__(encoded)

    def fallback(self):
        return self.fallback_model


@pytest.fixture(scope="module")
def source():
    return cm.srgb_to_linear(make_test_image(SRC_W, SRC_H).astype(np.float32) / 255)


def untiled(source, model, width=OUT_W, height=OUT_H):
    native = cm.srgb_to_linear(model(cm.linear_to_srgb(np.clip(source, 0, 1))))
    box = (0.0, 0.0, float(native.shape[1]), float(native.shape[0]))
    return sr.resample_box(native, width, height, box, Image.Resampling.LANCZOS)


def run(source, model, **kwargs):
    kwargs = {"tile_size": 96, "tile_overlap": 32, "log": lambda _: None, **kwargs}
    return sr.super_resolve(source, OUT_W, OUT_H, model, **kwargs)


# --------------------------------------------------------------------------- geometry


@pytest.mark.parametrize(
    ("length", "tile", "overlap", "align"),
    [(200, 96, 32, 1), (300, 64, 16, 1), (97, 96, 32, 1), (1000, 256, 64, 1), (1000, 256, 64, 8), (2160, 256, 64, 8)],
)
def test_tiles_cover_the_axis_with_at_least_the_overlap(length, tile, overlap, align):
    spans = sr.tile_spans(length, tile, overlap, align)
    assert spans[0][0] == 0 and spans[-1][1] == length
    assert all(start % align == 0 for start, _ in spans)
    assert all(tile <= end - start < tile + align for start, end in spans)
    assert all(a[1] - b[0] >= overlap for a, b in zip(spans, spans[1:], strict=False))


def test_single_tile_when_the_axis_fits():
    assert sr.tile_spans(80, 96, 32) == [(0, 80)]
    with pytest.raises(ValueError):
        sr.tile_spans(200, 64, 64)


@pytest.mark.parametrize(("length", "tile", "overlap", "align"), [(200, 96, 32, 1), (300, 64, 16, 1), (130, 96, 40, 8)])
def test_blend_weights_are_a_partition_of_unity_that_ignores_tile_edges(length, tile, overlap, align):
    spans = sr.tile_spans(length, tile, overlap, align)
    centers = (np.arange(length * 4) + 0.5) / 4
    weights = sr.axis_weights(centers, spans, length, overlap)
    assert np.allclose(weights.sum(axis=0), 1, atol=1e-6)
    assert weights.min() >= 0
    for index, (start, end) in enumerate(spans):
        assert not weights[index, (centers < start) | (centers >= end)].any()
        if index:  # the quarter of the overlap next to an interior edge carries no weight
            assert not weights[index, (centers >= start) & (centers < start + overlap / 4)].any()


# --------------------------------------------------------------------------- tiling correctness


@pytest.mark.parametrize(("tile", "overlap"), [(96, 32), (128, 64), (64, 16), (512, 64)])
def test_tiled_result_equals_untiled_result(source, tile, overlap):
    """Tile-boundary invariance: tiling must not change the image when tiles see enough context."""
    out, report = run(source, LocalModel(), tile_size=tile, tile_overlap=overlap)
    assert out.shape == (OUT_H, OUT_W, 3) and out.dtype == np.float32
    assert np.abs(out - untiled(source, LocalModel())).max() < 1e-5
    rows, cols = report.grid
    assert (
        report.tiles
        == rows * cols
        == len(sr.tile_spans(SRC_H, tile, overlap)) * len(sr.tile_spans(SRC_W, tile, overlap))
    )
    assert (report.to_dict()["overlap_disagreement"]["max_abs"] or 0) < 0.5  # below one 8-bit code value


def test_tiles_follow_the_window_grid_of_attention_models(source):
    """Window-attention networks only agree across tiles when the tiles share the window grid."""
    model = WindowModel()
    reference = untiled(source, model)
    aligned, report = run(source, model, tile_size=128, tile_overlap=64)
    assert np.abs(aligned - reference).max() < 1e-5
    assert report.to_dict()["overlap_disagreement"]["low_frequency_p99_abs_max"] < 0.05

    class Unaligned(WindowModel):
        alignment = 1

    unaligned, report = run(source, Unaligned(), tile_size=128, tile_overlap=64)
    assert np.abs(unaligned - reference).max() > 0.01
    assert report.to_dict()["overlap_disagreement"]["p99_abs_max"] > 1


def test_flat_image_stays_flat(source):
    flat = np.full((SRC_H, SRC_W, 3), 0.18, dtype=np.float32)
    out, _ = run(flat, PointModel())
    assert np.ptp(out) < 1e-5


@pytest.fixture(scope="module")
def scene():
    """A smooth photographic-like scene (gradients and soft texture), where seams are easy to see."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:SRC_H, 0:SRC_W] / np.array([SRC_H, SRC_W])[:, None, None]
    texture = ndimage.gaussian_filter(rng.normal(0, 0.04, (SRC_H, SRC_W)), 1.5)
    encoded = np.stack([0.2 + 0.5 * x, 0.3 + 0.4 * y, 0.6 - 0.3 * x * y], -1) + texture[..., None]
    return cm.srgb_to_linear(np.clip(encoded, 0, 1).astype(np.float32))


def test_seams_are_blended_and_detected(scene):
    """A context-dependent model disagrees across tiles: blending hides it, hard cuts show it."""
    out, report = run(scene, ContextModel())
    disagreement = report.to_dict()["overlap_disagreement"]
    assert disagreement["low_frequency_p99_abs_max"] > 2  # the tiles really disagree
    assert sr.seam_scores(out, report.boundaries, scene)["max_score"] < 0.3

    # Negative control: the same tiles pasted without overlap or blending leave visible seams.
    encoded = cm.linear_to_srgb(np.clip(scene, 0, 1))
    model, native = ContextModel(), np.empty((SRC_H * 4, SRC_W * 4, 3), np.float32)
    for y in range(0, SRC_H, 100):
        for x in range(0, SRC_W, 100):
            native[y * 4 : (y + 100) * 4, x * 4 : (x + 100) * 4] = model(encoded[y : y + 100, x : x + 100])
    native = cm.srgb_to_linear(native)
    cut = sr.resample_box(native, OUT_W, OUT_H, (0.0, 0.0, SRC_W * 4.0, SRC_H * 4.0), Image.Resampling.LANCZOS)
    lines = [{"axis": "x", "position_source_px": 100}, {"axis": "x", "position_source_px": 200}]
    lines += [{"axis": "y", "position_source_px": 100}]
    assert sr.seam_scores(cut, lines, scene)["max_score"] > 1.5  # in 8-bit code values
    assert sr.seam_scores(out, lines, scene)["max_score"] < 0.3


def test_seam_score_grows_with_an_injected_step(scene):
    source = scene
    output = untiled(source, LocalModel())
    boundary = [{"axis": "x", "position_source_px": 175.0}]
    column = round(175.0 * OUT_W / SRC_W)
    scores = []
    for step in (0, 2, 8):  # code values
        encoded = cm.linear_to_srgb(np.clip(output, 0, 1))
        encoded[:, column:] += step / 255
        scores.append(sr.seam_scores(cm.srgb_to_linear(np.clip(encoded, 0, 1)), boundary, source)["max_score"])
    assert scores[0] < 0.2
    # on smooth content a k-code seam scores about k; texture crossing the line lowers it somewhat
    assert 1.0 < scores[1] <= 2.2 and 5.0 < scores[2] <= 8.5


# --------------------------------------------------------------------------- memory fallback


def test_out_of_memory_tiles_are_split_and_the_result_is_unchanged(source):
    model = OutOfMemoryModel(limit=48)
    out, report = run(source, model, tile_size=128, tile_overlap=32)
    assert np.abs(out - untiled(source, LocalModel())).max() < 1e-5
    assert [e["kind"] for e in report.fallbacks] == ["split", "split"]  # 128 -> 64 -> 32
    assert model.releases == 2  # the device cache is emptied before every retry
    assert max(max(shape) for shape in model.calls) <= 48
    assert report.final_device == "gpu"


def test_out_of_memory_at_the_minimum_tile_falls_back_to_the_cpu(source):
    cpu = LocalModel("cpu")
    out, report = run(source, OutOfMemoryModel(limit=0, fallback_model=cpu), tile_size=64, tile_overlap=16)
    assert np.abs(out - untiled(source, LocalModel())).max() < 1e-5
    assert report.fallbacks[-1]["kind"] == "device" and "cpu" in report.fallbacks[-1]["message"]
    assert report.final_device == "cpu" and cpu.calls


def test_out_of_memory_without_a_fallback_device_raises(source):
    with pytest.raises(DeviceOutOfMemory):
        run(source, OutOfMemoryModel(limit=0), tile_size=64, tile_overlap=16)


@pytest.mark.parametrize("kwargs", [{"tile_overlap": 8}, {"tile_size": 64, "tile_overlap": 48}, {"detail_strength": 2}])
def test_invalid_tiling_parameters_are_rejected(source, kwargs):
    with pytest.raises(ValueError):
        run(source, LocalModel(), **kwargs)


# --------------------------------------------------------------------------- hallucination controls


def test_detail_strength_blends_towards_lanczos(source):
    full, _ = run(source, LocalModel(), detail_strength=1.0)
    none, _ = run(source, LocalModel(), detail_strength=0.0)
    half, _ = run(source, LocalModel(), detail_strength=0.5)
    lanczos = sr.resample_box(source, OUT_W, OUT_H, (0.0, 0.0, float(SRC_W), float(SRC_H)), Image.Resampling.LANCZOS)
    assert np.abs(none - lanczos).max() < 1e-5
    assert np.abs(half - (full + lanczos) / 2).max() < 1e-5


def test_projection_restores_low_frequency_agreement(source):
    from synthetic_raw_lab.baselines import downsample_area

    drifted, _ = run(source, LocalModel())
    drifted *= np.float32(1.1)  # a colour/tone drift the network might introduce
    original = drifted.copy()
    stats = sr.project_to_source(drifted, source, 1.0)
    assert stats["mean_abs_residual_linear_after"] < 0.3 * stats["mean_abs_residual_linear_before"]
    reduced = downsample_area(drifted, SRC_W, SRC_H)
    assert np.abs(reduced - source).mean() < 0.3 * np.abs(downsample_area(original, SRC_W, SRC_H) - source).mean()
    untouched = original.copy()
    assert sr.project_to_source(untouched, source, 0.0)["mean_abs_residual_linear_after"] > 0
    assert np.array_equal(untouched, original)


def test_interpolation_difference_is_zero_for_lanczos_and_positive_for_sr(source):
    lanczos = sr.resample_box(source, OUT_W, OUT_H, (0.0, 0.0, float(SRC_W), float(SRC_H)), Image.Resampling.LANCZOS)
    diff, stats = sr.interpolation_difference(lanczos, source, chunk_rows=100)
    assert diff.shape == (OUT_H, OUT_W) and diff.dtype == np.float16
    assert stats["mean"] < 0.01 and stats["p99"] == 0
    out, _ = run(source, LocalModel())
    _, stats = sr.interpolation_difference(out, source, chunk_rows=100)
    assert stats["mean"] > 0.5 and stats["p99"] >= 1


def test_presets_order_from_conservative_to_aggressive():
    modes = [sr.PRESETS[m] for m in sr.MODES]
    assert [p.mode for p in modes] == list(sr.MODES)
    assert [p.network for p in modes] == ["fidelity", "perceptual", "perceptual"]
    assert [p.consistency_strength for p in modes] == sorted((p.consistency_strength for p in modes), reverse=True)
    assert modes[2].warning and not modes[0].warning and not modes[1].warning
