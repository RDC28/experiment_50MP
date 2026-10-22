"""Phase 3 night run (night_pipeline) end to end on a small synthetic night scene."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from synthetic_raw_lab import baselines, night_pipeline, pipeline
from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import night_processing as night

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"runtime_preset": "quick", "quick_target_megapixels": 0.25, "crop_size": 40, "review_max_edge": 256}


def small_config(**overrides) -> night.NightConfig:
    return night.load_night_config(ROOT / "configs" / "night_default.yaml", **{**SMALL, **overrides})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    project = tmp_path_factory.mktemp("night_project")
    source = night.write_night_scene(project / "staging" / "night_scene.png", 192, 108)
    overrides = [{"region": "sky_atmosphere", "boxes": [[0, 0, 16, 4]], "note": "test override"}]
    result = night_pipeline.run_night(
        source, small_config(), project, region_overrides=overrides, notebook_version="test", log=lambda _: None
    )
    return project, result


def test_run_completes_with_validated_night_masters(run):
    project, result = run
    manifest = result.manifest
    assert manifest["status"] == "complete"
    assert manifest["schema"] == night_pipeline.SCHEMA
    assert result.masters_dir == project / "outputs" / "night" / result.run_id
    exports = manifest["final"]["exports"]
    assert set(exports) == {"exr", "tiff16"}
    assert all(e["valid"] for e in exports.values())
    assert exports["exr"]["path"].endswith("_lin_rec709_half.exr") and exports["exr"]["sample_type"] == "float16"
    target = manifest["target_resolution"]
    assert (exports["tiff16"]["width"], exports["tiff16"]["height"]) == (target["width"], target["height"])
    for export in exports.values():
        sidecar = json.loads(Path(export["path"] + ".json").read_text())
        provenance = sidecar["provenance"]
        assert provenance["information"] == manifest["night"]["information"]
        assert set(provenance["stage_provenance"].values()) <= set(night.PROVENANCE_LABELS)
    assert set(manifest["output_checksums"]) >= {Path(e["path"]).name for e in exports.values()}
    assert json.loads(result.manifest_path.read_text())["run_id"] == result.run_id
    assert all(stage["error"] is None for stage in manifest["stages"])


def test_comparison_slots_are_labelled_lanczos_stand_ins(run):
    _, result = run
    slots = result.manifest["comparisons"]
    assert list(slots) == ["lanczos", "bicubic", "conservative_sr", "balanced_sr"]
    for mode in ("conservative", "balanced"):
        slot = slots[f"{mode}_sr"]
        assert slot["label"] == f"Lanczos stand-in for {mode.capitalize()} SR (Phase 2 SR backend not merged)"
        assert slot["stand_in_for"] == f"{mode.capitalize()} SR"
        assert slot["working_pixel_sha256"] == slots["lanczos"]["working_pixel_sha256"]
    assert slots["bicubic"]["working_pixel_sha256"] != slots["lanczos"]["working_pixel_sha256"]
    assert result.manifest["night"]["input"]["slot"] == "balanced_sr"


def test_night_metrics_are_reported_before_and_after(run):
    _, result = run
    manifest = result.manifest
    record = manifest["night"]
    before, after = record["metrics_before"], record["metrics_after"]
    # The night input is the balanced SR slot; only the strip layout (summation order) differs.
    slot = manifest["comparisons"]["balanced_sr"]["night_metrics"]
    assert before["banding_score"] == slot["banding_score"]
    assert before["highlight_clipping_fraction"] == slot["highlight_clipping_fraction"]
    assert before["halo"]["fraction"] == slot["halo"]["fraction"]
    assert before["shadow_chroma_noise_codes"] == pytest.approx(slot["shadow_chroma_noise_codes"], rel=1e-6)
    assert after["banding_score"] < before["banding_score"]
    assert after["highlight_clipping_fraction"]["any_channel"] <= before["highlight_clipping_fraction"]["any_channel"]
    assert after["shadow_chroma_noise_codes"] < before["shadow_chroma_noise_codes"]
    assert after["halo"]["fraction"] <= before["halo"]["fraction"]
    header, rows = night_pipeline.metrics_table(manifest)
    assert header == [
        "Metric",
        "source",
        "lanczos",
        "bicubic",
        "Lanczos as cons. SR",
        "Lanczos as bal. SR",
        "night input (before)",
        "final night (after)",
    ]
    labels = [row[0] for row in rows]
    for label in ("Highlight clipping, any channel", "Gradient banding score", "Shadow chroma noise (8-bit codes)"):
        assert label in labels
    assert all(len(row) == len(header) and "-" not in row[1:] for row in rows if "Halo" not in row[0])


def test_section_19_3_figures_are_written(run):
    _, result = run
    figures = result.reports_dir / "figures"
    names = {p.name for p in figures.iterdir()}
    assert {"regions.png", "fallback_map.png", "banding_diagnostics.png", "halo_profile.png"} <= names
    assert {"source_preview.jpg", "lanczos_preview.jpg", "night_preview.jpg", "delta_e_night.png"} <= names
    crops = [panel["crop"] for panel in result.manifest["crop_panels"]]
    assert crops == ["highlight", "shadow", "reflection", "gradient"]
    assert all(Path(panel["path"]).exists() for panel in result.manifest["crop_panels"])


def test_report_has_every_section_and_the_metrics_table(run):
    _, result = run
    report = result.report_path.read_text()
    for heading in (
        "Run summary",
        "Scientific limitations",
        "Input validation",
        "Scene analysis: luminance regions",
        "Processing log",
        "Visual results",
        "Quantitative results",
        "Information provenance",
        "Conclusions",
        "Provenance record",
    ):
        assert heading in report
    assert "Night metrics before and after night processing" in report
    assert "| Gradient banding score |" in report and "| final night (after) |" in report
    assert "Lanczos stand-in for Balanced SR" in report
    assert "Failure rule (PRD 12.6)" in report
    assert "sky_atmosphere at boxes [[0, 0, 16, 4]] (64 px) test override" in report
    for figure in ("regions.png", "banding_diagnostics.png", "halo_profile.png", "crop_shadow.png"):
        assert f"(figures/{figure})" in report
    assert "—" not in report


def test_region_overrides_are_recorded(run):
    _, result = run
    overrides = result.manifest["regions"]["overrides"]
    assert overrides == [
        {"region": "sky_atmosphere", "where": "boxes [[0, 0, 16, 4]]", "pixels": 64, "note": "test override"}
    ]
    assert result.regions.mask("sky_atmosphere")[:4, :16].all()


def test_failed_stage_keeps_outputs_and_names_the_stage(tmp_path, monkeypatch):
    source = night.write_night_scene(tmp_path / "night.png", 128, 72)

    def broken(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(night, "process_night", broken)
    with pytest.raises(pipeline.StageError) as error:
        night_pipeline.run_night(source, small_config(), tmp_path, log=lambda _: None)
    assert error.value.stage == "night_processing"
    manifest = json.loads(error.value.manifest_path.read_text())
    assert manifest["status"] == "failed in stage night_processing"
    assert set(manifest["comparisons"]) == {"lanczos", "bicubic", "conservative_sr", "balanced_sr"}
    report = (error.value.manifest_path.parent / "report.md").read_text()
    assert "simulated failure" in report
    assert (error.value.manifest_path.parent / "figures" / "regions.png").exists()


def test_light_centre_picks_a_lamp_core_not_the_neon_tube():
    encoded = night.make_night_scene(320, 180)
    source = (encoded.astype(np.float32) / 255) ** 2.2
    regions = night.segment_regions(source)
    x, y = night_pipeline.light_centre(regions)
    assert regions.labels[y, x] == night.REGION_NAMES.index("clipped_highlight")
    assert abs(y - (0.62 - 0.08) * 180) <= 2  # the street-lamp row of make_night_scene


def test_halo_profile_shows_the_adjacent_source_row_that_upscales_blend_in():
    # A lit window ends one source row above the lamp centre (row 10), as in the full-size synthetic run, and
    # the output row through the lamp lies at source y 9.91: Lanczos sits between the two source rows.
    linear = np.full((20, 40, 3), 0.05, dtype=np.float32)
    linear[:10, :20] = 0.7
    source = SimpleNamespace(linear=linear, width=40, height=20)
    width, height = 98, 49
    profiles = night_pipeline._ProfileLines({"highlight": (0, 0, width, height)}, source, (width, height), (10, 10))
    profiles.add("lanczos", baselines.resample(linear, width, height, "lanczos"))
    rows = profiles.halo_rows
    assert (rows["output_row"], rows["nearest"], rows["adjacent"]) == (25, 10, 9)
    assert rows["source_y"] == pytest.approx(25.5 * 20 / 49 - 0.5)
    lines = profiles.lines["highlight"]
    nearest, adjacent, lanczos = lines["source"], lines["source_adjacent"], lines["lanczos"]
    under_window = slice(10, 40)
    assert np.all(lanczos[under_window] > nearest[under_window] + 10)
    assert np.all(lanczos[under_window] < adjacent[under_window])
    assert np.allclose(lanczos[65:85], nearest[65:85], atol=1)
    # Column mapping: the window's right edge (source x 20, output 49) falls in the same place in both curves.
    step = int(np.argmax(adjacent < adjacent[0] - 1))
    crossing = int(np.argmax(lanczos < (lanczos[under_window].mean() + nearest[70]) / 2))
    assert step == 49 and abs(crossing - step) <= 1


def test_night_crops_cover_each_category_and_manual_crops_replace_them():
    encoded = night.make_night_scene(320, 180)
    regions = night.segment_regions((encoded.astype(np.float32) / 255) ** 2.2)
    crops = night_pipeline.select_night_crops(regions, 48)
    assert [c.name for c in crops] == ["highlight", "shadow", "reflection", "gradient"]
    for crop, names in zip(crops, night_pipeline.CROP_CATEGORIES.values(), strict=True):
        assert 0 <= crop.x <= 320 - 48 and 0 <= crop.y <= 180 - 48
        assert regions.mask(*names)[crop.y : crop.y + 48, crop.x : crop.x + 48].any()
    manual = night_pipeline.select_night_crops(regions, 48, [[10, 20, 30, 40]])
    assert [(c.name, c.x, c.y, c.width, c.height) for c in manual] == [("manual_1", 10, 20, 30, 40)]
    with pytest.raises(ValueError, match="exceeds"):
        night_pipeline.select_night_crops(regions, 48, [[300, 0, 40, 40]])


def test_night_input_follows_the_super_resolution_settings():
    assert night_pipeline.night_input(small_config())[0] == "balanced_sr"
    conservative = small_config(super_resolution={"mode": "conservative"})
    assert night_pipeline.night_input(conservative)[0] == "conservative_sr"
    key, upscaler = night_pipeline.night_input(small_config(super_resolution={"enabled": False}))
    assert key == "lanczos" and upscaler.stand_in_for is None


def _ink_rows(band: np.ndarray) -> int:
    """Rows of a header band that carry label text (bright pixels)."""
    return int((band.max(axis=(1, 2)) > 128).sum())


def test_crop_panel_labels_are_legible_on_wide_tiles():
    from synthetic_raw_lab import comparison

    tile = np.full((628, 628, 3), 40, np.uint8)
    tiles = {"lanczos": tile, "final night": tile + 2}
    panel = night_pipeline.crop_panel(tile[:256, :256], tiles, ("lanczos", "final night"))
    width = 4 * 628 + 3 * 8
    assert panel.shape[1] == width
    header = panel[: panel.shape[0] - 628, :628]
    assert (panel[-628:, :628] == 40).all()  # the source column sits under its label
    # The shared Phase 1 panel draws ~10 px labels (14 rows with parentheses); the night panel's are 25 rows.
    assert _ink_rows(header) >= max(20, 1.6 * _ink_rows(comparison.crop_panel(tile[:256, :256], tiles)[:30, :628]))
    # A label wider than its column wraps instead of spilling into the next column.
    long = {"a very long method label that cannot fit on one line of a narrow tile": tile[:64, :64]}
    narrow = night_pipeline.crop_panel(tile[:32, :32], {**long, "b": tile[:64, :64]}, tuple(long) + ("b",))
    assert narrow.shape[0] - 64 > 2 * 16
    assert (narrow[: narrow.shape[0] - 64, 64:72] == 24).all()  # the gutter stays clear


def test_shadow_crop_panel_adds_a_display_only_gain_row(run):
    dark = np.full((64, 64, 3), 8, np.uint8)
    tiles = {"lanczos": dark, "final night": dark}
    plain = night_pipeline.crop_panel(dark[:32, :32], tiles, ("lanczos", "final night"))
    gained = night_pipeline.crop_panel(dark[:32, :32], tiles, ("lanczos", "final night"), display_gain_ev=3.0)
    assert (gained[: plain.shape[0]] == plain).all()  # the 100% row is unchanged
    lower = gained[-64:]
    # Code 8 is 0.00243 linear; x8 is 0.0194 linear, i.e. code 38 in sRGB, for every image column.
    for column in range(3):
        assert (lower[:, column * 72 : column * 72 + 64] == 38).all()
    assert (lower[:, 3 * 72 :] == 24).all()  # the difference column is not gained
    banner = gained[plain.shape[0] : -64]
    assert banner.shape[0] > 0 and (banner[..., 0] > 128).any()  # an amber banner labels the row

    _, result = run
    panels = {p["crop"]: p for p in result.manifest["crop_panels"]}
    assert panels["shadow"]["display_gain_ev"] == 3.0
    assert all("display_gain_ev" not in p for name, p in panels.items() if name != "shadow")
    from PIL import Image

    sizes = {name: Image.open(p["path"]).size for name, p in panels.items()}
    header = sizes["highlight"][1] - (panels["highlight"]["box"][3] - panels["highlight"]["box"][1])
    shadow_tile = panels["shadow"]["box"][3] - panels["shadow"]["box"][1]
    assert sizes["shadow"][1] > header + 2 * shadow_tile  # header, 100% row, banner and the gain row
    assert "+3 EV display gain (display only)" in result.report_path.read_text()


def test_fallback_figure_has_a_legend_and_marks_blocks_over_the_source(tmp_path):
    from PIL import Image

    canvas = np.full((90, 160, 3), night.FALLBACK_BACKGROUND, np.uint8)
    canvas[10:20, 10:30] = night.FALLBACK_COLORS["banding"]
    records = [{"reasons": ["banding"], "x": 10, "y": 10, "width": 20, "height": 10}]
    source = np.full((45, 80, 3), 0.2, np.float32)
    path = night_pipeline.fallback_figure(canvas, source, records, 0.0139, tmp_path / "fallback_map.png")
    picture = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)
    colours = {tuple(int(v) for v in c) for c in np.unique(picture, axis=0)}
    # Every reason has a legend patch, even with no blocks, so the colours are always explained.
    assert set(night.FALLBACK_COLORS.values()) <= colours
    # The source shows through, dimmed, outside the blocks instead of a flat background.
    grey = round(255 * (0.08 + 0.4 * float(cm.linear_to_srgb(np.float32(0.2)))))
    assert any(abs(c[0] - grey) <= 1 and c[0] == c[1] == c[2] for c in colours)
