from pathlib import Path

import numpy as np
import pytest
import yaml

from synthetic_raw_lab import baselines
from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import night_processing as night

ROOT = Path(__file__).resolve().parents[1]
NIGHT_YAML = ROOT / "configs" / "night_default.yaml"
SOURCE_SIZE = (192, 108)
OUTPUT_SIZE = (500, 282)
NO_DITHER = {"dither_amplitude": 0.0}
AGGRESSIVE = {
    "texture": {"sharpening_strength": 2.0, "architecture_detail_strength": 2.0},
    "shadows": {"invented_texture_tolerance": "high", **NO_DITHER},
    "highlights": {"halo_suppression": False},
}


@pytest.fixture(scope="module")
def scene():
    encoded = night.make_night_scene(*SOURCE_SIZE)
    source = cm.srgb_to_linear(encoded.astype(np.float32) / 255)
    regions = night.segment_regions(source)
    upscale = night.sr_slot("balanced")(source, *OUTPUT_SIZE)
    return encoded, source, regions, upscale


@pytest.fixture(scope="module")
def default_result(scene):
    _, source, regions, upscale = scene
    return night.process_night(upscale.image, regions, source, seed=7, upscale=upscale)


# --------------------------------------------------------------------------- parameters


def test_default_parameters_match_the_prd_profile():
    params = night.NightParams()
    assert params.super_resolution.mode == "balanced" and params.super_resolution.tile_overlap == 64
    assert params.shadows.lift_strength == "low" and params.shadows.preserve_black_level
    assert params.shadows.banding_reduction and params.shadows.invented_texture_tolerance == "low"
    assert params.highlights.protect and params.highlights.rolloff_strength == "moderate"
    assert params.highlights.halo_suppression and params.highlights.preserve_light_color
    assert params.color.preserve_source_mood and params.color.saturation_adjustment == "minimal"
    assert params.texture.architecture_detail_strength == "moderate" and params.texture.sky_detail_strength == "minimal"
    assert params.noise.denoise_source == "conservative" and params.noise.synthetic_sensor_noise is False
    assert params.optics.chromatic_aberration is False and params.optics.vignette is False


def test_parameters_round_trip_and_accept_levels_or_numbers():
    params = night.NightParams.from_dict({"shadows": {"lift_strength": 0.3}, "highlights": {"rolloff_strength": "low"}})
    assert params.shadows.lift_strength == 0.3
    assert night.NightParams.from_dict(params.to_dict()) == params
    assert night.tone_curve(params).lift == pytest.approx(0.3)
    assert night.tone_curve(params).rolloff == night.ROLLOFF_LEVELS["low"]


@pytest.mark.parametrize(
    "values",
    [
        {"shadow": {}},
        {"shadows": {"lift": 0.1}},
        {"shadows": {"lift_strength": "extreme"}},
        {"shadows": {"lift_strength": 1.5}},
        {"shadows": {"preserve_black_level": "yes"}},
        {"highlights": {"halo_suppression_strength": -0.1}},
        {"color": {"gamut_mapping": "squash"}},
        {"noise": {"synthetic_sensor_noise": True}},
        {"optics": {"vignette": True}},
        {"failure_rule": {"block_size": 8}},
        {"super_resolution": {"mode": "aggressive"}},
        {"super_resolution": {"tile_size": 32}},
        {"super_resolution": {"tile_size": "large"}},
    ],
)
def test_invalid_parameters_are_rejected(values):
    with pytest.raises(ValueError):
        night.NightParams.from_dict(values)


def test_parameter_table_documents_every_control():
    rows = night.parameter_table()
    names = {row["parameter"] for row in rows}
    for control in (
        "shadows.lift_strength",
        "shadows.color_preservation",
        "shadows.black_point",
        "highlights.rolloff_strength",
        "optics.bloom_amount",
        "highlights.halo_suppression",
        "noise.denoise_source",
        "shadows.banding_reduction",
        "texture.architecture_detail_strength",
        "texture.atmospheric_softness",
        "color.color_temperature_k",
        "noise.grain_amount",
        "texture.sharpening_radius",
        "texture.sharpening_strength",
    ):
        assert control in names
    assert all(row["description"] and row["accepted"] and row["effect"] for row in rows)


def test_conservative_parameters_drop_detail_and_tone_changes():
    params = night.NightParams.from_dict({"shadows": {"lift_strength": "high", "black_point": 0.01}})
    safe = params.conservative()
    assert night.tone_curve(safe).lift == 0 and safe.shadows.black_point == 0
    assert safe.texture.sharpening_strength == 0 and safe.highlights.halo_suppression
    assert safe.highlights.rolloff_strength == params.highlights.rolloff_strength


# --------------------------------------------------------------------------- night profile


def _prd_profile(section: str) -> str:
    text = (ROOT / "PRD.md").read_text()
    block = text.split(f"## {section} ", 1)[1]
    return block.split("```yaml\n", 1)[1].split("```", 1)[0]


def test_night_yaml_matches_code_defaults_and_the_prd_profile(tmp_path):
    prd = tmp_path / "prd_19_1.yaml"
    prd.write_text(_prd_profile("19.1"))
    assert night.load_night_config(NIGHT_YAML) == night.NightConfig() == night.load_night_config(prd)
    config = night.load_night_config(NIGHT_YAML)
    assert config.experiment.profile == "nighttime_scenery" and config.experiment.input_category == "night"
    assert config.experiment.export_tiff16 and config.experiment.export_exr
    assert config.experiment.exr_precision == "half"
    assert config.night.super_resolution.tile_size == "auto"


def test_night_yaml_lists_every_control():
    raw = yaml.safe_load(NIGHT_YAML.read_text())
    params = night.NightParams()
    for section, values in params.to_dict().items():
        assert set(raw[section]) == set(values), section
    for key in yaml.safe_load(_prd_profile("19.1")):
        assert key in raw


def test_night_config_overrides_merge_into_sections():
    config = night.load_night_config(NIGHT_YAML, runtime_preset="quick", shadows={"lift_strength": "none"})
    assert config.experiment.effective_target_megapixels == config.experiment.quick_target_megapixels
    assert config.night.shadows.lift_strength == "none"
    assert config.night.shadows.banding_reduction is True
    flat = config.to_dict()
    assert flat["runtime_preset"] == "quick" and flat["shadows"]["lift_strength"] == "none"


def test_night_export_block_maps_onto_the_experiment_config(tmp_path):
    path = tmp_path / "n.yaml"
    path.write_text("export:\n  tiff16: false\n  exr_half: false\n")
    config = night.load_night_config(path)
    assert config.experiment.export_tiff16 is False
    assert config.experiment.export_exr and config.experiment.exr_precision == "float"


@pytest.mark.parametrize(
    "text",
    [
        "export:\n  synthetic_dng: true\n",
        "export:\n  tiff: true\n",
        "export:\n  tiff16: yes please\n",
        "export: [tiff16]\n",
        "shadow_lift: low\n",
        "shadows:\n  lift: low\n",
        "night:\n  shadows: {}\n",
    ],
)
def test_invalid_night_profiles_are_rejected(tmp_path, text):
    path = tmp_path / "n.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        night.load_night_config(path)


# --------------------------------------------------------------------------- synthetic scene and regions


def test_night_scene_is_deterministic_and_has_night_content():
    a = night.make_night_scene(*SOURCE_SIZE)
    assert a.shape == (SOURCE_SIZE[1], SOURCE_SIZE[0], 3) and a.dtype == np.uint8
    assert np.array_equal(a, night.make_night_scene(*SOURCE_SIZE))
    assert not np.array_equal(a, night.make_night_scene(*SOURCE_SIZE, seed=1))
    assert (a == 255).any(axis=2).mean() > 0.001  # clipped practical lights
    assert (a.max(axis=2) < 40).mean() > 0.5  # mostly deep shadows
    saturated = (a.max(axis=2).astype(int) - a.min(axis=2)) > 150
    assert saturated.any()  # coloured lights
    with pytest.raises(ValueError):
        night.make_night_scene(32, 32)


def test_write_night_scene(tmp_path):
    path = night.write_night_scene(tmp_path / "night.png", 128, 72)
    from PIL import Image

    assert Image.open(path).size == (128, 72)


def test_segmentation_finds_every_night_region(scene):
    encoded, _, regions, _ = scene
    fractions = regions.fractions()
    assert set(fractions) == set(night.REGION_NAMES)
    assert all(value > 0 for value in fractions.values()), fractions
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert regions.mask("clipped_highlight")[(encoded == 255).any(axis=2)].all()
    sky = regions.mask("sky_atmosphere")
    assert sky[0].mean() > 0.9 and not sky[-1].any()


def _encoded_scene(encoded: np.ndarray) -> np.ndarray:
    return cm.srgb_to_linear(np.clip(encoded, 0, 1).astype(np.float32))


def test_sky_stops_at_the_skyline_even_where_facades_are_smooth():
    rng = np.random.default_rng(3)
    height, width = 160, 240
    v = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    encoded = np.broadcast_to(0.04 + 0.05 * v, (height, width, 3)).copy()  # smooth sky gradient
    # A building with the sky on both sides: floor lines, then wide facade bands at the sky's
    # luminance carrying the mild luma and chroma noise of make_night_scene, the top band noiseless.
    facade = (slice(40, height), slice(80, 160))
    shape = (height - 40, 80)
    noise = rng.normal(0, 1.2 / 255, shape)[..., None] + rng.normal(0, 1.6 / 255, (*shape, 3))
    encoded[facade] = 0.07 + noise.astype(np.float32)
    encoded[42:62, 80:160] = 0.07
    for row in range(40, height, 22):
        encoded[row : row + 2, 80:160] = 0.12  # roofline and floor lines
    encoded[8:10, 20:22] = encoded[12:14, 200:202] = 1.0  # stars
    regions = night.segment_regions(_encoded_scene(encoded))
    sky = regions.mask("sky_atmosphere")
    assert sky[:35].mean() > 0.9
    assert sky[20:120, 21].all() and sky[25:120, 201].all()  # a star does not end the sky below it
    assert not sky[facade].any()
    assert sky[:120, :70].mean() > 0.9  # open sky beside the building is still sky

    full = night.make_night_scene(384, 216)
    sky = night.segment_regions(_encoded_scene(full / 255)).mask("sky_atmosphere")
    assert sky[0].mean() > 0.9
    assert not sky[int(0.5 * 216) :].any()  # every column is facade or road below the tallest roofline


def test_reflections_below_lights_are_specular():
    height, width = 160, 240
    encoded = np.full((height, width, 3), 0.06, dtype=np.float32)
    encoded[20:26, 57:63] = 1.0  # a clipped lamp
    encoded[70:150, 54:66] = 0.4  # its reflection: narrow, long and below the lamp
    encoded[70:150, 174:186] = 0.4  # the same streak with no light above it
    encoded[100:108, 100:150] = 0.4  # a wide, short bright patch
    regions = night.segment_regions(_encoded_scene(encoded))
    specular = regions.mask("specular_reflection")
    assert specular[75:145, 56:64].mean() > 0.9
    assert not specular[:, 150:].any()
    assert not specular[100:108, 100:150].any()

    scene_size = (384, 216)
    full = night.make_night_scene(*scene_size)
    regions = night.segment_regions(_encoded_scene(full / 255))
    specular = regions.mask("specular_reflection")
    lit = regions.mask("specular_reflection", "bright_light", "clipped_highlight")
    assert regions.fractions()["specular_reflection"] > 0.005
    road = slice(int(0.66 * scene_size[1]), scene_size[1])
    for lamp_x in (0.12, 0.33, 0.55, 0.78, 0.93):  # the street lamps of make_night_scene
        column = int(lamp_x * scene_size[0])
        # Reflections are specular; only their brightest ripples count as lights.
        assert lit[road, column - 1 : column + 2].mean() > 0.95, lamp_x
        assert specular[road, column - 4 : column + 5].any(axis=1).mean() > 0.9, lamp_x


def test_regions_can_be_overridden_and_visualised(scene):
    _, source, regions, _ = scene
    edited = regions.override("sky_atmosphere", [[0, 0, 10, 5]], note="manual")
    assert edited.mask("sky_atmosphere")[:5, :10].all()
    assert edited.overrides[-1]["region"] == "sky_atmosphere" and edited.overrides[-1]["pixels"] == 50
    assert regions.overrides == ()  # the original map is unchanged
    mask = np.zeros(regions.shape, dtype=bool)
    mask[50:60, 50:60] = True
    assert edited.override("midtone", mask).mask("midtone")[mask].all()
    with pytest.raises(ValueError):
        regions.override("sky_atmosphere", [[0, 0, 1000, 5]])
    with pytest.raises(ValueError):
        regions.override("moon", [[0, 0, 1, 1]])
    picture = regions.visualize(source)
    assert picture.shape == source.shape and picture.dtype == np.uint8
    assert regions.to_dict()["regions"] == list(night.REGION_NAMES)


def test_soft_strips_match_a_full_resize(scene):
    _, _, regions, _ = scene
    full = regions.soft_strip(*OUTPUT_SIZE, 0, OUTPUT_SIZE[1])
    part = regions.soft_strip(*OUTPUT_SIZE, 100, 140)
    for name in full:
        assert np.array_equal(full[name][100:140], part[name])
    total = sum(full[name] for name in night.REGION_NAMES)
    assert np.allclose(total, 1.0, atol=0.05)


# --------------------------------------------------------------------------- pointwise operations


def test_tone_curve_keeps_black_lifts_shadows_and_rolls_off_highlights():
    curve = night.ToneCurve(lift=0.15, rolloff=0.6)
    y = np.linspace(0, 1, 1001, dtype=np.float32)
    mapped = curve(y)
    assert mapped[0] == 0
    assert np.all(np.diff(mapped) > 0)
    assert mapped[25] > y[25]  # lifted shadow
    assert np.allclose(mapped[(y > 0.06) & (y < curve.knee)], y[(y > 0.06) & (y < curve.knee)])
    assert mapped[-1] < 1.0  # roll-off: full scale ends below full scale, nothing recovered
    assert night.ToneCurve(lift=0.15, preserve_black_level=False)(np.zeros(1, np.float32))[0] > 0
    assert night.ToneCurve(black_point=0.01)(np.array([0.005], np.float32))[0] == 0


def test_chromatic_adaptation_is_identity_at_6504_and_cools_low_temperatures():
    assert np.allclose(night.adaptation_matrix(6504.0), np.eye(3), atol=1e-6)
    grey = np.array([0.2, 0.2, 0.2], dtype=np.float32) @ night.adaptation_matrix(3200.0).T
    assert grey[2] > grey[0]


def test_gamut_map_preserves_hue_and_leaves_in_gamut_colours():
    pixels = np.array([[[1.6, 0.4, 0.1], [0.5, 0.4, 0.3], [-0.05, 0.2, 0.3]]], dtype=np.float32)
    mapped = night.gamut_map(pixels)
    assert mapped.min() >= 0 and mapped.max() <= 1
    assert np.allclose(mapped[0, 1], pixels[0, 1])
    assert np.argmax(mapped[0, 0]) == 0 and mapped[0, 0, 0] > mapped[0, 0, 1] > mapped[0, 0, 2]
    y = pixels[0, 0] @ night.LUMA_WEIGHTS
    assert mapped[0, 0] @ night.LUMA_WEIGHTS == pytest.approx(y, abs=1e-5)


# --------------------------------------------------------------------------- upscaler interface


def test_sr_slots_are_labelled_lanczos_stand_ins(scene):
    _, source, _, _ = scene
    for mode in ("conservative", "balanced"):
        result = night.sr_slot(mode)(source, *OUTPUT_SIZE)
        assert "Lanczos stand-in" in result.label and "Phase 2" in result.label
        assert mode.capitalize() in result.label
        assert result.provenance == night.ESTIMATED and result.stand_in_for
        assert np.array_equal(result.image, baselines.resample(source, *OUTPUT_SIZE, "lanczos"))
    plain = night.BaselineUpscaler("bicubic", "bicubic")(source, *OUTPUT_SIZE)
    assert plain.label == "Bicubic baseline" and plain.stand_in_for is None
    with pytest.raises(ValueError):
        night.sr_slot("aggressive")


# --------------------------------------------------------------------------- processing and metrics


def test_default_processing_improves_night_metrics(default_result):
    result = default_result
    image = result.image
    assert image.shape == (OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3) and image.dtype == np.float32
    assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
    before, after = result.metrics_before, result.metrics_after
    assert before["halo"]["fraction"] > 0.05  # Lanczos ringing around the lights
    assert after["halo"]["fraction"] < 0.8 * before["halo"]["fraction"]
    assert after["halo"]["mean_excursion_codes"] < 0.5 * before["halo"]["mean_excursion_codes"]
    assert after["banding_score"] < before["banding_score"]
    assert after["shadow_chroma_noise_codes"] < before["shadow_chroma_noise_codes"]
    clip_before = before["highlight_clipping_fraction"]["any_channel"]
    assert after["highlight_clipping_fraction"]["any_channel"] <= clip_before
    assert after["black_crush_fraction"] <= before["black_crush_fraction"] + 1e-6
    assert result.fallbacks == [] and result.fallback_fraction == 0
    protected = result.comparison["protected_color_difference_ciede2000"]
    assert protected["pixels"] > 0 and protected["chromatic_mean"] <= protected["mean"] < 2
    assert set(result.comparison["local_contrast_change"]) == set(night.REGION_NAMES)


def test_halo_metric_sees_dark_rings_inside_the_halo_clamp_range(scene, default_result):
    # The clamp keeps each channel near lights inside the 3x3 source range plus one code. A halo metric
    # measured against that same range reads exactly 0 after the clamp, whatever the image looks like.
    _, source, regions, _ = scene
    params = night.NightParams()
    height, width = default_result.image.shape[:2]
    clean = night.night_metrics(default_result.image, regions, source, params=params)["halo"]
    # Darken every ring pixel to the bottom of the clamp range: a dark ring around each light
    # that halo suppression at full strength leaves exactly as it is.
    low, high = night._Envelope(source).strip(width, height, 0, height, params)[:2]
    ring = regions.soft_strip(width, height, 0, height)["ring"] > 0.5
    ringed = default_result.image.copy()
    ringed[ring] = cm.srgb_to_linear(low[ring])
    encoded = cm.linear_to_srgb(ringed)[ring]
    assert np.all((encoded >= low[ring] - 1e-5) & (encoded <= high[ring] + 1e-5))
    halo = night.night_metrics(ringed, regions, source, params=params)["halo"]
    assert halo["fraction"] > clean["fraction"] + 0.2
    assert halo["mean_excursion_codes"] > 3 * clean["mean_excursion_codes"]


def test_stages_carry_provenance_labels(default_result):
    stages = {s["name"]: s for s in default_result.stages}
    assert all(s["provenance"] in night.PROVENANCE_LABELS for s in stages.values())
    assert stages["upscale"]["provenance"] == night.ESTIMATED
    assert "stand-in" in stages["upscale"]["parameters"]["label"]
    assert stages["dither"]["provenance"] == night.SIMULATED and stages["dither"]["enabled"]
    assert not stages["grain"]["enabled"] and not stages["bloom"]["enabled"]
    assert "not recovered" in stages["highlight_rolloff"]["note"]
    info = default_result.information
    assert set(info) == {"preserved", "interpolated", "inferred", "simulated"}
    assert info["simulated"] == "dither" and info["inferred"].startswith("none")


def test_strip_height_and_in_place_processing_do_not_change_the_result(scene):
    _, source, regions, upscale = scene
    params = night.NightParams.from_dict(AGGRESSIVE)  # exercises the failure rule across strip borders
    whole = night.process_night(upscale.image, regions, source, params, strip_rows=1000)
    assert whole.fallbacks
    strips = night.process_night(upscale.image, regions, source, params, strip_rows=40)
    assert np.allclose(strips.image, whole.image, atol=1e-5)
    assert sorted((f["x"], f["y"]) for f in strips.fallbacks) == sorted((f["x"], f["y"]) for f in whole.fallbacks)
    buffer = upscale.image.copy()
    in_place = night.process_night(buffer, regions, source, params, strip_rows=40, out=buffer)
    assert in_place.image is buffer
    assert np.allclose(buffer, whole.image, atol=1e-5)


def test_failure_rule_falls_back_on_halos_and_texture(scene):
    _, source, regions, upscale = scene
    params = night.NightParams.from_dict(AGGRESSIVE)
    guarded = night.process_night(upscale.image, regions, source, params)
    reasons = {reason for record in guarded.fallbacks for reason in record["reasons"]}
    assert {"halo", "excess_texture"} <= reasons
    record = guarded.fallbacks[0]
    assert {"x", "y", "width", "height", "reasons", "regions"} <= set(record)
    assert 0 < guarded.fallback_fraction <= 1
    unguarded = night.process_night(upscale.image, regions, source, params.replace(failure_rule={"enabled": False}))
    assert unguarded.fallbacks == []
    assert guarded.metrics_after["halo"]["fraction"] < unguarded.metrics_after["halo"]["fraction"]
    assert (
        guarded.metrics_after["local_contrast_codes"]["low_luminance_textured"]
        < unguarded.metrics_after["local_contrast_codes"]["low_luminance_textured"]
    )


def test_failure_rule_falls_back_on_black_crush(scene):
    _, source, regions, upscale = scene
    params = night.NightParams.from_dict({"shadows": {"black_point": 0.02, **NO_DITHER}})
    guarded = night.process_night(upscale.image, regions, source, params)
    assert any("black_crush" in record["reasons"] for record in guarded.fallbacks)
    unguarded = night.process_night(upscale.image, regions, source, params.replace(failure_rule={"enabled": False}))
    assert guarded.metrics_after["black_crush_fraction"] < 0.5 * unguarded.metrics_after["black_crush_fraction"]


def test_failure_rule_falls_back_on_banding(scene, monkeypatch):
    _, source, regions, upscale = scene
    process = night._process

    def posterising(linear, strip, params, scale):
        out = process(linear, strip, params, scale)
        if params.texture.sharpening_strength > 0:  # the candidate, not the conservative result
            encoded = cm.linear_to_srgb(np.clip(out, 0, 1))
            out = cm.srgb_to_linear(np.round(encoded * 40) / 40)
        return out

    monkeypatch.setattr(night, "_process", posterising)
    result = night.process_night(upscale.image, regions, source, night.NightParams.from_dict({"shadows": NO_DITHER}))
    banded = [r for r in result.fallbacks if "banding" in r["reasons"]]
    assert banded
    assert any({"sky_atmosphere", "smooth_gradient"} & set(r["regions"]) for r in banded)


def test_debanding_smooths_quantised_ramps_without_flattening_texture_into_plateaus():
    # Left: an 8-bit quantised dark ramp. Right: a noisy facade at the synthetic scene's noise level.
    # Both are labelled sky, as the old segmentation labelled facades.
    height, width = 120, 160
    rows = np.linspace(20, 32, height, dtype=np.float32)[:, None] * np.ones((1, width), np.float32)
    encoded = np.round(rows) / 255
    noise = np.random.default_rng(3).normal(0, 1.4, (height, width // 2)).astype(np.float32)
    encoded[:, width // 2 :] = np.round(rows[:, width // 2 :] + noise) / 255
    source = np.repeat(cm.srgb_to_linear(encoded)[..., None], 3, axis=2)
    regions = night.RegionMap(np.full((height, width), night.REGION_NAMES.index("sky_atmosphere")))
    upscale = night.sr_slot("balanced")(source, 393, 295)
    result = night.process_night(upscale.image, regions, source, night.NightParams.from_dict({"shadows": NO_DITHER}))
    assert result.fallbacks == []

    margin, split = 40, upscale.image.shape[1] // 2

    def score(image, part):
        flat, ramp = night.banding_map(image, upscale.image.shape[1] / width)
        ys = slice(margin, -margin)
        xs = slice(margin, split - margin) if part == "ramp" else slice(split + margin, -margin)
        return flat[ys, xs].sum() / ramp[ys, xs].sum()

    assert score(upscale.image, "ramp") > 0.3 and score(result.image, "ramp") < 0.5 * score(upscale.image, "ramp")
    assert score(result.image, "noise") <= score(upscale.image, "noise") + 0.02


def test_banding_maps_separate_steps_from_ramps_and_dither():
    width = 400
    ramp = np.tile(np.linspace(0.1, 0.1 + 20 * night._CODE, width, dtype=np.float32), (32, 1))
    steps = np.round(ramp * 255) / 255
    dithered = steps + np.random.default_rng(0).uniform(-0.5, 0.5, steps.shape).astype(np.float32) * night._CODE

    def score(y):
        flat, is_ramp = night._banding_maps(y, 1.0)
        return flat.sum() / is_ramp.sum()

    assert score(ramp) < 0.05
    assert score(steps) > 0.6
    assert score(dithered) < 0.2


def test_night_metrics_measure_clipping_and_ringing_halos():
    source = np.full((64, 64, 3), 0.004, dtype=np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    source[(yy - 32) ** 2 + (xx - 32) ** 2 < 25] = 1.0
    regions = night.segment_regions(source)
    assert regions.mask("clipped_highlight")[32, 32]
    own = night.night_metrics(source, regions, source)
    assert own["halo"]["fraction"] == 0
    assert own["highlight_clipping_fraction"]["all_channels"] == pytest.approx(
        (source >= 1).all(axis=2).mean(), abs=1e-9
    )
    upscaled = baselines.resample(source, 160, 160, "lanczos")
    ringing = night.night_metrics(upscaled, regions, source)
    assert ringing["halo"]["fraction"] > 0.05 and ringing["halo"]["mean_excursion_codes"] > 0


def test_highlight_rolloff_preserves_light_colour():
    source = np.full((48, 48, 3), 0.01, dtype=np.float32)
    source[18:30, 18:30] = [1.0, 0.45, 0.12]  # saturated orange light at full scale
    regions = night.segment_regions(source)
    image = baselines.resample(source, 96, 96, "lanczos")
    base = {"shadows": NO_DITHER}
    kept = night.process_night(image, regions, source, night.NightParams.from_dict(base)).image[48, 48]
    bleached = night.process_night(
        image, regions, source, night.NightParams.from_dict({**base, "highlights": {"preserve_light_color": False}})
    ).image[48, 48]
    ratio = 0.45 / 1.0
    assert kept[1] / kept[0] == pytest.approx(ratio, abs=0.01)
    assert bleached[1] / bleached[0] > kept[1] / kept[0] + 0.01


def test_grain_and_bloom_are_simulated_and_seeded(scene):
    _, source, regions, upscale = scene
    params = night.NightParams.from_dict(
        {"noise": {"grain_amount": 0.01}, "optics": {"bloom_amount": 0.05}, "shadows": NO_DITHER}
    )
    a = night.process_night(upscale.image, regions, source, params, seed=1)
    b = night.process_night(upscale.image, regions, source, params, seed=1)
    c = night.process_night(upscale.image, regions, source, params, seed=2)
    assert np.array_equal(a.image, b.image) and not np.array_equal(a.image, c.image)
    assert "grain" in a.information["simulated"] and "bloom" in a.information["simulated"]
    stages = {s["name"]: s for s in a.stages}
    assert stages["grain"]["enabled"] and stages["grain"]["provenance"] == night.SIMULATED
    assert stages["bloom"]["enabled"] and stages["bloom"]["classification"] == night.SIMULATION


def test_colour_temperature_and_shadow_neutrality(scene):
    _, source, regions, upscale = scene
    base = {"shadows": NO_DITHER}
    neutral = night.process_night(upscale.image, regions, source, night.NightParams.from_dict(base)).image
    cooled = night.process_night(
        upscale.image, regions, source, night.NightParams.from_dict({**base, "color": {"color_temperature_k": 3200.0}})
    ).image
    assert (cooled[..., 2].mean() / cooled[..., 0].mean()) > (neutral[..., 2].mean() / neutral[..., 0].mean())
    grey = night.process_night(
        upscale.image,
        regions,
        source,
        night.NightParams.from_dict({"shadows": {**NO_DITHER, "color_preservation": 0.0}}),
    )
    keep = night.process_night(upscale.image, regions, source, night.NightParams.from_dict(base))
    assert grey.comparison["shadow_chroma_mean_shift_codes"] > keep.comparison["shadow_chroma_mean_shift_codes"]


def test_fallback_map_marks_failed_blocks(scene):
    _, source, regions, upscale = scene
    result = night.process_night(upscale.image, regions, source, night.NightParams.from_dict(AGGRESSIVE))
    picture = night.fallback_map(result, *OUTPUT_SIZE)
    assert picture.shape == (OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3)
    first = result.fallbacks[0]
    assert tuple(picture[first["y"], first["x"]]) != (24, 24, 24)


def test_process_night_rejects_mismatched_inputs(scene):
    _, source, regions, upscale = scene
    with pytest.raises(ValueError):
        night.process_night(upscale.image.astype(np.float64), regions, source)
    with pytest.raises(ValueError):
        night.process_night(upscale.image, regions, source[:50])
