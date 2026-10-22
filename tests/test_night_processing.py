import numpy as np
import pytest

from synthetic_raw_lab import baselines
from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import night_processing as night

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
    assert after["halo"]["fraction"] < 0.01
    assert after["banding_score"] < before["banding_score"]
    assert after["shadow_chroma_noise_codes"] < before["shadow_chroma_noise_codes"]
    clip_before = before["highlight_clipping_fraction"]["any_channel"]
    assert after["highlight_clipping_fraction"]["any_channel"] <= clip_before
    assert after["black_crush_fraction"] <= before["black_crush_fraction"] + 1e-6
    assert result.fallbacks == [] and result.fallback_fraction == 0
    protected = result.comparison["protected_color_difference_ciede2000"]
    assert protected["pixels"] > 0 and protected["chromatic_mean"] <= protected["mean"] < 2
    assert set(result.comparison["local_contrast_change"]) == set(night.REGION_NAMES)


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
