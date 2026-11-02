"""Driver step 4: build the benchmark, make the three outputs, score them and write verdict.json.

Sections 6 and 7.2: R1 and R2 renders at 4K and 2K, references at 2K, 4K and 50 MP, the frozen
pipeline, the gate, hard sanity and sensitivity checks, the error-budget ladder, the ablation, the
reported-only variants and the integrity gates. Every output is scored against the reference at its
own size, in black-subtracted DN16.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from prd_v6 import colour, constants, dng, metrics, pipeline, reference, resample, sr

LADDER_NAMES = {
    "noise": "noise",
    "illuminant": "illuminant estimation",
    "resolution": "resolution and SR",
    "remaining": "remaining inverse ISP error (matrix, demosaic, quantization)",
}


@dataclass
class Config:
    run_dir: Path
    arw: Path
    model: sr.Model
    preset: pipeline.Preset = pipeline.MAX
    require_dng_validate: bool = True
    dng_validate_binary: str | None = None
    expect_camera_size: bool = True
    log: Callable[[str], None] = print
    timings: dict = field(default_factory=dict)


@contextmanager
def _timed(cfg: Config, name: str):
    cfg.log(f"[benchmark] {name}")
    start = time.monotonic()
    yield
    cfg.timings[name] = round(time.monotonic() - start, 2)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=True))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


# ---------------------------------------------------------------- noise model and sanity checks


def _variance_fn(binned):
    return lambda x, p: metrics.noise_variance(x, binned=binned, plane=p)


def _variance_array(x: np.ndarray, layout, binned) -> np.ndarray:
    if layout is None:
        return np.stack([metrics.noise_variance(x[..., p], binned=binned, plane=p) for p in range(3)], axis=-1)
    out = np.empty(x.shape, dtype=np.float64)
    for p in range(3):
        sel = layout == p
        out[sel] = metrics.noise_variance(x[sel], binned=binned, plane=p)
    return out


def _blur(values: np.ndarray, layout) -> np.ndarray:
    out = np.array(values, dtype=np.float32, copy=True)
    if layout is None:
        for p in range(3):
            out[..., p] = cv2.GaussianBlur(out[..., p], (0, 0), 1.0)
        return out
    for dy in (0, 1):
        for dx in (0, 1):
            out[dy::2, dx::2] = cv2.GaussianBlur(np.ascontiguousarray(out[dy::2, dx::2]), (0, 0), 1.0)
    return out


def perturbations(noise_floor: np.ndarray, layout, mask) -> dict[str, np.ndarray]:
    """Section 6.4 sensitivity perturbations of the noise-floor realization."""
    constant = np.array(noise_floor, dtype=np.float32, copy=True)
    for p in range(3):
        if layout is None:
            constant[..., p] = float(noise_floor[..., p][mask].mean())
        else:
            sel = layout == p
            constant[sel] = float(noise_floor[sel & mask].mean())
    gain = np.array(noise_floor, dtype=np.float32, copy=True)
    if layout is None:
        gain[..., 0] *= 1.1
    else:
        gain[layout == 0] *= 1.1
    return {
        "constant_mean": constant,
        "shift": np.roll(noise_floor, 2 if layout is not None else 1, axis=1),
        "r_gain_plus_10pct": gain,
        "gaussian_blur": _blur(noise_floor, layout),
    }


def sanity_checks(x: np.ndarray, layout, mask, binned, seed: int) -> dict:
    variance = _variance_fn(binned)
    own = metrics.score(x, x, layout, mask, variance=variance, candidate_noisy=False)
    self_ok = all(abs(own["planes"][p]["A"] - 1.0) < 1e-9 and own["planes"][p]["P2"] == 1.0 for p in metrics.PLANES)
    rng = np.random.default_rng(seed)
    noise_floor = (x + rng.standard_normal(x.shape) * np.sqrt(_variance_array(x, layout, binned))).astype(np.float32)
    floor = metrics.score(x, noise_floor, layout, mask, variance=variance, candidate_noisy=True)
    floor_ok = all(
        floor["planes"][p]["A"] >= constants.NOISE_FLOOR_P1_MIN
        and floor["planes"][p]["P2"] >= constants.NOISE_FLOOR_P2_MIN
        for p in metrics.PLANES
    )
    sensitivity = {}
    for name, candidate in perturbations(noise_floor, layout, mask).items():
        scored = metrics.score(x, candidate, layout, mask, variance=variance, candidate_noisy=True)
        planes = ("R",) if name == "r_gain_plus_10pct" else metrics.PLANES
        drop = floor["min_A"] - scored["min_A"]
        p2_lower = all(scored["planes"][p]["P2"] < floor["planes"][p]["P2"] for p in planes)
        sensitivity[name] = {
            "min_A": scored["min_A"],
            "min_A_drop": drop,
            "P2": {p: scored["planes"][p]["P2"] for p in metrics.PLANES},
            "passed": bool(drop >= constants.SENSITIVITY_MIN_DROP and p2_lower),
        }
    return {
        "hard": {
            "self": {"min_A": own["min_A"], "min_P2": own["min_P2"], "passed": bool(self_ok)},
            "noise_floor": {
                "planes": {p: {k: floor["planes"][p][k] for k in ("A", "P2")} for p in metrics.PLANES},
                "k": floor["k"],
                "passed": bool(floor_ok),
            },
        },
        "sensitivity": sensitivity,
        "hard_passed": bool(self_ok and floor_ok),
    }


# ---------------------------------------------------------------- error maps


def _plane_images(rel: np.ndarray, layout) -> dict[str, np.ndarray]:
    if layout is None:
        return {name: rel[..., p] for p, name in enumerate(metrics.PLANES)}
    out = {}
    for p, name in enumerate(metrics.PLANES):
        subs = [rel[dy::2, dx::2] for dy in (0, 1) for dx in (0, 1) if layout[dy, dx] == p]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # masked pixels are NaN in both green sites
            out[name] = np.nanmean(np.stack(subs), axis=0) if len(subs) > 1 else subs[0]
    return out


def save_error_maps(folder: Path, rel: np.ndarray, layout, small_side: int = 1200) -> list[str]:
    from matplotlib import colormaps
    from PIL import Image

    cmap = colormaps["coolwarm"]
    written = []
    for name, plane in _plane_images(rel, layout).items():
        nan = np.isnan(plane)
        rgb = (cmap((np.nan_to_num(plane) + 0.2) / 0.4)[..., :3] * 255).astype(np.uint8)
        rgb[nan] = 128
        full = folder / f"error_map_{name}.png"
        Image.fromarray(rgb).save(full)
        scale = small_side / max(rgb.shape[:2])
        small = (
            rgb
            if scale >= 1
            else cv2.resize(
                rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA
            )
        )
        Image.fromarray(small).save(folder / f"error_map_{name}_small.png")
        written += [full.name, f"error_map_{name}_small.png"]
    return written


def save_preview(path: Path, encoded: np.ndarray, side: int = 2048) -> None:
    from PIL import Image

    scale = side / max(encoded.shape[:2])
    image = (
        encoded
        if scale >= 1
        else resample.area(encoded, (round(encoded.shape[0] * scale), round(encoded.shape[1] * scale)))
    )
    Image.fromarray(colour.to_uint8(image)).save(path, quality=92, icc_profile=colour.srgb_icc_profile())


def save_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


# ---------------------------------------------------------------- one delivered output


def _candidate_from_dng(path: Path, layout) -> np.ndarray:
    """Section 6.2: the DNG's stored samples minus its own black level (rawpy for the black)."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        if layout is not None:
            visible = raw.raw_image_visible.astype(np.float32)
            return visible - black[np.asarray(raw.raw_colors_visible)]
    return dng.read_samples(path).astype(np.float32) - black[:3]


def _write_output(folder: Path, name: str, samples, neutral, *, cfa: str | None, clean=None) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "dng": dng.write_dng(
            folder / f"exp50v6_{name}.dng",
            samples,
            cfa_pattern=cfa,
            color_matrix=pipeline.camera_matrix(),
            as_shot_neutral=neutral,
        )
    }
    if clean is not None:
        files["clean_dng"] = dng.write_dng(
            folder / "clean.dng", clean, cfa_pattern=cfa, color_matrix=pipeline.camera_matrix(), as_shot_neutral=neutral
        )
    encoded = pipeline.develop_cfa_dng(files["dng"]) if cfa else pipeline.develop_linear_raw(samples, neutral)
    files["tiff"] = dng.write_tiff16(folder / f"exp50v6_{name}.tif", encoded, colour.srgb_icc_profile())
    save_preview(folder / "preview.jpg", encoded)
    return files


def _integrity(files: dict, hw, cfa, cfg: Config) -> dict:
    from prd_v5.dng_validate import run_dng_validate

    record = {"libraw": {}, "dng_validate": {}, "tiff": dng.tiff16_check(files["tiff"], hw)}
    for key in ("dng", "clean_dng"):
        if key in files:
            record["libraw"][key] = dng.libraw_check(files[key], expected_hw=hw, cfa_pattern=cfa)
            record["dng_validate"][key] = run_dng_validate(files[key], cfg.dng_validate_binary)
    problems = [f"LibRaw {k}: {v['problems']}" for k, v in record["libraw"].items() if v["status"] != "passed"]
    if record["tiff"]["status"] != "passed":
        problems.append(f"TIFF read-back: {record['tiff']}")
    for key, result in record["dng_validate"].items():
        if result["status"] == "failed" or (cfg.require_dng_validate and result["status"] != "passed"):
            problems.append(
                f"dng_validate {key}: {result['status']} {result.get('reason') or result.get('error_lines')}"
            )
    record["problems"] = problems
    record["passed"] = not problems
    return record


def _ladder_steps(rungs: dict) -> dict:
    """Per-plane A cost of each step between rungs; their sum is 1 - A(L0)."""
    a = {rung: rungs[rung]["planes"] for rung in rungs}
    steps = {}
    for p in metrics.PLANES:
        if "L4" in a:
            steps[p] = {
                "noise": a["L1"][p]["A"] - a["L0"][p]["A"],
                "illuminant": a["L2"][p]["A"] - a["L1"][p]["A"],
                "resolution": a["L4"][p]["A"] - a["L2"][p]["A"],
                "remaining": 1.0 - a["L4"][p]["A"],
            }
        else:
            steps[p] = {"illuminant": a["L2"][p]["A"] - a["L0"][p]["A"], "remaining": 1.0 - a["L2"][p]["A"]}
    return steps


def _contributors(steps: dict, plane: str) -> list[tuple[str, float]]:
    return sorted(((name, cost) for name, cost in steps[plane].items()), key=lambda item: -item[1])


def _brief(score: dict) -> dict:
    keep = ("k", "k_ev", "min_A", "min_P2", "passes", "worst_plane")
    return {
        **{key: score[key] for key in keep},
        "planes": {p: {k: score["planes"][p][k] for k in ("A", "P2")} for p in metrics.PLANES},
    }


# ---------------------------------------------------------------- step 4


def run(cfg: Config) -> dict:
    root = Path(cfg.run_dir) / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    preset = cfg.preset
    icc = colour.srgb_icc_profile()
    cam_xyz = pipeline.camera_matrix()
    prior = colour.prior_neutral(cam_xyz)
    daylight = colour.daylight_neutral(cam_xyz)
    invalid: list[str] = []
    warnings: list[str] = []

    with _timed(cfg, "reference decode"):
        ref = reference.load(cfg.arw)
        visible = ref.shape
        if cfg.expect_camera_size and visible != constants.VISIBLE_HW:
            raise reference.UnsuitableReference(f"visible area {visible} != {constants.VISIBLE_HW}")
        sizes = {name: constants.output_shape(visible, name) for name in constants.OUTPUT_ORDER}
        oracle = ref.oracle_neutral()
        mask50, clip_stats = reference.mask_50mp(ref)
        reference.check_suitable(clip_stats)
        clipped_full = ref.clipped()
        x50 = ref.signal16()

    with _timed(cfg, "R1 and R2 renders"):
        linear_full = reference.render_linear(cfg.arw)
        renders = {}
        for name in ("4k", "2k"):
            renders[("R1", name)] = reference.save_png(
                root / "renders" / f"r1_{name}.png", reference.r1(linear_full, sizes[name]), icc
            )
            renders[("R2", name)] = reference.save_jpeg(
                root / "renders" / f"r2_{name}.jpg", reference.r2(linear_full, sizes[name]), icc
            )
        r1_full = colour.to_uint8(colour.srgb_encode(linear_full))
        del linear_full

    with _timed(cfg, "binned references"):
        camera16 = reference.camera_rgb16(cfg.arw)
        refs = {"50mp": (x50, ref.layout, mask50, (1.0, 1.0, 1.0))}
        for name in ("4k", "2k"):
            ratio = constants.OUTPUT_RATIOS[name]
            refs[name] = (
                reference.binned_reference(camera16, sizes[name]),
                None,
                reference.mask_binned(clipped_full, sizes[name], ratio),
                metrics.binned_counts(ratio),
            )
        del camera16

    camera_model = {
        "xyz_to_camera_d65": cam_xyz.tolist(),
        "libraw_rgb_xyz_matrix_for_r1": ref.libraw["rgb_xyz_matrix"],
        "matrix_max_abs_difference": float(np.max(np.abs(np.asarray(ref.libraw["rgb_xyz_matrix"]) - cam_xyz))),
        "prior_cct_k": constants.PRIOR_CCT_K,
        "prior_neutral": prior.tolist(),
        "daylight_neutral": daylight.tolist(),
        "oracle_neutral_diagnostic_only": oracle.tolist(),
        "gain_e_per_dn14": constants.GAIN_E_PER_DN14,
        "read_noise_e": constants.READ_NOISE_E,
        "full_well_e": constants.FULL_WELL_E,
        "black_16": constants.BLACK_16,
        "white_16": constants.WHITE_16,
        "reference": ref.libraw,
    }
    _write_json(Path(cfg.run_dir) / "camera_model.json", camera_model)
    if camera_model["matrix_max_abs_difference"] > 1e-3:
        warnings.append(
            "LibRaw's matrix for R1 differs from the measured matrix by "
            f"{camera_model['matrix_max_abs_difference']:.4f}; both recorded in camera_model.json"
        )

    results: dict = {}
    clipped_src = {name: resample.source_clipped(renders[("R1", name)]) for name in ("4k", "2k")}

    # 2K and 4K: invert the render at its own size (section 5.4), no SR, no noise.
    for name in ("2k", "4k"):
        with _timed(cfg, f"{name} output"):
            x, layout, mask, binned = refs[name]
            variance = _variance_fn(binned)
            folder = root / name
            linear = pipeline.decode(renders[("R1", name)])
            samples = pipeline.linear_raw_samples(linear, prior)
            files = _write_output(folder, name, samples, prior, cfa=None)
            y = _candidate_from_dng(files["dng"], None)
            extra = {"source_clipped": clipped_src[name]}
            gated = metrics.score(
                x, y, None, mask, variance=variance, candidate_noisy=False, zones=True, extra_masks=extra
            )
            oracle_samples = pipeline.linear_raw_samples(linear, oracle)
            rung_l2 = metrics.score(
                x, pipeline.stored_to_signal(oracle_samples), None, mask, variance=variance, candidate_noisy=False
            )
            r2_samples = pipeline.linear_raw_samples(pipeline.decode(renders[("R2", name)]), prior)
            r2_score = metrics.score(
                x, pipeline.stored_to_signal(r2_samples), None, mask, variance=variance, candidate_noisy=False
            )
            noop = {
                "camera_transform_not_identity": bool(np.abs(cam_xyz - np.eye(3)).max() > 1e-3),
                "neutral_not_unity": bool(np.abs(prior - 1).max() > 1e-3),
            }
            results[name] = {
                "size": list(sizes[name]),
                "layout": "LinearRaw",
                "gated": gated,
                "ladder": {"L0": _brief(gated), "L2": _brief(rung_l2)},
                "reported": {"R2_stress": _brief(r2_score)},
                "integrity": _integrity(files, sizes[name], None, cfg),
                "noop_checks": noop,
                "files": {k: str(Path(v).relative_to(cfg.run_dir)) for k, v in files.items()},
            }
            results[name]["ladder_steps"] = _ladder_steps({"L0": gated, "L2": rung_l2})
            rel = metrics.error_map(x, y, gated["k"], mask)
            results[name]["error_maps"] = save_error_maps(folder, rel, None)
            results[name]["histogram"] = metrics.histogram(x, y, gated["k"], mask, None)
            save_mask(folder / "mask.png", mask)
            results[name]["sanity"] = sanity_checks(x, None, mask, binned, constants.NOISE_SEED + 100)
            del linear, samples, y, rel

    # 50 MP: HAT-L x4 with the x8 ensemble on the 4K render, back-projection, mosaic, noise.
    name = "50mp"
    x, layout, mask, binned = refs[name]
    variance = _variance_fn(binned)
    folder = root / name
    hw = sizes[name]
    source4k = renders[("R1", "4k")]
    source_linear = pipeline.decode(source4k)
    with _timed(cfg, "SR on the 4K R1 render (x8 ensemble)"):
        up = pipeline.run_sr(source4k, cfg.model, preset, cfg.log)
    with _timed(cfg, "50 MP output"):
        hr_plain = pipeline.upscale_linear(
            up.ensemble, source_linear, hw, back_projection_iterations=0, source_clipped=None
        )
        hr = resample.back_project(
            hr_plain, source_linear, iterations=preset.back_projection_iterations, source_clipped=clipped_src["4k"]
        )
        clean, noisy = pipeline.cfa_samples(hr, prior, layout, noise_seed=preset.noise_seed)
        files = _write_output(folder, name, noisy, prior, cfa=ref.pattern, clean=clean)
        y0 = _candidate_from_dng(files["dng"], layout)
        y1 = _candidate_from_dng(files["clean_dng"], layout)
        src50 = resample.nearest(clipped_src["4k"].astype(np.uint8), hw).astype(bool)
        gated = metrics.score(
            x,
            y0,
            layout,
            mask,
            variance=variance,
            candidate_noisy=True,
            zones=True,
            extra_masks={"source_clipped": src50},
        )
        l1 = metrics.score(x, y1, layout, mask, variance=variance, candidate_noisy=False, zones=True)

    def clean_score(linear, neutral):
        samples, _ = pipeline.cfa_samples(linear, neutral, layout, noise_seed=None)
        return metrics.score(
            x, pipeline.stored_to_signal(samples), layout, mask, variance=variance, candidate_noisy=False
        )

    with _timed(cfg, "50 MP ladder"):
        l2 = clean_score(hr, oracle)
        full_linear = pipeline.decode(r1_full)
        l3 = clean_score(full_linear, prior)
        l4 = clean_score(full_linear, oracle)
        del full_linear
    with _timed(cfg, "50 MP ablation"):
        hr_identity = resample.back_project(
            pipeline.upscale_linear(up.identity, source_linear, hw, back_projection_iterations=0, source_clipped=None),
            source_linear,
            iterations=preset.back_projection_iterations,
            source_clipped=clipped_src["4k"],
        )
        hr_lanczos = resample.back_project(
            resample.lanczos(source_linear, hw),
            source_linear,
            iterations=preset.back_projection_iterations,
            source_clipped=clipped_src["4k"],
        )
        rows = {
            "x8 self-ensemble (vs identity member)": clean_score(hr_identity, prior),
            "HAT-L (vs Lanczos)": clean_score(hr_lanczos, prior),
            "back-projection (vs off)": clean_score(hr_plain, prior),
            "illuminant prior (vs daylight)": clean_score(hr, daylight),
        }
        ablation = {}
        for row, without in rows.items():
            delta = {p: l1["planes"][p]["A"] - without["planes"][p]["A"] for p in metrics.PLANES}
            ok = all(d >= constants.ABLATION_MIN_DELTA for d in delta.values())
            ablation[row] = {"delta_A": delta, "without": _brief(without), "meets_rule": bool(ok)}
            if not ok:
                warnings.append(f"ablation: {row} breaks the section 5.1 rule (delta A {delta})")
    with _timed(cfg, "50 MP tile seams and no-op checks"):
        from synthraw.super_resolution import diagnose_tile_seams

        lanczos4 = np.clip(sr.lanczos_model()(np.asarray(source4k, dtype=np.float32) / 255.0), 0, 1)
        seam_path = Path(cfg.run_dir) / "diagnostics" / "tile_seams_50mp.npy"
        seam_path.unlink(missing_ok=True)
        try:
            seams = diagnose_tile_seams(
                up.ensemble,
                lanczos4,
                seam_path,
                tile_size=preset.tile,
                overlap=preset.overlap,
                scale=constants.SR_SCALE,
            )
            seam = {
                "ratio": seams.ratio,
                "boundary_mean": seams.boundary_mean,
                "interior_mean": seams.interior_mean,
                "passed": bool(seams.ratio < constants.SEAM_RATIO_MAX),
            }
        except ValueError as error:
            seam = {"status": "not applicable", "reason": str(error), "passed": True}
        noop = {
            "sr_differs_from_lanczos": bool(np.mean(np.abs(up.ensemble - lanczos4)) > 1e-4),
            "ensemble_differs_from_identity": bool(
                not preset.self_ensemble or np.abs(up.ensemble - up.identity).max() > 0
            ),
            "back_projection_changed_estimate": bool(np.abs(hr - hr_plain).max() > 0),
            "noise_changed_mosaic": bool(np.abs(noisy.astype(np.int32) - clean.astype(np.int32)).max() > 0),
            "camera_transform_not_identity": bool(np.abs(cam_xyz - np.eye(3)).max() > 1e-3),
        }
        del lanczos4, hr_identity, hr_lanczos, hr_plain, up

    with _timed(cfg, "50 MP reported-only variants"):
        reported = {"clean_mosaic": _brief(l1)}
        for label, render, seed in (
            ("from_2k_render", renders[("R1", "2k")], 1),
            ("R2_stress", renders[("R2", "4k")], 2),
        ):
            linear = pipeline.linear_output(render, hw, model=cfg.model, preset=preset, progress=cfg.log)
            _, noisy_v = pipeline.cfa_samples(linear, prior, layout, noise_seed=preset.noise_seed + seed)
            reported[label] = _brief(
                metrics.score(
                    x, pipeline.stored_to_signal(noisy_v), layout, mask, variance=variance, candidate_noisy=True
                )
            )
            del linear, noisy_v

    rungs = {"L0": gated, "L1": l1, "L2": l2, "L3": l3, "L4": l4}
    results[name] = {
        "size": list(hw),
        "layout": f"CFA {ref.pattern}",
        "gated": gated,
        "ladder": {rung: _brief(score) for rung, score in rungs.items()},
        "ladder_steps": _ladder_steps(rungs),
        "ablation": ablation,
        "band_split": metrics.band_split(x, y0, layout, mask, gated["k"], sizes["4k"]),
        "reported": reported,
        "tile_seams": seam,
        "noop_checks": noop,
        "integrity": _integrity(files, hw, ref.pattern, cfg),
        "files": {k: str(Path(v).relative_to(cfg.run_dir)) for k, v in files.items()},
        "reference_clipped_fraction": clip_stats["clipped_fraction"],
    }
    rel = metrics.error_map(x, y0, gated["k"], mask)
    results[name]["error_maps"] = save_error_maps(folder, rel, layout)
    results[name]["histogram"] = metrics.histogram(x, y0, gated["k"], mask, layout)
    save_mask(folder / "mask.png", mask)
    del rel, hr, clean, noisy, y1
    with _timed(cfg, "50 MP sanity checks"):
        results[name]["sanity"] = sanity_checks(x, layout, mask, binned, constants.NOISE_SEED + 200)

    for name in constants.OUTPUT_ORDER:
        _write_json(root / name / "metrics.json", results[name])
        out = results[name]
        if not out["integrity"]["passed"]:
            invalid += [f"{constants.OUTPUT_LABELS[name]} integrity: {p}" for p in out["integrity"]["problems"]]
        if not out["sanity"]["hard_passed"]:
            invalid.append(f"{constants.OUTPUT_LABELS[name]} hard sanity check failed: {out['sanity']['hard']}")
        for check, passed in out["sanity"]["sensitivity"].items():
            if not passed["passed"]:
                warnings.append(f"metric-sensitivity warning: {constants.OUTPUT_LABELS[name]} {check}")
        for check, passed in out["noop_checks"].items():
            if not passed:
                invalid.append(f"{constants.OUTPUT_LABELS[name]} silent no-op: {check}")
        if not out.get("tile_seams", {"passed": True})["passed"]:
            invalid.append(f"{constants.OUTPUT_LABELS[name]} tile seams: {out['tile_seams']}")

    verdict = build_verdict(results, invalid, warnings)
    verdict["timings_s"] = dict(cfg.timings)
    verdict["preset"] = preset.as_dict()
    _write_json(Path(cfg.run_dir) / "verdict.json", verdict)
    cfg.log(verdict["text"])
    return verdict


def build_verdict(results: dict, invalid: list[str], warnings: list[str]) -> dict:
    outputs = {}
    for name in constants.OUTPUT_ORDER:
        gated = results[name]["gated"]
        worst = gated["worst_plane"]
        contributors = _contributors(results[name]["ladder_steps"], worst)
        outputs[name] = {
            "label": constants.OUTPUT_LABELS[name],
            "passes": bool(gated["passes"]),
            "min_A": gated["min_A"],
            "worst_plane": worst,
            "planes": {p: {"A": gated["planes"][p]["A"], "P2": gated["planes"][p]["P2"]} for p in metrics.PLANES},
            "k": gated["k"],
            "k_ev": gated["k_ev"],
            "contributors": [{"step": LADDER_NAMES[s], "points": round(100 * c, 2)} for s, c in contributors],
        }
    headline_name = min(constants.OUTPUT_ORDER, key=lambda n: outputs[n]["min_A"])
    headline = outputs[headline_name]
    per_output = " / ".join(
        f"{o['label']} {'PASS' if o['passes'] else 'MISSED'} {100 * o['min_A']:.1f}%" for o in outputs.values()
    )
    if invalid:
        status = "INVALID"
        text = f"INVALID: {invalid[0]}" + (f" (and {len(invalid) - 1} more)" if len(invalid) > 1 else "")
    else:
        status = "PASS" if all(o["passes"] for o in outputs.values()) else "MISSED"
        where = f"{headline['label']}, {headline['worst_plane']}"
        text = f"{status}: A = {100 * headline['min_A']:.1f}% ({where}). {per_output}."
        if status == "MISSED":
            failing = [o for o in outputs.values() if not o["passes"]]
            worst = min(failing, key=lambda o: o["min_A"])
            top = worst["contributors"]
            text += (
                f" Largest contributor ({worst['label']}, {worst['worst_plane']}): "
                f"{top[0]['step']} (-{top[0]['points']:.1f} points)."
            )
            if len(top) > 1:
                text += f" Next: {top[1]['step']} (-{top[1]['points']:.1f} points)."
    return {
        "status": status,
        "text": text,
        "headline": {
            "output": headline_name,
            "label": headline["label"],
            "min_A": headline["min_A"],
            "plane": headline["worst_plane"],
        },
        "outputs": outputs,
        "invalid_reasons": invalid,
        "warnings": warnings,
        "thresholds": {"P1_min": constants.P1_MIN, "P2_min": constants.P2_MIN},
        "in_sample": "Single reference frame; the score is in-sample (PRD v6 section 13).",
        "step": 4,
        "accuracy_percent": None if status == "INVALID" else round(100 * headline["min_A"], 2),
        "finite": all(math.isfinite(o["min_A"]) for o in outputs.values()),
    }
