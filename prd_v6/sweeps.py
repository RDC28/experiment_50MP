"""Driver step 6: edit sweeps and secondary reports (sections 5.1 and 6.5); none of this is gated."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from prd_v6 import colour, constants, pipeline, reference, resample

PREVIEW_SIDE = 1024


def _preview(path: Path, rgb8: np.ndarray) -> str:
    from PIL import Image

    scale = PREVIEW_SIDE / max(rgb8.shape[:2])
    if scale < 1:
        rgb8 = colour.to_uint8(
            resample.area(rgb8.astype(np.float32) / 255.0, (round(rgb8.shape[0] * scale), round(rgb8.shape[1] * scale)))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb8).save(path, quality=90)
    return path.name


def edit_sweeps(dng: Path, folder: Path) -> dict:
    """LibRaw edits on the 50 MP DNG: exposure +-1 EV, as-shot vs daylight WB, three demosaics."""
    import rawpy

    daylight = colour.daylight_neutral(pipeline.camera_matrix())
    daylight_wb = [float(1 / daylight[0]), 1.0, float(1 / daylight[2]), 1.0]
    variants = {
        "exposure_minus_1ev": dict(half_size=True, use_camera_wb=True, exp_shift=0.5),
        "exposure_0ev": dict(half_size=True, use_camera_wb=True),
        "exposure_plus_1ev": dict(half_size=True, use_camera_wb=True, exp_shift=2.0),
        "wb_as_shot": dict(half_size=True, use_camera_wb=True),
        "wb_daylight": dict(half_size=True, user_wb=daylight_wb),
        "demosaic_AHD": dict(use_camera_wb=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD),
        "demosaic_DHT": dict(use_camera_wb=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT),
        "demosaic_LINEAR": dict(use_camera_wb=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR),
    }
    out = {}
    for name, options in variants.items():
        with rawpy.imread(str(dng)) as raw:
            rgb = raw.postprocess(no_auto_bright=True, user_flip=0, output_bps=8, **options)
        out[name] = {"preview": _preview(folder / f"{name}.jpg", rgb), "mean_rgb": rgb.reshape(-1, 3).mean(0).tolist()}
    return out


def noise_gate(arw: Path, dng: Path) -> dict:
    """G-plane conversion gain on the 14-bit scale (full well / 14,848 DN) within 15% (section 6.5)."""
    from synthraw.realism import measure_raw

    def green(measured: dict, key: str):
        values = [(measured["planes"][g].get("photon_transfer_fit") or {}).get(key) for g in ("G1", "G2")]
        values = [v for v in values if v is not None and math.isfinite(v)]
        return float(np.mean(values)) if values else None

    ref, syn = measure_raw(arw), measure_raw(dng)
    ref_gain = (green(ref, "full_well_e") or math.nan) / constants.SIGNAL_14
    syn_gain = (green(syn, "full_well_e") or math.nan) / constants.SIGNAL_14
    ratio = syn_gain / ref_gain if ref_gain else math.nan
    return {
        "reference_gain_e_per_dn14": ref_gain,
        "output_gain_e_per_dn14": syn_gain,
        "ratio": ratio,
        "passed": bool(math.isfinite(ratio) and abs(ratio - 1) <= constants.NOISE_GATE_TOLERANCE),
        "read_noise_e_reported_only": {"reference": green(ref, "read_noise_e"), "output": green(syn, "read_noise_e")},
    }


def render_space(arw: Path, dng: Path) -> dict:
    """PSNR and SSIM of the developed 50 MP DNG against the full-resolution R1 render (reported only)."""
    from skimage.metrics import structural_similarity

    target = colour.srgb_encode(reference.render_linear(arw))
    developed = pipeline.develop_cfa_dng(dng)
    mse = float(np.mean((developed - target) ** 2, dtype=np.float64))
    ssim = structural_similarity(target, developed, channel_axis=2, data_range=1.0)
    return {"psnr_db": 10 * math.log10(1 / mse) if mse else math.inf, "ssim": float(ssim)}


def run(run_dir: Path, arw: Path, log=print) -> dict:
    run_dir = Path(run_dir)
    dng = run_dir / "benchmark" / "50mp" / "exp50v6_50mp.dng"
    if not dng.exists():
        raise FileNotFoundError(f"{dng} missing: the benchmark did not write the 50 MP output")
    secondary: dict = {}
    for name, task in (
        ("edit_sweeps", lambda: edit_sweeps(dng, run_dir / "sweeps")),
        ("noise_gate", lambda: noise_gate(arw, dng)),
        ("render_space_50mp", lambda: render_space(arw, dng)),
    ):
        log(f"[sweeps] {name}")
        secondary[name] = task()
    folder = run_dir / "secondary"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "secondary.json").write_text(json.dumps(secondary, indent=2, default=str))
    return secondary
