"""Driver step 4: every frame to a validated 16-bit RGGB DNG at source resolution (README sections 2-3).

The clip calibration (camera neutral, white balance and exposure) is fixed once, before the first
frame, and every frame uses it unchanged: the v6 MAX preset's frozen illuminant prior (Planckian at
the mired midpoint of 2850-7500 K) and render white = sensor clip. Nothing is estimated per frame,
so nothing can flicker. Each frame gets fresh v6 shot and read noise from its own seed.
"""

from __future__ import annotations

import csv
import time
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import numpy as np

from prd_v5.dng_validate import run_dng_validate
from prd_v6 import colour, constants, dng, pipeline, production, sr
from prd_v6.delivery import _read_json, _sha256, _write_json
from video_v1 import sequence, source

STAGES = ("decode", "raw", "write", "libraw", "dng_validate")
CHECKPOINT_EVERY = 10
# HAT-L x4 cost measured in v6 run 20261104T064058Z_ad151cd on an A100-SXM4-80GB: 70 tiles of
# 512 px (128 px overlap) with the x8 self-ensemble in 2386 s (run.log 06:45:23 to 07:25:09).
HAT_L_RUN = "20261104T064058Z_ad151cd"
HAT_L_TILES = 70
HAT_L_SECONDS = 2386.0
UHD_HW = (2160, 3840)


def calibration() -> dict:
    """The one clip-wide calibration every frame uses (README section 3)."""
    cam_xyz = pipeline.camera_matrix()
    neutral = colour.prior_neutral(cam_xyz)
    return {
        "method": "prd_v6 MAX preset illuminant prior, fixed for the whole clip",
        "illuminant": "prior",
        "cct_k": constants.PRIOR_CCT_K,
        "as_shot_neutral": [float(v) for v in neutral],
        "white_balance_gains": [float(v) for v in colour.balance_gains(neutral)],
        "exposure": f"render white (linear 1.0) = sensor clip, DN16 signal {constants.SIGNAL_16}",
        "black_level_16": constants.BLACK_16,
        "white_level_16": constants.WHITE_16,
        "color_matrix": cam_xyz.tolist(),
        "per_frame_estimation": False,
    }


def frame_seed(index: int) -> np.random.SeedSequence:
    """Independent, reproducible noise for every frame (v6 NOISE_SEED, frame index)."""
    return np.random.SeedSequence((constants.NOISE_SEED, int(index)))


def raw_frame(linear: np.ndarray, neutral, layout: np.ndarray, index: int) -> np.ndarray:
    """Linear BT.709 frame to stored RGGB samples with the frame's own v6 noise (the 50 MP path, no SR)."""
    _clean, noisy = pipeline.cfa_samples(linear, neutral, layout, noise_seed=frame_seed(index))
    return noisy


def sequence_dir(run_dir, clip_name: str) -> Path:
    return Path(run_dir) / "dng" / clip_name


def manifest_path(run_dir, clip_name: str) -> Path:
    return sequence_dir(run_dir, clip_name) / f"{clip_name}_manifest.json"


def frames_planned(clip: source.Clip, max_frames: int | None) -> int:
    return clip.frames if not max_frames else min(clip.frames, int(max_frames))


def run(run_dir, inputs: dict, settings: dict, log=print) -> dict:
    run_dir = Path(run_dir)
    clip: source.Clip = inputs["clip"]
    tools = inputs["tools"]
    rate = Fraction(clip.frame_rate)
    planned = frames_planned(clip, settings.get("MAX_FRAMES"))
    name = sequence.clip_name(clip.path)
    hw = source.even_crop(clip.size)
    layout = production.rggb_layout(hw)
    cal = calibration()
    _write_json(run_dir / "calibration.json", cal)
    neutral, cam_xyz = np.asarray(cal["as_shot_neutral"]), pipeline.camera_matrix()
    folder = sequence_dir(run_dir, name)
    folder.mkdir(parents=True, exist_ok=True)
    log(f"converting {planned} of {clip.frames} frames at {hw[1]}x{hw[0]} to {folder}")
    records: list[dict] = []
    started = time.perf_counter()
    decoded = source.frames(clip, ffmpeg=tools["ffmpeg"], ffmpeg_major=tools.get("major"), limit=planned)
    with closing(decoded) as frames:
        while True:
            t0 = time.perf_counter()
            linear = next(frames, None)
            if linear is None:
                break
            index = len(records)
            t1 = time.perf_counter()
            samples = raw_frame(linear[: hw[0], : hw[1]], neutral, layout, index)
            t2 = time.perf_counter()
            path = sequence.write_frame(
                folder / sequence.frame_name(name, index),
                samples,
                index=index,
                frames=planned,
                clip=name,
                rate=rate,
                color_matrix=cam_xyz,
                as_shot_neutral=neutral,
            )
            t3 = time.perf_counter()
            libraw = dng.libraw_check(path, expected_hw=hw, cfa_pattern=constants.CFA_PATTERN)
            t4 = time.perf_counter()
            validated = run_dng_validate(path, inputs.get("dng_validate"))
            t5 = time.perf_counter()
            records.append(
                {
                    "frame": index,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "time_s": float(index / rate),
                    "timecode": sequence.timecode_text(index, rate),
                    "noise_seed": [constants.NOISE_SEED, index],
                    "libraw": libraw["status"],
                    "libraw_problems": libraw.get("problems", []),
                    "dng_validate": validated["status"],
                    "dng_validate_errors": validated.get("error_lines", []) or validated.get("reason", ""),
                    "seconds": dict(zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4), strict=True)),
                }
            )
            if (index + 1) % CHECKPOINT_EVERY == 0 or index + 1 == planned:
                elapsed = time.perf_counter() - started
                log(
                    f"frame {index + 1}/{planned}: {elapsed / (index + 1):.2f} s per frame, "
                    f"about {elapsed / (index + 1) * (planned - index - 1) / 60:.1f} min left"
                )
                _write_sequence_manifest(run_dir, clip, name, hw, planned, records, cal)
    wall = time.perf_counter() - started
    _write_sequence_manifest(run_dir, clip, name, hw, planned, records, cal)
    _write_timings(run_dir / "timings.csv", records)
    summary = summarise(records, planned=planned, total=clip.frames, wall_s=wall, hw=hw)
    summary.update(
        clip_name=name,
        size=list(hw),
        cropped_from=list(clip.size) if tuple(hw) != tuple(clip.size) else None,
        frame_rate=clip.frame_rate,
        sequence_dir=str(folder.relative_to(run_dir)),
        manifest=str(manifest_path(run_dir, name).relative_to(run_dir)),
        estimate_4k=estimate_4k(summary_seconds(records), hw, planned),
    )
    if summary["frames_converted"] != planned:
        summary["frame_count_mismatch"] = (
            f"ffmpeg decoded {summary['frames_converted']} frames, ffprobe counted {planned}"
        )
    _write_json(run_dir / "convert.json", summary)
    log(
        f"converted {summary['frames_converted']} frames in {wall / 60:.1f} min "
        f"({summary['seconds_per_frame']['mean']:.2f} s per frame); validation failures: "
        f"{summary['validation_failures']}"
    )
    if summary["frames_converted"] != planned:
        log(f"warning: ffmpeg decoded {summary['frames_converted']} frames, ffprobe counted {planned}")
    return summary


def _write_sequence_manifest(run_dir, clip, name, hw, planned, records, cal) -> None:
    _write_json(
        manifest_path(run_dir, name),
        {
            "sequence": name,
            "format": "CinemaDNG-style sequence: one DNG 1.4 CFA (RGGB) file per frame, uncompressed 16-bit",
            "naming": f"{name}_<frame:06d>.dng, frame numbers from 0",
            "frames_planned": planned,
            "frames_written": len(records),
            "source_frames": clip.frames,
            "size": list(hw),
            "frame_rate": clip.frame_rate,
            "frame_rate_float": float(Fraction(clip.frame_rate)),
            "timecode": "SMPTE 12M from 00:00:00:00 in the TimeCodes tag"
            if sequence.timecode(0, Fraction(clip.frame_rate)) is not None
            else "none: above 30 fps the TimeCodes tag is omitted (FrameRate and ImageNumber remain)",
            "calibration": {k: cal[k] for k in ("illuminant", "cct_k", "as_shot_neutral", "exposure")},
            "frames": records,
        },
    )


def _write_timings(path: Path, records: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", *STAGES, "total"])
        for record in records:
            seconds = [record["seconds"][s] for s in STAGES]
            writer.writerow([record["frame"], *(f"{v:.4f}" for v in seconds), f"{sum(seconds):.4f}"])


def summary_seconds(records: list[dict]) -> dict:
    """Mean seconds per frame for each stage."""
    if not records:
        return {stage: 0.0 for stage in STAGES}
    return {stage: float(np.mean([r["seconds"][stage] for r in records])) for stage in STAGES}


def summarise(records: list[dict], *, planned: int, total: int, wall_s: float, hw) -> dict:
    totals = [sum(r["seconds"].values()) for r in records] or [0.0]
    failures = [r["frame"] for r in records if r["libraw"] != "passed" or r["dng_validate"] != "passed"]
    return {
        "frames_source": total,
        "frames_planned": planned,
        "frames_converted": len(records),
        "libraw_passed": sum(r["libraw"] == "passed" for r in records),
        "dng_validate_passed": sum(r["dng_validate"] == "passed" for r in records),
        "validation_failures": len(failures),
        "failed_frames": failures[:50],
        "bytes": int(sum(r["bytes"] for r in records)),
        "wall_s": float(wall_s),
        "seconds_per_frame": {
            "mean": float(np.mean(totals)),
            "median": float(np.median(totals)),
            "max": float(np.max(totals)),
        },
        "stage_seconds_mean": summary_seconds(records),
        "megapixels": hw[0] * hw[1] / 1e6,
    }


def sr_tiles(hw) -> int:
    """HAT-L tiles the v6 SR stage runs on one input frame of this size (512 px, 128 px overlap)."""
    rows = sr.tile_starts(int(hw[0]), constants.SR_TILE, constants.SR_OVERLAP)
    cols = sr.tile_starts(int(hw[1]), constants.SR_TILE, constants.SR_OVERLAP)
    return len(rows) * len(cols)


def estimate_4k(stage_seconds: dict, hw, frames: int) -> dict:
    """What a 4K UHD version of this clip would cost: v6 SR x4 per frame, then the same stages at 4K.

    Not measured here. SR uses the v6 run's measured HAT-L seconds per tile on an A100; the other
    stages scale with the pixel count from this run's measured times.
    """
    per_tile = HAT_L_SECONDS / HAT_L_TILES
    tiles = sr_tiles(hw)
    ratio = (UHD_HW[0] * UHD_HW[1]) / (hw[0] * hw[1])
    other = sum(stage_seconds.values()) * ratio
    with_ensemble = tiles * per_tile + other
    without_ensemble = tiles * per_tile / 8 + other
    return {
        "target": f"{UHD_HW[1]}x{UHD_HW[0]}",
        "sr_tiles_per_frame": tiles,
        "hat_l_seconds_per_tile_a100": per_tile,
        "hat_l_measured_in": HAT_L_RUN,
        "pixel_ratio": ratio,
        "other_stages_seconds_per_frame": other,
        "seconds_per_frame_x8_ensemble": with_ensemble,
        "hours_for_clip_x8_ensemble": with_ensemble * frames / 3600,
        "seconds_per_frame_no_ensemble": without_ensemble,
        "hours_for_clip_no_ensemble": without_ensemble * frames / 3600,
        "dng_bytes_per_frame": UHD_HW[0] * UHD_HW[1] * 2,
        "note": (
            "Estimate only, A100. Per-frame SR has no temporal model, so a 4K version would also need its own "
            "flicker check; the v6 benchmark number does not transfer to it either."
        ),
    }


def load(run_dir) -> dict | None:
    return _read_json(Path(run_dir) / "convert.json")
