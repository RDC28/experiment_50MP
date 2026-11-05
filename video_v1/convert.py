"""Driver step 4: every frame to raw outputs at source resolution (README sections 2-3).

Per frame the v6 50 MP path makes one RGGB mosaic, and three outputs carry exactly that mosaic: a
validated 16-bit DNG (optional, on by default), a developed 16-bit sRGB TIFF, and the next frame of
the clip's 14-bit MLV.

The clip calibration (camera neutral, white balance and exposure) is fixed once, before the first
frame, and every frame uses it unchanged: the source's own white (D65, the sRGB/BT.709 white) as the
as-shot neutral and render white = sensor clip. Nothing is estimated per frame, so nothing can
flicker. Each frame gets fresh v6 shot and read noise from its own seed.
"""

from __future__ import annotations

import csv
import math
import tempfile
import time
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import numpy as np

from prd_v5.dng_validate import run_dng_validate
from prd_v6 import colour, constants, dng, pipeline, production, sr
from prd_v6.delivery import _read_json, _sha256, _write_json
from video_v1 import mlv, sequence, source

STAGES = ("decode", "raw", "write", "libraw", "dng_validate", "develop", "tiff", "mlv")
# Stages whose work grows with the output pixel count; decode (with the YCbCr to linear conversion) reads the source.
OUTPUT_STAGES = STAGES[1:]
CHECKPOINT_EVERY = 10
# DNG BaselineExposure. Adobe Camera Raw, Resolve and macOS place linear mid grey well above the sRGB
# rendering by default (macOS renders linear 0.18 at 0.65 instead of 0.46 with 0 EV); -0.5 EV brings
# their default look within reach of the source without crushing it. LibRaw, the TIFFs and the MLV
# ignore it. README section 2 has the measurements.
BASELINE_EXPOSURE_EV = -0.5
MLV_BITS = 14  # the deepest raw MLV App reads; 16-bit samples are rounded to it
MLV_CHUNK_MB = 480  # each .MLV/.Mnn chunk fits whole in one archive part (README section 5)
TIFF_COMPRESSION = "zlib"
TIFF_COMPRESSION_LEVEL = 6
SHUTTER_DEGREES = 180  # the nominal shutter the EXPO block reports; the source has none
# HAT-L x4 cost measured in v6 run 20261104T064058Z_ad151cd on an A100-SXM4-80GB: 70 tiles of
# 512 px (128 px overlap) with the x8 self-ensemble in 2386 s (run.log 06:45:23 to 07:25:09).
HAT_L_RUN = "20261104T064058Z_ad151cd"
HAT_L_TILES = 70
HAT_L_SECONDS = 2386.0
UHD_HW = (2160, 3840)


D65_CCT_K = 6504.0


def calibration() -> dict:
    """The one clip-wide calibration every frame uses (README section 3)."""
    cam_xyz = pipeline.camera_matrix()
    neutral = colour.daylight_neutral(cam_xyz)
    return {
        "method": "the source's own white (BT.709/sRGB, D65), fixed for the whole clip",
        "illuminant": "source white (D65)",
        "cct_k": D65_CCT_K,
        "as_shot_neutral": [float(v) for v in neutral],
        "white_balance_gains": [float(v) for v in colour.balance_gains(neutral)],
        "exposure": f"render white (linear 1.0) = sensor clip, DN16 signal {constants.SIGNAL_16}",
        "baseline_exposure_ev": BASELINE_EXPOSURE_EV,
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


def tiff_dir(run_dir, clip_name: str) -> Path:
    return Path(run_dir) / "tiff" / clip_name


def tiff_name(clip_name: str, index: int) -> str:
    return f"{clip_name}_{index:06d}.tif"


def mlv_dir(run_dir) -> Path:
    return Path(run_dir) / "mlv"


def setting(settings: dict, key: str, default):
    value = settings.get(key)
    return default if value is None else value


def write_tiff(path: Path, encoded: np.ndarray, icc: bytes, description: str) -> Path:
    """One developed frame: 16-bit sRGB RGB with the sRGB ICC profile, lossless deflate."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        colour.to_uint16(encoded),
        photometric="rgb",
        planarconfig="contig",
        compression=TIFF_COMPRESSION,
        compressionargs={"level": TIFF_COMPRESSION_LEVEL},
        predictor=True,
        iccprofile=icc,
        description=description,
        software=sequence.SOFTWARE,
        metadata=None,
    )
    return path


def mlv_info(hw, rate: Fraction, cal: dict) -> mlv.ClipInfo:
    """The MLV metadata: the DNG's black, white and ColorMatrix1 at 14 bits, the clip rate, D65."""
    shift = 16 - MLV_BITS
    black, white = constants.BLACK_16 >> shift, constants.WHITE_16 >> shift
    neutral = np.asarray(cal["as_shot_neutral"], dtype=np.float64)
    return mlv.ClipInfo(
        width=mlv.mlv_width(hw[1]),
        height=hw[0],
        bits=MLV_BITS,
        black=black,
        white=white,
        cfa_pattern=constants.CFA_PATTERN,
        color_matrix=cal["color_matrix"],
        frame_rate=rate,
        kelvin=int(round(cal["cct_k"])),
        wb_gains=tuple(int(round(1024 / v)) for v in neutral / neutral.max()),
        shutter_us=int(round(1_000_000 * SHUTTER_DEGREES / 360 / rate)),
        dynamic_range_ev=math.log2(white - black),
    )


def run(run_dir, inputs: dict, settings: dict, log=print) -> dict:
    run_dir = Path(run_dir)
    clip: source.Clip = inputs["clip"]
    tools = inputs["tools"]
    rate = Fraction(clip.frame_rate)
    name = sequence.clip_name(clip.path)
    hw = source.even_crop(clip.size)
    layout = production.rggb_layout(hw)
    keep_dng = bool(setting(settings, "WRITE_DNG", True))
    chunk_mb = float(setting(settings, "MLV_CHUNK_MB", MLV_CHUNK_MB))
    cal = calibration()
    _write_json(run_dir / "calibration.json", cal)
    neutral, cam_xyz = np.asarray(cal["as_shot_neutral"]), pipeline.camera_matrix()
    folder, tiffs = sequence_dir(run_dir, name), tiff_dir(run_dir, name)
    folder.mkdir(parents=True, exist_ok=True)
    tiffs.mkdir(parents=True, exist_ok=True)
    icc = colour.srgb_icc_profile()
    info = mlv_info(hw, rate, cal)
    writer = mlv.Writer(mlv_dir(run_dir), name, info, chunk_bytes=int(chunk_mb * 1e6))
    scratch = tempfile.TemporaryDirectory(prefix="video_v1_dng_")
    outputs = ("DNG, " if keep_dng else "") + "TIFF and MLV"
    log(f"converting {clip.frames} frames at {hw[1]}x{hw[0]} to {outputs} under {run_dir}")
    records: list[dict] = []
    started = time.perf_counter()
    decoded = source.frames(clip, ffmpeg=tools["ffmpeg"], ffmpeg_major=tools.get("major"), limit=clip.frames)
    with closing(decoded) as frames, scratch, writer:
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
                (folder if keep_dng else Path(scratch.name)) / sequence.frame_name(name, index),
                samples,
                index=index,
                frames=clip.frames,
                clip=name,
                rate=rate,
                color_matrix=cam_xyz,
                as_shot_neutral=neutral,
                baseline_exposure=BASELINE_EXPOSURE_EV,
            )
            t3 = time.perf_counter()
            record = {"frame": index}
            if keep_dng:
                libraw = dng.libraw_check(path, expected_hw=hw, cfa_pattern=constants.CFA_PATTERN)
                t4 = time.perf_counter()
                validated = run_dng_validate(path, inputs.get("dng_validate"))
                t5 = time.perf_counter()
                record.update(
                    file=path.name,
                    bytes=path.stat().st_size,
                    sha256=_sha256(path),
                    libraw=libraw["status"],
                    libraw_problems=libraw.get("problems", []),
                    dng_validate=validated["status"],
                    dng_validate_errors=validated.get("error_lines", []) or validated.get("reason", ""),
                )
            else:
                t4 = t5 = t3
            developed = pipeline.develop_cfa_dng(path)
            if not keep_dng:
                path.unlink()
            t6 = time.perf_counter()
            tiff = write_tiff(
                tiffs / tiff_name(name, index), developed, icc, sequence.description(index, clip.frames, name)
            )
            tiff_record = dng.tiff16_check(tiff, hw)
            t7 = time.perf_counter()
            mosaic = mlv.reduce_bits(samples[:, : info.width], MLV_BITS)
            writer.write(mosaic)
            t8 = time.perf_counter()
            record.update(
                time_s=float(index / rate),
                timecode=sequence.timecode_text(index, rate),
                noise_seed=[constants.NOISE_SEED, index],
                tiff=tiff.name,
                tiff_bytes=tiff.stat().st_size,
                tiff_check=tiff_record["status"],
                mlv_sha256=mlv.samples_sha256(mosaic),
                seconds=dict(zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7),
                                 strict=True)),
            )  # fmt: skip
            records.append(record)
            if (index + 1) % CHECKPOINT_EVERY == 0 or index + 1 == clip.frames:
                elapsed = time.perf_counter() - started
                log(
                    f"frame {index + 1}/{clip.frames}: {elapsed / (index + 1):.2f} s per frame, "
                    f"about {elapsed / (index + 1) * (clip.frames - index - 1) / 60:.1f} min left"
                )
                _write_manifests(run_dir, clip, name, hw, records, cal, keep_dng)
    t9 = time.perf_counter()
    mlv_check = mlv.verify(
        writer.paths[0],
        {
            **{k: getattr(info, k) for k in ("width", "height", "bits", "black", "white", "cfa_pattern")},
            "color_matrix": info.color_matrix,
            "frame_rate": rate,
            "frames": len(records),
            "frame_sha256": [r["mlv_sha256"] for r in records],
        },
    )
    mlv_check.update(
        verify_s=time.perf_counter() - t9,
        bytes=int(sum(p.stat().st_size for p in writer.paths)),
        paths=[str(p.relative_to(run_dir)) for p in writer.paths],
    )
    _write_json(mlv_dir(run_dir) / f"{name}_mlv_check.json", mlv_check)
    log(f"MLV {mlv_check['status']}: {len(writer.paths)} file(s), {mlv_check['frames']} frames read back bit-exact"
        if mlv_check["status"] == "passed" else f"MLV check FAILED: {mlv_check['problems']}")  # fmt: skip
    wall = time.perf_counter() - started
    _write_manifests(run_dir, clip, name, hw, records, cal, keep_dng)
    _write_timings(run_dir / "timings.csv", records)
    summary = summarise(records, total=clip.frames, wall_s=wall, hw=hw, dng_written=keep_dng)
    summary.update(
        clip_name=name,
        size=list(hw),
        cropped_from=list(clip.size) if tuple(hw) != tuple(clip.size) else None,
        frame_rate=clip.frame_rate,
        dng_written=keep_dng,
        sequence_dir=str(folder.relative_to(run_dir)) if keep_dng else None,
        manifest=str(manifest_path(run_dir, name).relative_to(run_dir)) if keep_dng else None,
        tiff_dir=str(tiffs.relative_to(run_dir)),
        tiff_manifest=str(tiff_manifest_path(run_dir, name).relative_to(run_dir)),
        mlv={
            "files": mlv_check["paths"],
            "bytes": mlv_check["bytes"],
            "bits": MLV_BITS,
            "width": info.width,
            "status": mlv_check["status"],
            "problems": mlv_check["problems"],
            "check": str((mlv_dir(run_dir) / f"{name}_mlv_check.json").relative_to(run_dir)),
        },
        estimate_4k=estimate_4k(summary_seconds(records), hw, clip.frames),
    )
    if summary["frames_converted"] != clip.frames:
        summary["frame_count_mismatch"] = (
            f"ffmpeg decoded {summary['frames_converted']} frames, ffprobe counted {clip.frames}"
        )
    _write_json(run_dir / "convert.json", summary)
    log(
        f"converted {summary['frames_converted']} frames in {wall / 60:.1f} min "
        f"({summary['seconds_per_frame']['mean']:.2f} s per frame); validation failures: "
        f"{summary['validation_failures']}"
    )
    if summary["frames_converted"] != clip.frames:
        log(f"warning: ffmpeg decoded {summary['frames_converted']} frames, ffprobe counted {clip.frames}")
    return summary


def tiff_manifest_path(run_dir, clip_name: str) -> Path:
    return tiff_dir(run_dir, clip_name) / f"{clip_name}_manifest.json"


def _write_manifests(run_dir, clip, name, hw, records, cal, keep_dng) -> None:
    common = {
        "frames_written": len(records),
        "source_frames": clip.frames,
        "size": list(hw),
        "frame_rate": clip.frame_rate,
        "frame_rate_float": float(Fraction(clip.frame_rate)),
        "calibration": {k: cal[k] for k in ("illuminant", "cct_k", "as_shot_neutral", "exposure")},
    }
    if keep_dng:
        _write_sequence_manifest(run_dir, clip, name, hw, records, cal)
    _write_json(
        tiff_manifest_path(run_dir, name),
        {
            "sequence": name,
            "format": "one developed 16-bit RGB TIFF per frame: sRGB primaries and transfer, sRGB ICC profile, "
            "lossless deflate",
            "develop": "LibRaw (rawpy) on the frame's mosaic: as-shot neutral, AHD demosaic, linear sRGB, then the "
            "sRGB transfer; no tone curve, sharpening or noise reduction",
            "naming": f"{tiff_name(name, 0).replace('000000', '<frame:06d>')}, frame numbers from 0",
            **common,
            "frames": [
                {k: r[k] for k in ("frame", "tiff", "tiff_bytes", "tiff_check", "time_s", "timecode")} for r in records
            ],
        },
    )


def _write_sequence_manifest(run_dir, clip, name, hw, records, cal) -> None:
    _write_json(
        manifest_path(run_dir, name),
        {
            "sequence": name,
            "format": "CinemaDNG-style sequence: one DNG 1.4 CFA (RGGB) file per frame, uncompressed 16-bit",
            "naming": f"{name}_<frame:06d>.dng, frame numbers from 0",
            "frames_written": len(records),
            "source_frames": clip.frames,
            "size": list(hw),
            "frame_rate": clip.frame_rate,
            "frame_rate_float": float(Fraction(clip.frame_rate)),
            "timecode": "SMPTE 12M from 00:00:00:00 in the TimeCodes tag"
            if sequence.timecode(0, Fraction(clip.frame_rate)) is not None
            else "none: above 30 fps the TimeCodes tag is omitted (FrameRate and ImageNumber remain)",
            "calibration": {
                k: cal[k] for k in ("illuminant", "cct_k", "as_shot_neutral", "exposure", "baseline_exposure_ev")
            },
            "frames": [{k: v for k, v in r.items() if not k.startswith(("tiff", "mlv"))} for r in records],
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


def _frame_passed(record: dict, dng_written: bool) -> bool:
    dng_ok = not dng_written or (record.get("libraw") == "passed" and record.get("dng_validate") == "passed")
    return dng_ok and record.get("tiff_check", "passed") == "passed"


def summarise(records: list[dict], *, total: int, wall_s: float, hw, dng_written: bool = True) -> dict:
    """Per-frame totals. A frame fails validation when its DNG fails LibRaw or dng_validate, or its TIFF
    reads back wrong; the MLV is checked as a whole after the last frame."""
    totals = [sum(r["seconds"].values()) for r in records] or [0.0]
    failures = [r["frame"] for r in records if not _frame_passed(r, dng_written)]
    return {
        "frames_source": total,
        "frames_converted": len(records),
        "libraw_passed": sum(r.get("libraw") == "passed" for r in records) if dng_written else None,
        "dng_validate_passed": sum(r.get("dng_validate") == "passed" for r in records) if dng_written else None,
        "tiff_passed": sum(r.get("tiff_check") == "passed" for r in records),
        "validation_failures": len(failures),
        "failed_frames": failures[:50],
        "bytes": int(sum(r.get("bytes", 0) for r in records)),
        "tiff_bytes": int(sum(r.get("tiff_bytes", 0) for r in records)),
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
    """What a 4K UHD version of this clip would cost: v6 SR x4 per frame, then the output stages at 4K.

    Computed, not measured. SR uses the v6 run's measured HAT-L seconds per tile on an A100. A 4K
    version still decodes the source, so decode keeps this run's measured time; the output stages
    scale with the pixel count from this run's measured times.
    """
    per_tile = HAT_L_SECONDS / HAT_L_TILES
    tiles = sr_tiles(hw)
    ratio = (UHD_HW[0] * UHD_HW[1]) / (hw[0] * hw[1])
    source_stages = sum(v for k, v in stage_seconds.items() if k not in OUTPUT_STAGES)
    output_stages = sum(stage_seconds[k] for k in OUTPUT_STAGES) * ratio
    other = source_stages + output_stages
    with_ensemble = tiles * per_tile + other
    without_ensemble = tiles * per_tile / 8 + other
    return {
        "target": f"{UHD_HW[1]}x{UHD_HW[0]}",
        "sr_tiles_per_frame": tiles,
        "hat_l_seconds_per_tile_a100": per_tile,
        "hat_l_measured_in": HAT_L_RUN,
        "pixel_ratio": ratio,
        "source_stages_seconds_per_frame": source_stages,
        "output_stages_seconds_per_frame": output_stages,
        "seconds_per_frame_x8_ensemble": with_ensemble,
        "hours_for_clip_x8_ensemble": with_ensemble * frames / 3600,
        "seconds_per_frame_no_ensemble": without_ensemble,
        "hours_for_clip_no_ensemble": without_ensemble * frames / 3600,
        "dng_bytes_per_frame": UHD_HW[0] * UHD_HW[1] * 2,
        "mlv_bytes_per_frame": UHD_HW[0] * UHD_HW[1] * MLV_BITS // 8,
        "note": (
            "Computed estimate, not measured: A100. Per-frame SR has no temporal model, so a 4K version would "
            "also need its own flicker check; the v6 benchmark number does not transfer to it either."
        ),
    }


def load(run_dir) -> dict | None:
    return _read_json(Path(run_dir) / "convert.json")
