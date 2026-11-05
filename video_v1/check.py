"""Driver step 5: check the developed frames' stability and make the preview (README section 4).

Each developed TIFF (step 4 develops every frame's mosaic through LibRaw exactly as v6 develops its
50 MP output: as-shot neutral, AHD, linear, sRGB transfer) is compared with the decoded source frame
it came from. The check is about
consistency over time, not accuracy: per frame it takes the developed/source ratio of mean luma and
of the R/G and B/G colour ratios, and asks that those ratios do not jump between frames or drift over
the clip. A pipeline with no per-frame decisions should keep them flat; a jump means flicker.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile

from prd_v6 import colour
from prd_v6.delivery import _write_json
from video_v1 import convert, source

STEP_MAX = 0.005  # frame to frame: |ratio(t) / ratio(t-1) - 1| at most 0.5%
SPAN_MAX = 0.02  # whole clip: (max - min) / median of each ratio at most 2%
DARK_LUMA = 0.002  # frames darker than this (linear mean luma) are reported but not scored
BORDER = 8  # pixels ignored at each edge (demosaic border)
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)  # BT.709 luma weights on linear light
METRICS = ("level", "red_green", "blue_green")
PREVIEW_CRF = 16
THUMB_COUNT = 3
LABELS = ("source (decoded)", "developed TIFF (from the raw mosaic)")


def frame_stats(src_linear: np.ndarray, dev_linear: np.ndarray) -> dict:
    """Mean linear RGB and luma of the source and developed frames (both clipped to [0, 1])."""
    inner = (slice(BORDER, -BORDER or None), slice(BORDER, -BORDER or None))
    out = {}
    for key, image in (("src", src_linear), ("dev", dev_linear)):
        crop = np.clip(np.asarray(image, dtype=np.float32)[inner], 0.0, 1.0)
        if crop.size == 0:
            crop = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        rgb = crop.reshape(-1, 3).mean(axis=0, dtype=np.float64)
        out[key] = {"rgb": [float(v) for v in rgb], "luma": float(rgb @ LUMA)}
    return out


def ratios(stats: dict) -> dict:
    """Developed/source ratios of mean luma, R/G and B/G (1.0 means the developed frame matches)."""
    s, d = stats["src"]["rgb"], stats["dev"]["rgb"]
    tiny = 1e-9
    return {
        "level": stats["dev"]["luma"] / max(stats["src"]["luma"], tiny),
        "red_green": (d[0] / max(d[1], tiny)) / max(s[0] / max(s[1], tiny), tiny),
        "blue_green": (d[2] / max(d[1], tiny)) / max(s[2] / max(s[1], tiny), tiny),
    }


def stability(per_frame: list[dict]) -> dict:
    """Frame-to-frame jumps and clip-wide drift of each ratio, against STEP_MAX and SPAN_MAX."""
    scored = [f for f in per_frame if f["stats"]["src"]["luma"] >= DARK_LUMA]
    result: dict = {
        "thresholds": {"step_max": STEP_MAX, "span_max": SPAN_MAX, "dark_luma": DARK_LUMA},
        "frames_scored": len(scored),
        "frames_dark": len(per_frame) - len(scored),
        "metrics": {},
    }
    passed = True
    for metric in METRICS:
        values = np.array([f["ratios"][metric] for f in scored], dtype=np.float64)
        frames = [f["frame"] for f in scored]
        record: dict = {"median": None, "span": None, "step_max": None, "worst_step_frame": None, "over_step": []}
        if len(values):
            median = float(np.median(values))
            record.update(median=median, span=float((values.max() - values.min()) / median))
        if len(values) > 1:
            steps = np.abs(values[1:] / values[:-1] - 1.0)
            worst = int(np.argmax(steps))
            record.update(
                step_max=float(steps[worst]),
                worst_step_frame=frames[worst + 1],
                over_step=[frames[i + 1] for i in np.flatnonzero(steps > STEP_MAX)][:50],
            )
        record["passed"] = (record["span"] is None or record["span"] <= SPAN_MAX) and (
            record["step_max"] is None or record["step_max"] <= STEP_MAX
        )
        passed &= record["passed"]
        result["metrics"][metric] = record
    luma = np.array([f["stats"]["src"]["luma"] for f in per_frame], dtype=np.float64)
    if len(luma) > 1:
        source_steps = np.abs(luma[1:] / np.maximum(luma[:-1], 1e-9) - 1.0)
        result["source_level_step_max"] = float(source_steps.max())
        result["source_level_step_median"] = float(np.median(source_steps))
    result["passed"] = bool(passed and scored)
    if not scored:
        result["reason"] = "every frame is darker than the scoring floor"
    return result


def label(image: np.ndarray, text: str) -> np.ndarray:
    import cv2

    scale = max(0.3, image.shape[1] / 1200)
    thickness = max(1, int(round(scale * 2)))
    (width, height), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = int(8 * scale)
    out = image.copy()
    out[: height + base + 2 * pad, : width + 2 * pad] //= 3
    cv2.putText(
        out, text, (pad, pad + height), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
    return out


def side_by_side(src_encoded: np.ndarray, dev_encoded: np.ndarray) -> np.ndarray:
    left = label(colour.to_uint8(src_encoded), LABELS[0])
    right = label(colour.to_uint8(dev_encoded), LABELS[1])
    return np.ascontiguousarray(np.concatenate([left, right], axis=1))


def encoder_command(ffmpeg: str, hw, rate: Fraction, path: Path, *, libx264: bool) -> list[str]:
    """sRGB-encoded RGB frames on stdin to an MP4 with a BT.709 matrix, tagged with the sRGB transfer."""
    codec = (
        ["-c:v", "libx264", "-preset", "medium", "-crf", str(PREVIEW_CRF)]
        if libx264
        else ["-c:v", "mpeg4", "-q:v", "2"]
    )
    return [
        ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{hw[1]}x{hw[0]}",
        "-framerate", f"{rate.numerator}/{rate.denominator}", "-i", "pipe:0",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", *codec,
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "tv",
        "-movflags", "+faststart", str(path),
    ]  # fmt: skip


def thumbnail_frames(count: int) -> list[int]:
    return sorted({0, count // 2, count - 1}) if count else []


def plot(path: Path, per_frame: list[dict], result: dict) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception:  # noqa: BLE001 - the plot is optional
        return None
    frames = [f["frame"] for f in per_frame]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for metric in METRICS:
        top.plot(frames, [f["ratios"][metric] for f in per_frame], label=metric.replace("_", "/"))
    top.set_ylabel("developed / source")
    top.set_title(
        f"Stability: {'passed' if result['passed'] else 'FAILED'} (step <= {STEP_MAX:.1%}, span <= {SPAN_MAX:.0%})"
    )
    top.legend(loc="best")
    top.grid(alpha=0.3)
    bottom.plot(frames, [f["stats"]["src"]["luma"] for f in per_frame], label="source")
    bottom.plot(frames, [f["stats"]["dev"]["luma"] for f in per_frame], label="developed", linestyle="--")
    bottom.set_ylabel("mean linear luma")
    bottom.set_xlabel("frame")
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in (top, bottom):
        axis.ticklabel_format(axis="y", useOffset=False)
    bottom.legend(loc="best")
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def run(run_dir, inputs: dict, settings: dict, log=print) -> dict:
    import cv2

    run_dir = Path(run_dir)
    clip: source.Clip = inputs["clip"]
    tools = inputs["tools"]
    converted = convert.load(run_dir)
    if not converted or not converted.get("frames_converted"):
        raise RuntimeError("step 4 wrote no frames")
    count = int(converted["frames_converted"])
    hw = tuple(converted["size"])
    folder = run_dir / converted["tiff_dir"]
    files = [folder / convert.tiff_name(converted["clip_name"], i) for i in range(count)]
    rate = Fraction(clip.frame_rate)
    out = run_dir / "check"
    out.mkdir(parents=True, exist_ok=True)
    preview = out / "preview_side_by_side.mp4"
    command = encoder_command(tools["ffmpeg"], (hw[0], 2 * hw[1]), rate, preview, libx264=tools.get("libx264", False))
    thumbs = set(thumbnail_frames(count))
    per_frame: list[dict] = []
    encoding = True
    started = time.perf_counter()
    decoded = source.frames(clip, ffmpeg=tools["ffmpeg"], ffmpeg_major=tools.get("major"), limit=count)
    with tempfile.TemporaryFile() as errors:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for index, (src, path) in enumerate(zip(decoded, files, strict=False)):
                t0 = time.perf_counter()
                src = np.clip(src[: hw[0], : hw[1]], 0.0, 1.0)
                dev_encoded = tifffile.imread(path).astype(np.float32) / 65535.0
                t1 = time.perf_counter()
                stats = frame_stats(src, colour.srgb_decode(dev_encoded))
                pair = side_by_side(colour.srgb_encode(src), dev_encoded)
                if encoding:
                    try:
                        encoder.stdin.write(pair.tobytes())
                    except BrokenPipeError:
                        encoding = False  # the encoder died; its exit code and stderr are reported below
                if index in thumbs:
                    small = cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out / f"frame_{index:06d}.jpg"), small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                per_frame.append(
                    {"frame": index, "stats": stats, "ratios": ratios(stats), "read_s": t1 - t0,
                     "total_s": time.perf_counter() - t0}
                )  # fmt: skip
                if (index + 1) % convert.CHECKPOINT_EVERY == 0:
                    log(f"developed and checked {index + 1}/{count}")
        finally:
            decoded.close()
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
            code = encoder.wait()
        if code != 0:
            errors.seek(0)
            log(f"preview encoder exited {code}: {errors.read().decode('utf-8', 'replace').strip()[-500:]}")
    result = stability(per_frame)
    result.update(
        frames_checked=len(per_frame),
        frames_expected=count,
        wall_s=time.perf_counter() - started,
        read_seconds_mean=float(np.mean([f["read_s"] for f in per_frame])) if per_frame else None,
        preview=str(preview.relative_to(run_dir)) if code == 0 and preview.exists() else None,
        preview_encoder="libx264" if tools.get("libx264") else "mpeg4",
        thumbnails=sorted(str(p.relative_to(run_dir)) for p in out.glob("frame_*.jpg")),
        develop=(
            "the step 4 TIFFs, developed by LibRaw (rawpy) as in v6: as-shot neutral, AHD demosaic, linear sRGB, "
            "then the sRGB transfer; compared with the decoded source frame clipped to [0, 1]"
        ),
        per_frame=per_frame,
    )
    if len(per_frame) != count:
        result["passed"] = False
        result["reason"] = f"developed {len(per_frame)} of {count} frames"
    plotted = plot(out / "stability.png", per_frame, result)
    result["plot"] = str(plotted.relative_to(run_dir)) if plotted else None
    _write_json(run_dir / "check.json", result)
    metrics = result["metrics"]
    worst = ", ".join(f"{m} step {metrics[m]['step_max'] or 0:.3%} span {metrics[m]['span'] or 0:.3%}" for m in METRICS)
    log(f"stability {'passed' if result['passed'] else 'FAILED'}: {worst}")
    if code != 0:
        raise RuntimeError(f"the preview encoder failed (exit {code})")
    return result
