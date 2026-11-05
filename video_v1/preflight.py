"""Driver step 3: runtime, tools, the input clip and free disk, before any frame is converted (README section 6).

Any failure raises PreflightError, which makes the run INVALID with the reason. A runtime without a
GPU only warns: nothing in this notebook uses the GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.parse
from contextlib import closing
from pathlib import Path

from prd_v6 import preflight as v6preflight
from prd_v6.delivery import _write_json
from video_v1 import source

MIN_RAM_GB = 8.0
DISK_MARGIN = 2.2  # the outputs, their zip copies, and room for the preview
DISK_RESERVE_GB = 1.0
DNG_HEADER_BYTES = 64 * 1024
UPLOAD_RECORD = "inputs/upload.json"


class PreflightError(RuntimeError):
    pass


def gpu_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return names[0] if done.returncode == 0 and names else None


def classify(gpu: str | None, ram_gb: float) -> tuple[list[str], list[str]]:
    """(problems that stop the run, warnings that do not)."""
    problems, warnings = [], []
    if not gpu:
        warnings.append(
            "no GPU: continuing on CPU (nothing in this notebook uses the GPU; A100 or L4 is the stated tier)"
        )
    elif not any(name in gpu for name in ("A100", "H100", "L4")):
        warnings.append(f"{gpu} is not the stated A100 or L4 tier; continuing (the GPU is not used)")
    if ram_gb < MIN_RAM_GB:
        problems.append(f"host RAM {ram_gb:.1f} GB < {MIN_RAM_GB:.0f} GB")
    return problems, warnings


def disk_needed_gb(hw, frames: int, *, write_dng: bool = False) -> float:
    """Per frame: the TIFF (6 bytes a pixel before deflate), the 14-bit MLV frame and the DNG when written."""
    per_frame = hw[0] * hw[1] * (6 + 14 / 8) + (hw[0] * hw[1] * 2 + DNG_HEADER_BYTES if write_dng else 0)
    return per_frame * frames * DISK_MARGIN / 1e9 + DISK_RESERVE_GB


def video_dest(inputs: Path, url: str) -> Path:
    """Where a downloaded clip goes: its own name for a plain URL, a fixed name for a Drive link."""
    name = "" if v6preflight.drive_file_id(url) else Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._") or "source_video"
    return Path(inputs) / safe


def resolve_video(run_dir: Path, video, log=print) -> tuple[Path, dict]:
    """The clip on local disk, and how it got there (upload, download or a runtime path)."""
    inputs = run_dir / "inputs"
    if video is None:
        record = json.loads((run_dir / UPLOAD_RECORD).read_text()) if (run_dir / UPLOAD_RECORD).exists() else None
        if not record:
            raise PreflightError("VIDEO is None and no file was uploaded in step 1")
        return Path(record["path"]), {"how": "uploaded", **record}
    video = str(video).strip()
    if video.lower().startswith(("http://", "https://")):
        try:
            got = v6preflight.download(video, video_dest(inputs, video), size=None, digest=None, log=log)
        except Exception as error:  # noqa: BLE001 - any download failure stops the run with its reason
            raise PreflightError(f"could not download VIDEO: {error}") from error
        return Path(got["path"]), {"how": "downloaded", **got}
    path = Path(video).expanduser()
    if not path.is_file():
        hint = " (is Google Drive mounted? set the path under /content/drive/MyDrive)" if "/drive/" in video else ""
        raise PreflightError(f"VIDEO {video!r} is not a file on the runtime{hint}")
    return path, {"how": "runtime path", "path": str(path), "bytes": path.stat().st_size}


def run(run_dir, settings: dict, log=print) -> dict:
    from prd_v5.dng_validate import ENV_VAR, find_dng_validate
    from prd_v6.install import DNG_VALIDATE_BUILD

    run_dir = Path(run_dir)
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    tier = {"gpu": gpu_name(), "ram_gb": ram_gb, "cpus": os.cpu_count()}
    problems, warnings = classify(tier["gpu"], ram_gb)
    tier.update(problems=problems, warnings=warnings)
    environment: dict = {"tier": tier, "versions": v6preflight.versions()}
    _write_json(run_dir / "environment.json", environment)
    for warning in warnings:
        log(f"preflight warning: {warning}")
    log(f"runtime: GPU {tier['gpu'] or 'none'}, RAM {ram_gb:.0f} GB, {tier['cpus']} CPUs")
    if problems:
        raise PreflightError("; ".join(problems))
    try:
        tools = source.tools()
    except source.SourceError as error:
        raise PreflightError(str(error)) from error
    if not tools["libx264"]:
        warnings.append("ffmpeg has no libx264: the preview uses MPEG-4 Part 2 at high quality instead")
        log(f"preflight warning: {warnings[-1]}")
    build = json.loads((run_dir / DNG_VALIDATE_BUILD).read_text()) if (run_dir / DNG_VALIDATE_BUILD).exists() else {}
    tool = find_dng_validate(build.get("binary")) if build.get("status") == "built" else None
    if tool is None:
        raise PreflightError(f"dng_validate is not available: {build.get('reason') or 'setup did not build it'}")
    os.environ[ENV_VAR] = str(tool)
    path, origin = resolve_video(run_dir, settings.get("VIDEO"), log)
    try:
        clip = source.probe(path, tools["ffprobe"])
        with closing(source.frames(clip, ffmpeg=tools["ffmpeg"], ffmpeg_major=tools["major"], limit=1)) as decoded:
            first = next(decoded)
    except source.SourceError as error:
        raise PreflightError(f"{path.name}: {error}") from error
    hw = source.even_crop(clip.size)
    need = disk_needed_gb(hw, clip.frames, write_dng=bool(settings.get("WRITE_DNG")))
    free = shutil.disk_usage(run_dir).free / 1e9
    record = clip.as_dict()
    record.update(
        origin=origin,
        output_size=list(hw),
        cropped_to_even=tuple(hw) != tuple(clip.size),
        first_frame={"min": float(first.min()), "max": float(first.max()), "mean": float(first.mean())},
        audio="dropped: the DNG sequence has no audio" if clip.audio_streams else "none in the source",
    )
    _write_json(run_dir / "source.json", record)
    log(
        f"source: {clip.codec} {clip.pix_fmt} {clip.size[1]}x{clip.size[0]} at {float(clip.fps):.3f} fps, "
        f"{clip.frames} frames, {clip.signal['matrix']}/{clip.signal['transfer']}/"
        f"{'full' if clip.signal['full_range'] else 'limited'} range; {len(clip.audio_streams)} audio stream(s) dropped"
    )
    for warning in clip.probe_warnings:
        log(f"source warning: {warning}")
    environment.update(
        tools=tools,
        dng_validate=str(tool),
        disk={"free_gb": free, "needed_gb": need, "frames": clip.frames},
    )
    _write_json(run_dir / "environment.json", environment)
    if free < need:
        raise PreflightError(
            f"free disk {free:.1f} GB < {need:.1f} GB needed for {clip.frames} frames at {hw[1]}x{hw[0]}"
        )
    return {"clip": clip, "tools": tools, "dng_validate": str(tool), "path": path}
