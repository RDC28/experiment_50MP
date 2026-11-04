"""The source clip: ffprobe metadata and every decoded frame as linear BT.709 RGB (ffmpeg).

ffmpeg only demuxes and decodes. It hands over the decoder's own planes (repacked to a planar layout
of the same bit depth, subsampling and range, so no colour conversion happens inside ffmpeg), and
`video_v1.ycbcr` does chroma upsampling, the matrix, the range and the transfer in numpy. Rotation
metadata is applied after the conversion, so chroma siting stays correct. Audio is never read.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from video_v1 import ycbcr

RGB_PREFIXES = ("gbr", "rgb", "bgr", "argb", "abgr", "0rgb", "0bgr", "x2rgb", "x2bgr")
RGB_8_BIT = {"gbrp", "gbrap", "rgb24", "bgr24", "rgba", "bgra", "argb", "abgr", "rgb0", "bgr0", "0rgb", "0bgr"}
_PLANAR_YUV = re.compile(r"^yuva?(420|422|444)p(\d+)?(le|be)?$")
_SEMI_PLANAR = {"nv12": ("420", 8), "nv21": ("420", 8), "nv16": ("422", 8), "nv24": ("444", 8), "nv42": ("444", 8)}
_PACKED_422 = {"yuyv422": 8, "uyvy422": 8, "yvyu422": 8, "y210le": 10, "y212le": 12}
_SUBSAMPLING = {"420": (2, 2), "422": (2, 1), "444": (1, 1)}
_HIGH_BIT_DEPTHS = (9, 10, 12, 14, 16)


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    """The planar format ffmpeg is asked for: same bit depth, subsampling and range as the decoder."""

    pix_fmt: str
    rgb: bool
    bits: int
    subsampling: tuple[int, int]
    full_range_format: bool
    converted_by_ffmpeg: str | None = None

    def plane_shapes(self, height: int, width: int) -> list[tuple[int, int]]:
        sx, sy = self.subsampling
        chroma = (-(-height // sy), -(-width // sx))
        return [(height, width), chroma, chroma]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8 if self.bits == 8 else "<u2")

    def frame_bytes(self, height: int, width: int) -> int:
        return sum(h * w for h, w in self.plane_shapes(height, width)) * self.dtype.itemsize


def layout_for(pix_fmt: str) -> Layout:
    """Map a decoder pixel format to a planar one with the same samples (or say what ffmpeg converts)."""
    name = pix_fmt or ""
    if name.startswith(RGB_PREFIXES):
        bits = 8 if name in RGB_8_BIT else 16
        target = "gbrp" if bits == 8 else "gbrp16le"
        note = None if name in {"gbrp", "gbrp16le"} else f"{name} repacked to {target}"
        return Layout(target, True, bits, (1, 1), True, note)
    if name.startswith("yuvj"):
        sub = name[4:7]
        if sub in _SUBSAMPLING:
            return Layout(f"yuvj{sub}p", False, 8, _SUBSAMPLING[sub], True)
    match = _PLANAR_YUV.match(name)
    if match and (match.group(2) is None or int(match.group(2)) in _HIGH_BIT_DEPTHS):
        return _yuv(match.group(1), int(match.group(2) or 8))
    if name in _SEMI_PLANAR:
        return _yuv(*_SEMI_PLANAR[name])
    found = re.match(r"^p(0|2|4)(\d\d)(le|be)$", name)
    if found and int(found.group(2)) in _HIGH_BIT_DEPTHS:
        return _yuv({"0": "420", "2": "422", "4": "444"}[found.group(1)], int(found.group(2)))
    if name in _PACKED_422:
        return _yuv("422", _PACKED_422[name])
    if name.startswith(("gray", "ya", "pal", "bayer", "monob", "monow")):
        raise SourceError(f"pixel format {name!r} is not a colour video format this notebook converts")
    return Layout("yuv444p16le", False, 16, (1, 1), False, f"{name} converted to yuv444p16le by ffmpeg (swscale)")


def _yuv(sub: str, bits: int) -> Layout:
    suffix = "" if bits == 8 else f"{bits}le"
    return Layout(f"yuv{sub}p{suffix}", False, bits, _SUBSAMPLING[sub], False)


@dataclass
class Clip:
    """What the report states about the source (section 1 of the README)."""

    path: str
    container: str
    codec: str
    profile: str | None
    pix_fmt: str
    coded_size: list[int]  # [height, width] as decoded, before rotation
    size: list[int]  # [height, width] after rotation
    rotation_ccw: int
    frame_rate: str  # exact fraction, e.g. "30000/1001"
    frame_rate_source: str
    frames: int
    duration_s: float | None
    bit_rate: int | None
    audio_streams: list[dict]
    signal: dict
    layout: dict
    probe_warnings: list[str] = field(default_factory=list)

    @property
    def fps(self) -> Fraction:
        return Fraction(self.frame_rate)

    def as_dict(self) -> dict:
        record = asdict(self)
        record["frame_rate_float"] = float(self.fps)
        return record


def tools() -> dict:
    """ffmpeg and ffprobe paths and versions; raises SourceError when either is missing."""
    found = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    missing = [name for name, path in found.items() if path is None]
    if missing:
        raise SourceError(f"{' and '.join(missing)} not found on PATH")
    version = subprocess.run([found["ffmpeg"], "-hide_banner", "-version"], capture_output=True, text=True).stdout
    encoders = subprocess.run([found["ffmpeg"], "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    first = version.splitlines()[0] if version else ""
    major = re.search(r"version\s+n?(\d+)\.", first)
    return {
        "ffmpeg": found["ffmpeg"],
        "ffprobe": found["ffprobe"],
        "version": first,
        "major": int(major.group(1)) if major else None,
        "libx264": " libx264 " in encoders,
    }


def _fraction(text: str | None) -> Fraction | None:
    try:
        value = Fraction(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def rotation_ccw(stream: dict) -> int:
    """Counter-clockwise display rotation in degrees, a multiple of 90 (display matrix or legacy tag)."""
    degrees = None
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            degrees = float(side["rotation"])
    if degrees is None and "rotate" in (stream.get("tags") or {}):
        degrees = -float(stream["tags"]["rotate"])  # the legacy tag is clockwise
    if degrees is None:
        return 0
    quarter = round(degrees / 90.0)
    if abs(degrees - 90.0 * quarter) > 1.0:
        raise SourceError(f"rotation {degrees} degrees is not a multiple of 90")
    return int(quarter % 4) * 90


def probe(path: str | Path, ffprobe: str = "ffprobe") -> Clip:
    """Read the first video stream's metadata, counting its packets (fast; no decoding)."""
    done = subprocess.run(
        [ffprobe, "-v", "error", "-count_packets", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise SourceError(f"ffprobe could not read {Path(path).name}: {done.stderr.strip()[-500:]}")
    data = json.loads(done.stdout or "{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video" and not _is_cover_art(s)), None)
    if video is None:
        raise SourceError(f"{Path(path).name} has no video stream")
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if width < 2 or height < 2:
        raise SourceError(f"video stream size {width}x{height} is not usable")
    layout = layout_for(video.get("pix_fmt", ""))
    signal = ycbcr.resolve(
        video,
        width=width,
        height=height,
        bits=layout.bits,
        subsampling=layout.subsampling,
        rgb=layout.rgb,
        full_range_format=layout.full_range_format,
    )
    warnings = list(signal.warnings)
    rate, rate_source = _fraction(video.get("avg_frame_rate")), "avg_frame_rate"
    real = _fraction(video.get("r_frame_rate"))
    if rate is None:
        rate, rate_source = real, "r_frame_rate"
    if rate is None:
        raise SourceError("the video stream states no frame rate")
    if real is not None and abs(float(real) - float(rate)) > 0.01 * float(rate):
        warnings.append(f"variable frame rate: average {float(rate):.3f} fps, nominal {float(real):.3f} fps")
    rate = rate.limit_denominator(1001 * 1000)
    frames = int(video.get("nb_read_packets") or video.get("nb_frames") or 0)
    if frames < 1:
        raise SourceError("the video stream has no frames")
    rotation = rotation_ccw(video)
    size = [height, width] if rotation in (0, 180) else [width, height]
    fmt = data.get("format") or {}
    audio = [
        {"index": s.get("index"), "codec": s.get("codec_name"), "channels": s.get("channels")}
        for s in streams
        if s.get("codec_type") == "audio"
    ]
    if layout.converted_by_ffmpeg:
        warnings.append(layout.converted_by_ffmpeg)
    duration = video.get("duration") or fmt.get("duration")
    bit_rate = video.get("bit_rate") or fmt.get("bit_rate")
    return Clip(
        path=str(path),
        container=fmt.get("format_name", ""),
        codec=video.get("codec_name", ""),
        profile=video.get("profile"),
        pix_fmt=video.get("pix_fmt", ""),
        coded_size=[height, width],
        size=size,
        rotation_ccw=rotation,
        frame_rate=f"{rate.numerator}/{rate.denominator}",
        frame_rate_source=rate_source,
        frames=frames,
        duration_s=float(duration) if duration else None,
        bit_rate=int(bit_rate) if bit_rate else None,
        audio_streams=audio,
        signal=signal.as_dict(),
        layout=asdict(layout),
        probe_warnings=warnings,
    )


def _is_cover_art(stream: dict) -> bool:
    return bool((stream.get("disposition") or {}).get("attached_pic"))


def signal_of(clip: Clip) -> ycbcr.Signal:
    record = dict(clip.signal)
    record["subsampling"] = tuple(record["subsampling"])
    record["assumed"] = tuple(record["assumed"])
    record["warnings"] = tuple(record["warnings"])
    return ycbcr.Signal(**record)


def layout_of(clip: Clip) -> Layout:
    record = dict(clip.layout)
    record["subsampling"] = tuple(record["subsampling"])
    return Layout(**record)


def decode_command(clip: Clip, ffmpeg: str = "ffmpeg", ffmpeg_major: int | None = None) -> list[str]:
    """Every decoded frame once, in order, as raw planes; no rotation, scaling or audio."""
    passthrough = (
        ["-vsync", "passthrough"]
        if ffmpeg_major is not None and ffmpeg_major < 5
        else [
            "-fps_mode",
            "passthrough",
        ]
    )
    return [
        ffmpeg, "-v", "error", "-nostdin", "-noautorotate", "-i", clip.path,
        "-map", "0:v:0", "-an", "-sn", "-dn", *passthrough,
        "-f", "rawvideo", "-pix_fmt", layout_of(clip).pix_fmt, "pipe:1",
    ]  # fmt: skip


def frames(
    clip: Clip, *, ffmpeg: str = "ffmpeg", ffmpeg_major: int | None = None, limit: int | None = None
) -> Iterator[np.ndarray]:
    """Yield linear BT.709 RGB frames (float32, rotated for display, 1.0 = sensor clip, unclipped)."""
    layout, signal = layout_of(clip), signal_of(clip)
    height, width = clip.coded_size
    shapes = layout.plane_shapes(height, width)
    sizes = [h * w * layout.dtype.itemsize for h, w in shapes]
    frame_bytes = sum(sizes)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            decode_command(clip, ffmpeg, ffmpeg_major), stdout=subprocess.PIPE, stderr=errors, bufsize=frame_bytes
        )
        count, ended = 0, False
        try:
            while limit is None or count < limit:
                buffer = _read_exactly(process.stdout, frame_bytes)
                if buffer is None:
                    ended = True
                    break
                planes, offset = [], 0
                for (h, w), size in zip(shapes, sizes, strict=True):
                    planes.append(np.frombuffer(buffer, dtype=layout.dtype, count=h * w, offset=offset).reshape(h, w))
                    offset += size
                linear = ycbcr.frame_to_linear(planes, signal)
                yield np.ascontiguousarray(np.rot90(linear, clip.rotation_ccw // 90)) if clip.rotation_ccw else linear
                count += 1
        finally:
            if not ended and process.poll() is None:
                process.kill()  # stopped early (a frame limit, or the consumer closed the generator)
            process.stdout.close()
            code = process.wait()
        if ended and code != 0:
            errors.seek(0)
            raise SourceError(f"ffmpeg exited {code}: {errors.read().decode('utf-8', 'replace').strip()[-800:]}")
        if count == 0:
            raise SourceError("ffmpeg decoded no frames")


def _read_exactly(stream, size: int) -> bytes | None:
    """One whole frame, or None at the end of the stream; a partial frame is an error."""
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining == size:
        return None
    if remaining:
        raise SourceError(f"ffmpeg ended mid-frame ({size - remaining} of {size} bytes)")
    return b"".join(chunks)


def even_crop(hw) -> tuple[int, int]:
    """The RGGB mosaic needs even sides; an odd last row or column is dropped (and reported)."""
    return int(hw[0]) // 2 * 2, int(hw[1]) // 2 * 2
