"""Magic Lantern Video (MLV v2.0) raw clip writer and reader (README section 2).

Written from the open format definition, Magic Lantern's `mlv.h` and `raw.h`, with MLV App's reader as
the reference. Every structure is packed and little-endian. A clip is one MLVI file header, then the
metadata blocks (RAWI, RAWC, IDNT, EXPO, WBAL), then one VIDF block per frame. Frames hold the Bayer
mosaic bit-packed the way the camera writes it: the samples form one big-endian bit stream that is
stored as little-endian 16-bit words.

A clip larger than `chunk_bytes` spans files the way Magic Lantern spans them: `<clip>.MLV`, then
`<clip>.M00`, `<clip>.M01` and onwards, each starting with its own MLVI header carrying the same GUID.
MLV App opens the .MLV and loads the other chunks next to it as one clip.

The reader parses the files back, checks the headers and decodes frames, so a test and the convert
step can prove the clip holds exactly the mosaic written.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

VERSION = b"v2.0"
VIDEO_CLASS_RAW = 0x01
WB_KELVIN = 9
# raw_info.cfa_pattern: the CFA colours (0 red, 1 green, 2 blue) of the 2x2 tile, one per byte, LSB first.
CFA_CODES = {"RGGB": 0x02010100, "BGGR": 0x00010102, "GRBG": 0x01000201, "GBRG": 0x01020001}
SUPPORTED_BITS = (10, 12, 14)
MATRIX_DENOMINATOR = 10000

FILE_HEADER = struct.Struct("<4sI8sQHHIHHIIII")  # mlv_file_hdr_t, 52 bytes
BLOCK_HEADER = struct.Struct("<4sIQ")  # type, size, timestamp (microseconds)
RAW_INFO = struct.Struct("<IIiiiiiii4i4i2iii18ii")  # raw_info_t, 160 bytes
RAWI = struct.Struct("<HH")  # xRes, yRes, then raw_info
RAWC = struct.Struct("<HHHHBBBBhh")
IDNT = struct.Struct("<32sI32s")
EXPO = struct.Struct("<IIIIQ")
WBAL = struct.Struct("<IIIIIII")
VIDF = struct.Struct("<IHHHHI")  # frameNumber, cropPosX, cropPosY, panPosX, panPosY, frameSpace
assert FILE_HEADER.size == 52 and RAW_INFO.size == 160 and BLOCK_HEADER.size + RAWI.size + RAW_INFO.size == 180


def chunk_name(stem: str, number: int) -> str:
    """Chunk 0 is `<stem>.MLV`; chunk n >= 1 is `<stem>.M<n-1:02d>`, as Magic Lantern names them."""
    return f"{stem}.MLV" if number == 0 else f"{stem}.M{number - 1:02d}"


def mlv_width(width: int) -> int:
    """MLV frames are a whole number of 16-bit words per row: the width is cropped to a multiple of 8."""
    return int(width) // 8 * 8


def pack(samples: np.ndarray, bits: int) -> bytes:
    """Bit-pack (H, W) samples below 2**bits into the MLV frame layout."""
    flat = np.ascontiguousarray(samples, dtype=">u2").reshape(-1)
    if flat.size * bits % 16:
        raise ValueError("a packed frame must fill whole 16-bit words")
    stream = np.unpackbits(flat.view(np.uint8).reshape(-1, 2), axis=1)[:, 16 - bits :]
    return np.packbits(stream.reshape(-1)).view(">u2").astype("<u2").tobytes()


def unpack(data: bytes, hw: tuple[int, int], bits: int) -> np.ndarray:
    """Inverse of `pack`: the (H, W) uint16 samples of one frame."""
    count = hw[0] * hw[1]
    words = np.frombuffer(data, dtype="<u2", count=count * bits // 16).astype(">u2")
    stream = np.unpackbits(words.view(np.uint8)).reshape(count, bits)
    padded = np.zeros((count, 16), dtype=np.uint8)
    padded[:, 16 - bits :] = stream
    return np.packbits(padded, axis=1).view(">u2").reshape(hw).astype(np.uint16)


def reduce_bits(samples16: np.ndarray, bits: int) -> np.ndarray:
    """16-bit stored samples to `bits`: divide by 2**(16 - bits) with rounding, clipped to the range."""
    shift = 16 - bits
    reduced = (np.asarray(samples16, dtype=np.uint32) + (1 << (shift - 1))) >> shift
    return np.minimum(reduced, (1 << bits) - 1).astype(np.uint16)


def frame_timestamp_us(index: int, frame_rate) -> int:
    """A VIDF block's timestamp: the frame's start time in microseconds at the clip frame rate."""
    return int(round(index * 1_000_000 / Fraction(frame_rate)))


def _matrix_rationals(matrix) -> list[int]:
    out: list[int] = []
    for value in np.asarray(matrix, dtype=np.float64).reshape(9):
        out += [int(round(value * MATRIX_DENOMINATOR)), MATRIX_DENOMINATOR]
    return out


def _text(value: str, length: int) -> bytes:
    return value.encode("ascii", "replace")[: length - 1].ljust(length, b"\0")


@dataclass
class ClipInfo:
    """What the metadata blocks say about the clip."""

    width: int
    height: int
    bits: int
    black: int
    white: int
    cfa_pattern: str
    color_matrix: list[list[float]]
    frame_rate: Fraction
    camera_name: str = "Synthetic exp50 video_v1"
    kelvin: int = 6500
    wb_gains: tuple[int, int, int] = (1024, 1024, 1024)
    iso: int = 100
    shutter_us: int = 0
    dynamic_range_ev: float = 12.9
    extra: dict = field(default_factory=dict)


class Writer:
    """Streams frames into a (possibly spanned) MLV clip; `close` fixes up the frame counts."""

    def __init__(self, folder, stem: str, info: ClipInfo, *, chunk_bytes: int = 0, guid: int | None = None):
        if info.bits not in SUPPORTED_BITS:
            raise ValueError(f"MLV App reads {SUPPORTED_BITS}-bit raw; got {info.bits}")
        if info.width % 8 or info.height % 2:
            raise ValueError("an MLV frame needs a width that is a multiple of 8 and an even height")
        if info.cfa_pattern not in CFA_CODES:
            raise ValueError(f"unknown CFA pattern {info.cfa_pattern}")
        self.folder, self.stem, self.info = Path(folder), stem, info
        self.folder.mkdir(parents=True, exist_ok=True)
        self.chunk_bytes = int(chunk_bytes)
        self.guid = guid if guid is not None else int.from_bytes(hashlib.sha256(stem.encode()).digest()[:8], "little")
        self.frame_size = info.width * info.height * info.bits // 8
        self.frames = 0
        self.paths: list[Path] = []
        self._chunk_frames: list[int] = []
        self._handle = None
        self._open_chunk()
        self._write_metadata()

    # -------------------------------------------------------------- structure

    def _file_header(self, number: int, frames: int, chunks: int) -> bytes:
        rate = Fraction(self.info.frame_rate)
        return FILE_HEADER.pack(
            b"MLVI", FILE_HEADER.size, VERSION.ljust(8, b"\0"), self.guid, number, chunks, 0,
            VIDEO_CLASS_RAW, 0, frames, 0, rate.numerator, rate.denominator,
        )  # fmt: skip

    def _open_chunk(self) -> None:
        if self._handle is not None:
            self._handle.close()
        path = self.folder / chunk_name(self.stem, len(self.paths))
        self._handle = open(path, "wb")
        self.paths.append(path)
        self._chunk_frames.append(0)
        self._handle.write(self._file_header(len(self.paths) - 1, 0, 0))

    def _block(self, kind: bytes, payload: bytes, timestamp: int = 0) -> None:
        self._handle.write(BLOCK_HEADER.pack(kind, BLOCK_HEADER.size + len(payload), timestamp) + payload)

    def _write_metadata(self) -> None:
        info = self.info
        pitch = info.width * info.bits // 8
        raw_info = RAW_INFO.pack(
            1, 0, info.height, info.width, pitch, self.frame_size, info.bits, info.black, info.white,
            0, 0, info.width, info.height,  # crop: x, y, width, height
            0, 0, info.height, info.width,  # active area: y1, x1, y2, x2
            0, 0,  # exposure bias
            CFA_CODES[info.cfa_pattern], 21,  # calibration illuminant 1: D65
            *_matrix_rationals(info.color_matrix), int(round(info.dynamic_range_ev * 100)),
        )  # fmt: skip
        self._block(b"RAWI", RAWI.pack(info.width, info.height) + raw_info)
        self._block(b"RAWC", RAWC.pack(info.width, info.height, 100, 0, 1, 0, 1, 0, 0, 0))
        self._block(b"IDNT", IDNT.pack(_text(info.camera_name, 32), 0, _text("synthetic", 32)))
        self._block(b"EXPO", EXPO.pack(0, info.iso, info.iso, 0, info.shutter_us))
        self._block(b"WBAL", WBAL.pack(WB_KELVIN, info.kelvin, *info.wb_gains, 0, 0))

    # -------------------------------------------------------------- frames

    def timestamp(self, index: int) -> int:
        return frame_timestamp_us(index, self.info.frame_rate)

    def write(self, samples: np.ndarray) -> None:
        """Append one (H, W) frame of samples already at `info.bits`."""
        if samples.shape != (self.info.height, self.info.width):
            raise ValueError(f"frame {samples.shape} != {(self.info.height, self.info.width)}")
        if int(np.max(samples)) >= 1 << self.info.bits:
            raise ValueError(f"a sample does not fit in {self.info.bits} bits")
        block = BLOCK_HEADER.size + VIDF.size + self.frame_size
        if self.chunk_bytes and self._chunk_frames[-1] and self._handle.tell() + block > self.chunk_bytes:
            self._open_chunk()
        header = VIDF.pack(self.frames, 0, 0, 0, 0, 0)
        self._block(b"VIDF", header + pack(samples, self.info.bits), self.timestamp(self.frames))
        self.frames += 1
        self._chunk_frames[-1] += 1

    def close(self) -> list[Path]:
        """Close the last chunk and write the frame counts: the clip total in the first file's header,
        each later chunk's own count in its header, and the chunk count in all of them."""
        if self._handle is None:
            return self.paths
        self._handle.close()
        self._handle = None
        for number, path in enumerate(self.paths):
            frames = self.frames if number == 0 else self._chunk_frames[number]
            with open(path, "r+b") as handle:
                handle.write(self._file_header(number, frames, len(self.paths)))
        return self.paths

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ reader


@dataclass
class Clip:
    paths: list[Path]
    headers: list[dict]
    blocks: dict
    info: dict
    frames: list[tuple[int, Path, int, int]]  # (frame number, file, data offset, timestamp)

    @property
    def hw(self) -> tuple[int, int]:
        return self.info["height"], self.info["width"]

    def frame(self, number: int) -> np.ndarray:
        _, path, offset, _ = self.frames[number]
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(self.info["frame_size"])
        return unpack(data, self.hw, self.info["bits"])


def chunk_paths(first: str | Path) -> list[Path]:
    first = Path(first)
    paths = [first]
    for number in range(1, 101):
        candidate = first.with_name(chunk_name(first.stem, number))
        if not candidate.exists():
            break
        paths.append(candidate)
    return paths


def read(first: str | Path) -> Clip:
    """Parse a clip (and its chunks) into headers, metadata blocks and a frame index."""
    paths = chunk_paths(first)
    headers, blocks, frames = [], {}, []
    for path in paths:
        with open(path, "rb") as handle:
            raw = handle.read(FILE_HEADER.size)
            fields = FILE_HEADER.unpack(raw)
            if fields[0] != b"MLVI" or fields[1] != FILE_HEADER.size:
                raise ValueError(f"{path.name}: not an MLV file header")
            keys = ("guid", "file_num", "file_count", "file_flags", "video_class", "audio_class",
                    "video_frames", "audio_frames", "fps_nom", "fps_denom")  # fmt: skip
            header = dict(zip(keys, fields[3:], strict=True), version=fields[2].rstrip(b"\0").decode())
            headers.append(header)
            while True:
                head = handle.read(BLOCK_HEADER.size)
                if len(head) < BLOCK_HEADER.size:
                    break
                kind, size, timestamp = BLOCK_HEADER.unpack(head)
                if size < BLOCK_HEADER.size:
                    raise ValueError(f"{path.name}: block {kind!r} has size {size}")
                start = handle.tell()
                if kind == b"VIDF":
                    number, *_, space = VIDF.unpack(handle.read(VIDF.size))
                    frames.append((number, path, start + VIDF.size + space, timestamp))
                else:
                    blocks.setdefault(kind.decode("ascii", "replace"), []).append(handle.read(size - BLOCK_HEADER.size))
                handle.seek(start + size - BLOCK_HEADER.size)
    if "RAWI" not in blocks:
        raise ValueError("no RAWI block")
    x_res, y_res = RAWI.unpack(blocks["RAWI"][0][: RAWI.size])
    raw = RAW_INFO.unpack(blocks["RAWI"][0][RAWI.size : RAWI.size + RAW_INFO.size])
    matrix = np.array(raw[21:39], dtype=np.float64).reshape(9, 2)
    info = {
        "x_res": x_res,
        "y_res": y_res,
        "api_version": raw[0],
        "height": raw[2],
        "width": raw[3],
        "pitch": raw[4],
        "frame_size": raw[5],
        "bits": raw[6],
        "black": raw[7],
        "white": raw[8],
        "active_area": list(raw[13:17]),
        "cfa_pattern": next((k for k, v in CFA_CODES.items() if v == raw[19]), hex(raw[19])),
        "calibration_illuminant": raw[20],
        "color_matrix": (matrix[:, 0] / matrix[:, 1]).reshape(3, 3).tolist(),
        "dynamic_range_ev": raw[39] / 100,
    }
    if "EXPO" in blocks:
        _iso_mode, iso, _analog, _digital, shutter_us = EXPO.unpack(blocks["EXPO"][0][: EXPO.size])
        info.update(iso=iso, shutter_us=shutter_us)
    if "WBAL" in blocks:
        _mode, kelvin, *gains, _shift_gm, _shift_ba = WBAL.unpack(blocks["WBAL"][0][: WBAL.size])
        info.update(kelvin=kelvin, wb_gains=gains)
    frames.sort(key=lambda f: f[0])
    return Clip(paths=paths, headers=headers, blocks=blocks, info=info, frames=frames)


def verify(first: str | Path, expected: dict) -> dict:
    """Read the clip back and check it against what was written.

    `expected` holds what `expected_metadata` gives for the written `ClipInfo`, the frame count (`frames`)
    and, per frame, the sha256 of the samples written (`frame_sha256`); every frame is decoded and
    must hash the same, so the clip is proven bit for bit. The EXPO and WBAL values and every VIDF
    timestamp are checked too.
    """
    problems: list[str] = []
    try:
        clip = read(first)
    except (OSError, ValueError, struct.error) as error:
        return {"status": "failed", "problems": [f"{type(error).__name__}: {error}"]}
    info, first_header = clip.info, clip.headers[0]
    for key in ("width", "height", "bits", "black", "white", "cfa_pattern"):
        if info[key] != expected[key]:
            problems.append(f"RAWI {key} {info[key]} != {expected[key]}")
    if info["pitch"] * info["height"] != info["frame_size"] or info["frame_size"] * 8 != (
        info["width"] * info["height"] * info["bits"]
    ):
        problems.append("RAWI pitch and frame size do not match the resolution and bit depth")
    if not np.allclose(info["color_matrix"], expected["color_matrix"], atol=1.0 / MATRIX_DENOMINATOR):
        problems.append("RAWI colour matrix differs from the DNG ColorMatrix1")
    for kind in ("RAWC", "IDNT", "EXPO", "WBAL"):
        if kind not in clip.blocks:
            problems.append(f"no {kind} block")
    for kind, keys in (("EXPO", ("iso", "shutter_us")), ("WBAL", ("kelvin", "wb_gains"))):
        for key in keys:
            if kind in clip.blocks and info[key] != expected[key]:
                problems.append(f"{kind} {key} {info[key]} != {expected[key]}")
    rate = Fraction(expected["frame_rate"])
    if Fraction(first_header["fps_nom"], first_header["fps_denom"] or 1) != rate:
        problems.append(f"frame rate {first_header['fps_nom']}/{first_header['fps_denom']} != {rate}")
    if {h["guid"] for h in clip.headers} != {first_header["guid"]}:
        problems.append("chunks carry different GUIDs")
    if [h["file_num"] for h in clip.headers] != list(range(len(clip.paths))):
        problems.append("chunk numbers are out of order")
    if first_header["video_frames"] != expected["frames"] or len(clip.frames) != expected["frames"]:
        problems.append(f"{first_header['video_frames']} frames in the header, {len(clip.frames)} VIDF blocks, "
                        f"expected {expected['frames']}")  # fmt: skip
    if [f[0] for f in clip.frames] != list(range(len(clip.frames))):
        problems.append("VIDF frame numbers are not 0, 1, 2, ...")
    late = [number for number, *_, stamp in clip.frames if stamp != frame_timestamp_us(number, rate)]
    if late:
        problems.append(f"{len(late)} VIDF timestamps do not match the frame rate (first {late[:10]})")
    mismatched = [
        number
        for number, digest in enumerate(expected.get("frame_sha256", []))
        if number < len(clip.frames) and hashlib.sha256(clip.frame(number).tobytes()).hexdigest() != digest
    ]
    if mismatched:
        problems.append(f"{len(mismatched)} decoded frames differ from the written mosaic (first {mismatched[:10]})")
    return {
        "status": "failed" if problems else "passed",
        "problems": problems,
        "files": [p.name for p in clip.paths],
        "frames": len(clip.frames),
        "frames_compared": len(expected.get("frame_sha256", [])) - len(mismatched),
        "info": info,
        "header": first_header,
    }


def expected_metadata(info: ClipInfo) -> dict:
    """What `verify` checks the metadata blocks and file header against, from the `ClipInfo` written."""
    keys = ("width", "height", "bits", "black", "white", "cfa_pattern", "color_matrix", "frame_rate", "iso",
            "shutter_us", "kelvin")  # fmt: skip
    return {**{key: getattr(info, key) for key in keys}, "wb_gains": list(info.wb_gains)}


def samples_sha256(samples: np.ndarray) -> str:
    """The digest `verify` compares: the uint16 samples in native order, row-major."""
    return hashlib.sha256(np.ascontiguousarray(samples, dtype=np.uint16).tobytes()).hexdigest()
