"""MLV writer and reader on tiny synthetic frames: layout per mlv.h/raw.h, packing, chunks, the check."""

from __future__ import annotations

import struct
from fractions import Fraction

import numpy as np
import pytest

from video_v1 import mlv

HW = (6, 16)
MATRIX = [[0.7, -0.2, -0.05], [-0.4, 1.2, 0.2], [-0.05, 0.1, 0.6]]


def _info(bits: int = 14, **overrides) -> mlv.ClipInfo:
    values = dict(width=HW[1], height=HW[0], bits=bits, black=512 >> (14 - bits), white=15360 >> (14 - bits),
                  cfa_pattern="RGGB", color_matrix=MATRIX, frame_rate=Fraction(30000, 1001))  # fmt: skip
    values.update(overrides)
    return mlv.ClipInfo(**values)


def _frames(count: int, bits: int, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 1 << bits, size=HW, dtype=np.uint16) for _ in range(count)]


def _write(tmp_path, frames, bits=14, chunk_bytes=0):
    with mlv.Writer(tmp_path, "clip", _info(bits), chunk_bytes=chunk_bytes) as writer:
        for frame in frames:
            writer.write(frame)
    return writer


def _expected(frames, bits=14) -> dict:
    return {
        **mlv.expected_metadata(_info(bits)),
        "frames": len(frames),
        "frame_sha256": [mlv.samples_sha256(f) for f in frames],
    }


@pytest.mark.parametrize("bits", mlv.SUPPORTED_BITS)
def test_packing_round_trips(bits):
    (frame,) = _frames(1, bits)
    packed = mlv.pack(frame, bits)
    assert len(packed) == frame.size * bits // 8
    np.testing.assert_array_equal(mlv.unpack(packed, HW, bits), frame)


def test_packing_is_the_camera_bit_order():
    """raw.h: samples form one big-endian bit stream, stored as little-endian 16-bit words."""
    samples = np.array([[0x3FFF, 0, 0x2AAA, 0x1555, 1, 2, 3, 4]], dtype=np.uint16)
    stream = "".join(f"{v:014b}" for v in samples.reshape(-1))
    words = [int(stream[i : i + 16], 2) for i in range(0, len(stream), 16)]
    assert mlv.pack(samples, 14) == struct.pack(f"<{len(words)}H", *words)


def test_reduce_bits_rounds_and_clips():
    samples16 = np.array([[0, 1, 2, 2048, 61440, 65535]], dtype=np.uint16)
    assert mlv.reduce_bits(samples16, 14).tolist() == [[0, 0, 1, 512, 15360, 16383]]


def test_clip_layout_follows_mlv_h(tmp_path):
    frames = _frames(3, 14)
    writer = _write(tmp_path, frames)
    data = writer.paths[0].read_bytes()
    magic, size, version, _guid, num, count, _flags, video_class, audio_class, vframes, _af, nom, den = (
        mlv.FILE_HEADER.unpack_from(data)
    )
    assert (magic, size, version.rstrip(b"\0"), num, count) == (b"MLVI", 52, b"v2.0", 0, 1)
    assert (video_class, audio_class, vframes, nom, den) == (1, 0, 3, 30000, 1001)
    blocks, offset = [], 52
    while offset < len(data):
        kind, length, timestamp = mlv.BLOCK_HEADER.unpack_from(data, offset)
        blocks.append((kind.decode(), length, timestamp))
        offset += length
    assert offset == len(data)
    assert [b[0] for b in blocks] == ["RAWI", "RAWC", "IDNT", "EXPO", "WBAL", "VIDF", "VIDF", "VIDF"]
    assert dict((k, n) for k, n, _ in blocks[:5]) == {"RAWI": 180, "RAWC": 32, "IDNT": 84, "EXPO": 40, "WBAL": 44}
    assert [t for k, _, t in blocks if k == "VIDF"] == [0, 33367, 66733]
    raw = mlv.RAW_INFO.unpack_from(data, 52 + 16 + 4)
    assert raw[2:9] == (HW[0], HW[1], HW[1] * 14 // 8, HW[0] * HW[1] * 14 // 8, 14, 512, 15360)
    assert raw[19:21] == (0x02010100, 21)  # RGGB, D65
    assert raw[21:23] == (7000, 10000)


def test_reader_decodes_every_frame_bit_exact(tmp_path):
    frames = _frames(3, 14)
    writer = _write(tmp_path, frames)
    clip = mlv.read(writer.paths[0])
    assert clip.info["cfa_pattern"] == "RGGB" and clip.info["bits"] == 14
    np.testing.assert_allclose(clip.info["color_matrix"], MATRIX)
    for number, frame in enumerate(frames):
        np.testing.assert_array_equal(clip.frame(number), frame)
    result = mlv.verify(writer.paths[0], _expected(frames))
    assert result["status"] == "passed", result["problems"]
    assert result["frames_compared"] == 3 and result["files"] == ["clip.MLV"]


def test_a_spanned_clip_reads_as_one(tmp_path):
    frames = _frames(7, 14)
    # 432 bytes of headers and metadata, then 200-byte frames: chunks of 2, 4 and 1 frames under 900 bytes
    writer = _write(tmp_path, frames, chunk_bytes=900)
    assert [p.name for p in writer.paths] == ["clip.MLV", "clip.M00", "clip.M01"]
    headers = mlv.read(writer.paths[0]).headers
    assert [h["file_num"] for h in headers] == [0, 1, 2] and {h["file_count"] for h in headers} == {3}
    assert len({h["guid"] for h in headers}) == 1
    assert [h["video_frames"] for h in headers] == [7, 4, 1]  # the clip total, then each chunk's own
    result = mlv.verify(writer.paths[0], _expected(frames))
    assert result["status"] == "passed", result["problems"]


def test_the_check_catches_a_changed_sample(tmp_path):
    frames = _frames(2, 14)
    writer = _write(tmp_path, frames)
    data = bytearray(writer.paths[0].read_bytes())
    data[-1] ^= 0x01
    writer.paths[0].write_bytes(bytes(data))
    result = mlv.verify(writer.paths[0], _expected(frames))
    assert result["status"] == "failed" and "1 decoded frames differ" in result["problems"][0]


def test_the_check_catches_a_wrong_header(tmp_path):
    frames = _frames(2, 14)
    writer = _write(tmp_path, frames)
    result = mlv.verify(writer.paths[0], {**_expected(frames), "frames": 3, "white": 16383})
    assert result["status"] == "failed"
    assert any("white" in p for p in result["problems"]) and any("expected 3" in p for p in result["problems"])


def test_the_check_reads_back_the_exposure_and_white_balance(tmp_path):
    frames = _frames(2, 14)
    with mlv.Writer(tmp_path, "clip", _info(shutter_us=16683, kelvin=6504, wb_gains=(2100, 1024, 1500))) as writer:
        for frame in frames:
            writer.write(frame)
    expected = {**_expected(frames), "shutter_us": 16683, "kelvin": 6504, "wb_gains": [2100, 1024, 1500]}
    assert mlv.verify(writer.paths[0], expected)["status"] == "passed"
    result = mlv.verify(writer.paths[0], {**expected, "iso": 200, "kelvin": 5000, "wb_gains": [1, 2, 3]})
    assert result["status"] == "failed" and [p.split()[:2] for p in result["problems"]] == [
        ["EXPO", "iso"],
        ["WBAL", "kelvin"],
        ["WBAL", "wb_gains"],
    ]


def test_the_check_catches_a_wrong_vidf_timestamp(tmp_path):
    frames = _frames(3, 14)
    writer = _write(tmp_path, frames)
    data = bytearray(writer.paths[0].read_bytes())
    clip = mlv.read(writer.paths[0])
    header = clip.frames[1][2] - mlv.VIDF.size - mlv.BLOCK_HEADER.size  # frame 1's block header
    struct.pack_into("<Q", data, header + 8, writer.timestamp(1) + 1000)
    writer.paths[0].write_bytes(bytes(data))
    result = mlv.verify(writer.paths[0], _expected(frames))
    assert result["status"] == "failed"
    assert result["problems"] == ["1 VIDF timestamps do not match the frame rate (first [1])"]


def test_writer_rejects_what_mlv_app_cannot_read(tmp_path):
    with pytest.raises(ValueError):
        mlv.Writer(tmp_path, "a", _info(16))
    with pytest.raises(ValueError):
        mlv.Writer(tmp_path, "b", _info(width=12))
    with mlv.Writer(tmp_path, "c", _info()) as writer, pytest.raises(ValueError):
        writer.write(np.full(HW, 1 << 14, np.uint16))
    assert mlv.mlv_width(1918) == 1912 and mlv.chunk_name("x", 3) == "x.M02"
