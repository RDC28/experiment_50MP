"""Test helpers: hand-built ICC profiles."""

from __future__ import annotations

import struct

import numpy as np

from synthetic_raw_lab import color_management as cm


def build_icc_profile(
    colorants: np.ndarray | None = None,
    curve: tuple = ("gamma", 2.2),
    *,
    description: str = "Test profile",
    color_space: bytes = b"RGB ",
) -> bytes:
    """A minimal ICC v2 matrix/TRC (or grey TRC) display profile for tests.

    ``colorants`` holds the D50-adapted XYZ of the R, G, B primaries as columns.
    ``curve`` is ("gamma", g), ("para", function_type, params) or ("table", values in [0, 1]).
    """

    def s15(value: float) -> bytes:
        return struct.pack(">i", round(value * 65536))

    def xyz_tag(xyz) -> bytes:
        return b"XYZ " + b"\0" * 4 + b"".join(s15(v) for v in xyz)

    kind = curve[0]
    if kind == "gamma":
        trc = b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", round(curve[1] * 256)) + b"\0\0"
    elif kind == "para":
        trc = b"para" + b"\0" * 4 + struct.pack(">H", curve[1]) + b"\0\0" + b"".join(s15(p) for p in curve[2])
    else:
        values = np.round(np.asarray(curve[1]) * 65535).astype(">u2")
        trc = b"curv" + b"\0" * 4 + struct.pack(">I", len(values)) + values.tobytes()
    text = description.encode("latin-1") + b"\0"
    desc = b"desc" + b"\0" * 4 + struct.pack(">I", len(text)) + text + b"\0" * 12 + b"\0" * 67

    tags = [(b"desc", desc), (b"wtpt", xyz_tag(cm.ICC_D50_XYZ))]
    if color_space == b"GRAY":
        tags.append((b"kTRC", trc))
    else:
        colorants = d50_colorants(cm.SRGB_PRIMARIES) if colorants is None else colorants
        tags += [
            (b"rXYZ", xyz_tag(colorants[:, 0])),
            (b"gXYZ", xyz_tag(colorants[:, 1])),
            (b"bXYZ", xyz_tag(colorants[:, 2])),
        ]
        tags += [(b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc)]

    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    table, data = struct.pack(">I", len(tags)), b""
    for signature, payload in tags:
        padded = payload + b"\0" * (-len(payload) % 4)
        table += signature + struct.pack(">II", offset + len(data), len(payload))
        data += padded
    size = 128 + table_size + len(data)
    header = bytearray(128)
    header[0:4] = struct.pack(">I", size)
    header[8:12] = bytes([2, 0x10, 0, 0])
    header[12:16] = b"mntr"
    header[16:20] = color_space
    header[20:24] = b"XYZ "
    header[36:40] = b"acsp"
    header[68:80] = b"".join(s15(v) for v in cm.ICC_D50_XYZ)
    return bytes(header) + table + data


def d50_colorants(primaries) -> np.ndarray:
    """D50-adapted (Bradford) XYZ colorants for primaries with a D65 white, as an ICC profile stores them."""
    adapt = cm.bradford_adaptation(cm.xy_to_xyz(cm.D65_XY), cm.ICC_D50_XYZ)
    return adapt @ cm.rgb_to_xyz_matrix(primaries, cm.D65_XY)


def evaluation_config(project, root, **sections):
    """A Phase 6 evaluation profile writing under ``project`` with the repository's specialised profiles."""
    from synthetic_raw_lab import evaluation_config as ec

    universal = {
        "output_root": str(project / "outputs"),
        "reports_root": str(project / "reports"),
        "inputs_root": str(project / "inputs"),
        "analysis": {"face_detection": False},
        "human": {"profile": str(root / "configs" / "human_default.yaml"), "overrides": {}},
        "night": {"profile": str(root / "configs" / "night_default.yaml"), "overrides": {}},
    }
    for key, value in sections.pop("universal_overrides", {}).items():
        universal[key] = {**universal[key], **value} if isinstance(universal.get(key), dict) else value
    sections.setdefault("ablation", {})
    sections["ablation"] = {"performance_sweep": [], "review_crop_px": 48, **sections["ablation"]}
    return ec.load_evaluation_config(
        root / "configs" / "evaluation_default.yaml", root, universal_overrides=universal, **sections
    )


def fake_sr(mode, linear, width, height):
    """A stand-in for the Phase 2 SR path: Lanczos plus an unsharp mask (stronger for the perceptual modes)."""
    import cv2

    from synthetic_raw_lab import baselines

    image = baselines.resample(linear, width, height, "lanczos")
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    image = (image + (0.5 if mode == "conservative" else 1.5) * (image - blur)).astype(np.float32)
    summary = {
        "seams": {"passed": mode == "conservative"},
        "tiling": {"inference_seconds": 0.01, "tile_size": 64, "tile_overlap": 16},
        "model": {"peak_device_memory_mb": 1.0, "precision": "float32"},
    }
    return image, summary
