"""Driver step 5: the frozen pipeline on the captain's own 2K and 4K sources (section 8.2).

There is no reference RAW for these, so their outputs carry no accuracy claim.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from prd_v6 import benchmark, colour, constants, pipeline, preflight, sr

SOURCES = (("4k_source", "SOURCE_4K_URL"), ("2k_source", "SOURCE_2K_URL"))
SIZES = {"2k": (1280, 1920), "4k": (2560, 3840), "50mp": constants.VISIBLE_HW}


def load_source(path: Path) -> tuple[np.ndarray, dict]:
    """EXIF orientation, ICC to sRGB, 8-bit RGB, centre crop to 3:2 (2:3 for portrait)."""
    from PIL import Image, ImageCms, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        info = {"format": opened.format, "mode": opened.mode, "size": list(opened.size)}
        icc = image.info.get("icc_profile")
        image = image.convert("RGB")
        if icc:
            source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            image = ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
            info["icc"] = "converted to sRGB"
        rgb = np.asarray(image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    portrait = height > width
    num, den = (2, 3) if portrait else (3, 2)
    if width * den > height * num:
        crop_w, crop_h = height * num // den, height
    else:
        crop_w, crop_h = width, width * den // num
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    info.update(portrait=portrait, crop=[top, left, crop_h, crop_w])
    return np.ascontiguousarray(rgb[top : top + crop_h, left : left + crop_w]), info


def rggb_layout(hw: tuple[int, int]) -> np.ndarray:
    return np.tile(np.array([[0, 1], [1, 2]], dtype=np.int8), (hw[0] // 2, hw[1] // 2))


def run(run_dir: Path, settings: dict, model: sr.Model, *, preset: pipeline.Preset = pipeline.MAX, log=print) -> dict:
    run_dir = Path(run_dir)
    verdict = json.loads((run_dir / "verdict.json").read_text()) if (run_dir / "verdict.json").exists() else {}
    prior = colour.prior_neutral(pipeline.camera_matrix())
    record = {
        "claim": (
            f"The pipeline measured {verdict.get('accuracy_percent')}% on the benchmark ({verdict.get('status')}). "
            "That number does not transfer automatically to other scenes; these outputs carry no accuracy claim."
        ),
        "sources": {},
    }
    for key, setting in SOURCES:
        url = settings.get(setting)
        if not url:
            continue
        folder = run_dir / "production" / key
        fetched = preflight.download(
            url,
            run_dir / "inputs" / f"{key}{Path(url.split('?')[0]).suffix or '.img'}",
            size=None,
            digest=None,
            log=log,
        )
        source, info = load_source(Path(fetched["path"]))
        log(f"production {key}: {info}")
        upscaled = None
        outputs = {}
        for name in constants.OUTPUT_ORDER:
            hw = SIZES[name] if not info["portrait"] else SIZES[name][::-1]
            needs_sr = hw[0] > source.shape[0] or hw[1] > source.shape[1]
            if needs_sr and upscaled is None:
                upscaled = pipeline.run_sr(source, model, preset, log)
            linear = pipeline.linear_output(source, hw, model=model, preset=preset, upscaled=upscaled)
            if name == "50mp":
                _, samples = pipeline.cfa_samples(linear, prior, rggb_layout(hw), noise_seed=preset.noise_seed)
                files = benchmark._write_output(folder / name, name, samples, prior, cfa="RGGB")
            else:
                samples = pipeline.linear_raw_samples(linear, prior)
                files = benchmark._write_output(folder / name, name, samples, prior, cfa=None)
            outputs[name] = {
                "size": list(hw),
                "super_resolution": needs_sr,
                "files": {k: str(Path(v).relative_to(run_dir)) for k, v in files.items()},
            }
            del linear, samples
        record["sources"][key] = {"download": fetched, "decode": info, "outputs": outputs}
        (folder / "production.json").write_text(json.dumps(record["sources"][key], indent=2, default=str))
    (run_dir / "production").mkdir(exist_ok=True)
    (run_dir / "production" / "production.json").write_text(json.dumps(record, indent=2, default=str))
    return record
