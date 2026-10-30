"""Real LPIPS against a supplied aligned reference, with bounded patch tiling."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile

from prd_v5.comparison import validate_region

MAX_PATCHES = 4096


def load_lpips(net: str = "alex", device: str = "cpu"):
    """Return a distance callable backed by the official `lpips` package (v0.1 heads).

    The package ships the linear heads; the torchvision ImageNet backbone is
    fetched by torchvision on first use (Colab only). Inputs are NCHW float32
    arrays in [-1, 1]; output is one float distance per sample.
    """
    if net not in ("alex", "vgg", "squeeze"):
        raise ValueError("net must be alex, vgg or squeeze")
    import lpips
    import torch

    model = lpips.LPIPS(net=net, version="0.1").to(device).eval()

    def distance(a, b):
        with torch.no_grad():
            out = model(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device))
        return out.reshape(-1).cpu().numpy().astype(np.float64)

    return distance


def lpips_distance(
    candidate: str | Path,
    reference: str | Path | None,
    *,
    calibration: dict | None = None,
    region: tuple[int, int, int, int] | None = None,
    patch: int = 256,
    distance=None,
    net: str = "alex",
    device: str = "cuda",
) -> dict:
    """Mean LPIPS over non-overlapping patches of aligned uint16 sRGB TIFFs.

    Same caller-attested alignment contract as calibrated_delta_e. Patches
    smaller than the network minimum (edge remainders) are skipped and counted.
    Perceptual distance to a reference is not recovered-detail or RAW truth.
    `distance` may inject a callable (NCHW float32 [-1,1] pairs -> distances);
    otherwise the official lpips package is loaded lazily.
    """
    if type(patch) is not int or not 64 <= patch <= 1024:
        raise ValueError("patch must be an integer from 64 to 1024")
    validate_region(region)
    report = {
        "status": "not measured",
        "metric": f"LPIPS v0.1 {net} (richzhang/PerceptualSimilarity, BSD-2-Clause)",
        "domain": "encoded sRGB scaled to [-1, 1]; lower is perceptually closer",
        "limitations": "Caller-attested alignment; no recovered RAW or HDR truth claim",
    }
    if reference is None:
        return {**report, "reason": "No aligned reference supplied"}
    if calibration is None:
        return {**report, "reason": "No alignment contract supplied"}
    required = {"camera_profile", "trusted_development", "alignment", "colour_space"}
    if not isinstance(calibration, dict) or set(calibration) != required:
        raise ValueError(f"calibration requires exactly {sorted(required)}")
    if any(not isinstance(value, str) or not value.strip() for value in calibration.values()):
        raise ValueError("calibration evidence must contain nonempty strings")
    if calibration["colour_space"] != "sRGB_D65_2deg":
        raise ValueError("Only explicitly calibrated sRGB_D65_2deg is supported")
    report["calibration"] = dict(calibration)
    with ExitStack() as stack:
        arrays = []
        for path in (candidate, reference):
            array = tifffile.memmap(path, mode="r")
            stack.callback(array._mmap.close)
            if array.dtype != np.uint16 or array.ndim != 3 or array.shape[2] != 3 or not all(array.shape):
                raise ValueError("LPIPS requires nonempty uint16 RGB TIFFs")
            arrays.append(array)
        if arrays[0].shape != arrays[1].shape:
            return {**report, "reason": "Unequal rasters; no implicit registration or resampling"}
        top, left, bottom, right = region or (0, 0, *arrays[0].shape[:2])
        if bottom > arrays[0].shape[0] or right > arrays[0].shape[1]:
            raise ValueError("region exceeds reference raster")
        rows, cols = (bottom - top) // patch, (right - left) // patch
        if rows * cols == 0:
            return {**report, "reason": f"Region smaller than one {patch}px patch"}
        if rows * cols > MAX_PATCHES:
            raise ValueError(f"region needs {rows * cols} patches; limit is {MAX_PATCHES}, select a smaller region")
        if distance is None:
            distance = load_lpips(net, device)
        values = []
        for r in range(rows):
            for c in range(cols):
                y, x = top + r * patch, left + c * patch
                pair = [
                    (array[y : y + patch, x : x + patch].astype(np.float32) / 65535 * 2 - 1).transpose(2, 0, 1)[None]
                    for array in arrays
                ]
                value = float(np.asarray(distance(pair[0], pair[1])).reshape(-1)[0])
                if not np.isfinite(value):
                    raise ValueError("LPIPS distance was not finite")
                values.append(value)
        skipped = (bottom - top) * (right - left) - rows * cols * patch * patch
        return {
            **report,
            "status": "measured reference perceptual distance",
            "region": [top, left, bottom, right],
            "patch": patch,
            "patches": len(values),
            "uncovered_pixels": skipped,
            "mean_lpips": float(np.mean(values)),
            "max_lpips": float(np.max(values)),
        }
