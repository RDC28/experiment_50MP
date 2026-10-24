"""Pretrained super-resolution weights, loading and devices (PRD sections 10.1, 21.4).

Inference only: no training, fine-tuning or distillation. ``docs/models.md``
records the model evaluation, exact versions, weight sources, checksums,
licences and commercial-use terms behind this registry.

Weights are downloaded at run time into an ignored cache folder (by default
``<project>/weights``), never committed, and their SHA-256 is checked after
every download and before every load. ``torch.load(weights_only=True)`` reads
them as plain tensors, so a tampered file cannot run code; spandrel then builds
the matching architecture from the state dict.

PyTorch and spandrel are imported lazily (the ``sr`` extra in ``pyproject.toml``),
so Phases 0 and 1 run without them.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

DEVICES = ("auto", "cuda", "mps", "cpu")
PRECISIONS = ("auto", "float32", "float16")
WEIGHTS_SUBDIR = "weights"
MIN_TILE = 32
# Below about 256 pixels the perceptual network sees too little context and neighbouring tiles disagree
# visibly on dense detail (measured with SwinIR in docs/models.md); out-of-memory splitting may still go below.
MIN_AUTO_TILE = 256
MAX_AUTO_TILE = 1024

_SWINIR_LICENSE = "Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)"
_REALESRGAN_LICENSE = "BSD-3-Clause (Real-ESRGAN code and released weights, Copyright (c) 2021 Xintao Wang)"
_DIV2K_TERMS = "DIV2K is published for academic research only"


@dataclass(frozen=True)
class WeightFile:
    name: str
    url: str
    sha256: str
    size_bytes: int
    architecture: str
    role: str  # "fidelity" (pixel loss) or "perceptual" (adversarial loss)
    license: str
    training_data: str
    commercial_use: str
    description: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[1]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "file": self.filename,
            "url": self.url,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "architecture": self.architecture,
            "role": self.role,
            "license": self.license,
            "training_data": self.training_data,
            "commercial_use": self.commercial_use,
            "description": self.description,
        }


WEIGHTS = {
    w.name: w
    for w in (
        WeightFile(
            "swinir_m_x4_classical",
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
            "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215",
            67_869_037,
            "SwinIR-M x4, pixel-shuffle upsampler (11.9M parameters)",
            "fidelity",
            _SWINIR_LICENSE,
            "DF2K (DIV2K + Flickr2K), bicubic degradation, L1 loss",
            f"Licence permits commercial use; {_DIV2K_TERMS}, so commercial use of these weights is unresolved.",
            "Classical SR: reconstructs edges faithfully, keeps source texture, invents the least.",
        ),
        WeightFile(
            "swinir_m_x4_realsr_gan",
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
            "b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6",
            67_129_861,
            "SwinIR-M x4, nearest+conv upsampler (11.7M parameters)",
            "perceptual",
            _SWINIR_LICENSE,
            "DIV2K + Flickr2K + OST, BSRGAN real-world degradation, L1 + perceptual + adversarial loss",
            f"Licence permits commercial use; {_DIV2K_TERMS}, so commercial use of these weights is unresolved.",
            "Real-world SR: sharper edges and synthesised texture.",
        ),
        WeightFile(
            "realesrnet_x4plus",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
            "a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df",
            67_040_989,
            "RRDBNet x4 (ESRGAN generator, 16.7M parameters)",
            "fidelity",
            _REALESRGAN_LICENSE,
            "DF2K (DIV2K + Flickr2K) + OST, Real-ESRGAN degradation, L1 loss",
            f"Licence permits commercial use; {_DIV2K_TERMS}, so commercial use of these weights is unresolved.",
            "Real-world SR without adversarial loss: smooth, removes fine texture.",
        ),
        WeightFile(
            "realesrgan_x4plus",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            67_040_989,
            "RRDBNet x4 (ESRGAN generator, 16.7M parameters)",
            "perceptual",
            _REALESRGAN_LICENSE,
            "DF2K (DIV2K + Flickr2K) + OST, Real-ESRGAN degradation, L1 + perceptual + adversarial loss",
            f"Licence permits commercial use; {_DIV2K_TERMS}, so commercial use of these weights is unresolved.",
            "Real-world SR: strong sharpening and painted texture.",
        ),
    )
}

# Which weights each backend uses for the fidelity (Mode A) and perceptual (Modes B, C) networks.
BACKENDS = {
    "swinir": {"fidelity": "swinir_m_x4_classical", "perceptual": "swinir_m_x4_realsr_gan"},
    "realesrgan": {"fidelity": "realesrnet_x4plus", "perceptual": "realesrgan_x4plus"},
}
# Peak device memory per input pixel at float32, measured on Apple MPS with tiles of 192 to 320 pixels.
BYTES_PER_INPUT_PIXEL = {"SwinIR": 33_000, "ESRGAN": 17_000}


class DeviceOutOfMemory(RuntimeError):
    """The inference device ran out of memory for one tile; the tiler splits the tile or falls back."""


class WeightsError(RuntimeError):
    """Weights are missing, cannot be downloaded, or fail their checksum."""


# --------------------------------------------------------------------------- weights


def default_weights_dir(project_root: str | Path) -> Path:
    """``<project>/weights``, ignored by Git."""
    return Path(project_root) / WEIGHTS_SUBDIR


def weights_path(name: str, weights_dir: str | Path) -> Path:
    return Path(weights_dir) / WEIGHTS[name].filename


def cached(name: str, weights_dir: str | Path) -> bool:
    """True when the weights are in the cache with the recorded checksum."""
    path = weights_path(name, weights_dir)
    return path.is_file() and sha256_file(path) == WEIGHTS[name].sha256


def ensure_weights(
    name: str, weights_dir: str | Path, *, download: bool = True, log: Callable[[str], None] = print
) -> Path:
    """Return the verified local weights file, downloading it when missing or corrupt."""
    spec = WEIGHTS[name]
    path = weights_path(name, weights_dir)
    if path.is_file():
        if sha256_file(path) == spec.sha256:
            return path
        log(f"{path.name}: checksum mismatch in the cache, downloading again")
        path.unlink()
    if not download:
        raise WeightsError(f"{spec.filename} is not in {weights_dir} and downloading is disabled")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    log(f"downloading {spec.url} ({spec.size_bytes / 1e6:.0f} MB)")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise WeightsError(f"could not download {spec.url}: {error}") from error
    if digest.hexdigest() != spec.sha256:
        partial.unlink(missing_ok=True)
        raise WeightsError(f"{spec.filename}: SHA-256 {digest.hexdigest()} does not match the recorded {spec.sha256}")
    os.replace(partial, path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- devices


def available() -> bool:
    """True when PyTorch and spandrel are installed (``pip install .[sr]``)."""
    try:
        import spandrel  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(requested: str = "auto") -> str:
    """auto prefers CUDA (Colab), then Apple MPS, then the CPU."""
    import torch

    if requested not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("sr_device is cuda but no CUDA device is available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("sr_device is mps but the MPS backend is not available")
    return requested


def resolve_precision(requested: str, device: str, supports_half: bool) -> str:
    """auto: float16 on CUDA when the architecture supports it (half the activation memory), else float32."""
    if requested not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {requested!r}")
    if requested == "auto":
        return "float16" if device == "cuda" and supports_half else "float32"
    if requested == "float16" and (device == "cpu" or not supports_half):
        raise ValueError(f"float16 inference is not supported here (device {device}, supports_half={supports_half})")
    return requested


def device_info(device: str) -> dict:
    import torch

    info = {"device": device, "torch": torch.__version__}
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        info.update(
            name=torch.cuda.get_device_name(0),
            memory_total_gb=round(total / 1e9, 2),
            memory_free_gb=round(free / 1e9, 2),
            cuda=torch.version.cuda,
        )
    elif device == "mps":
        info.update(
            name="Apple MPS (unified memory)",
            recommended_max_memory_gb=round(torch.mps.recommended_max_memory() / 1e9, 2),
        )
    else:
        info.update(name="CPU", threads=torch.get_num_threads())
    return info


def auto_tile_size(device: str, precision: str, architecture: str, *, source_width: int, scale: int) -> tuple[int, str]:
    """Largest tile (input pixels per side, multiple of 32) whose activations fit the memory budget.

    Device budget: 60% of free CUDA memory; on MPS (unified memory) and the CPU, 30% of available RAM.
    The host also holds one strip of tiles at model scale in float32, kept below 15% of available RAM.
    The result is never below MIN_AUTO_TILE; a tile that then does not fit is split on out-of-memory.
    """
    import torch

    per_pixel = BYTES_PER_INPUT_PIXEL.get(architecture, 33_000) * (0.5 if precision == "float16" else 1.0)
    ram = psutil.virtual_memory().available
    if device == "cuda":
        budget, basis = 0.6 * torch.cuda.mem_get_info()[0], "60% of free CUDA memory"
    else:
        budget, basis = 0.3 * ram, "30% of available RAM"
    tile = int((budget / per_pixel) ** 0.5)
    strip_rows = int(0.15 * ram / (source_width * scale * scale * 3 * 4))
    tile = max(MIN_AUTO_TILE, min(MAX_AUTO_TILE, tile, strip_rows) // 32 * 32)
    return tile, f"{basis} at {per_pixel / 1000:.0f} kB per input pixel ({architecture}, {precision})"


# --------------------------------------------------------------------------- inference


class TorchModel:
    """A loaded network as a numpy callable on sRGB-encoded (H, W, 3) float32 tiles in [0, 1]."""

    def __init__(self, descriptor, weights: WeightFile, device: str, precision: str):
        import torch

        self._torch = torch
        self.descriptor = descriptor
        self.weights = weights
        self.device = device
        self.precision = precision
        self.scale = int(descriptor.scale)
        self.architecture = descriptor.architecture.name
        self.minimum_side = int(descriptor.size_requirements.minimum)
        # Window-attention networks (SwinIR: 8) need tiles on their window grid; convolutional ones do not.
        self.alignment = int(getattr(descriptor.model, "window_size", 1) or 1)
        self.dtype = torch.float16 if precision == "float16" else torch.float32
        descriptor.model.eval().to(device=device, dtype=self.dtype)
        for parameter in descriptor.model.parameters():
            parameter.requires_grad_(False)
        self.peak_device_bytes = 0

    def __call__(self, encoded: np.ndarray) -> np.ndarray:
        torch = self._torch
        height, width = encoded.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(encoded.transpose(2, 0, 1)))[None]
        pad_h, pad_w = max(0, self.minimum_side - height), max(0, self.minimum_side - width)
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        try:
            with torch.inference_mode():
                y = self.descriptor.model(x.to(device=self.device, dtype=self.dtype))
                y = y[0, :, : height * self.scale, : width * self.scale]
                out = y.float().clamp_(0, 1).permute(1, 2, 0).cpu().numpy()
            self._record_peak()
        except (RuntimeError, MemoryError) as error:
            if not _is_out_of_memory(error):
                raise
            # The tiler empties the device cache once this exception (and the tensors it holds) is released.
            raise DeviceOutOfMemory(f"{self.device}: {str(error).splitlines()[0]}") from error
        return out

    def fallback(self) -> TorchModel | None:
        """The same network on the CPU in float32, for tiles the device cannot fit at the minimum size."""
        if self.device == "cpu":
            return None
        self.descriptor.model.to(device="cpu", dtype=self._torch.float32)
        self.release_cache()
        return TorchModel(self.descriptor, self.weights, "cpu", "float32")

    def release_cache(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
        elif self.device == "mps":
            self._torch.mps.empty_cache()

    def describe(self) -> dict:
        return {
            "weights": self.weights.to_dict(),
            "architecture": self.architecture,
            "scale": self.scale,
            "tile_alignment": self.alignment,
            "device": device_info(self.device),
            "precision": self.precision,
            "peak_device_memory_mb": round(self.peak_device_bytes / 1e6, 1) if self.peak_device_bytes else None,
        }

    def _record_peak(self) -> None:
        if self.device == "cuda":
            peak = self._torch.cuda.max_memory_allocated()
        elif self.device == "mps":
            peak = self._torch.mps.driver_allocated_memory()
        else:
            return
        self.peak_device_bytes = max(self.peak_device_bytes, int(peak))


def load_model(
    name: str,
    weights_dir: str | Path,
    *,
    device: str = "auto",
    precision: str = "auto",
    download: bool = True,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> TorchModel:
    """Verify (and if needed download) the named weights and load them for inference."""
    import spandrel
    import torch

    spec = WEIGHTS[name]
    torch.manual_seed(seed)
    device = resolve_device(device)
    state = torch.load(
        ensure_weights(name, weights_dir, download=download, log=log), map_location="cpu", weights_only=True
    )
    descriptor = spandrel.ModelLoader().load_from_state_dict(state)
    del state
    if not isinstance(descriptor, spandrel.ImageModelDescriptor):
        raise WeightsError(f"{spec.filename} is not an image-to-image model")
    precision = resolve_precision(precision, device, bool(descriptor.supports_half))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return TorchModel(descriptor, spec, device, precision)


def _is_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError":
        return True
    message = str(error).lower()
    return "out of memory" in message or "can't allocate memory" in message
