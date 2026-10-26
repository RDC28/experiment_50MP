"""Runtime inspection and tile budgets from PRD_v3 compute budget section."""

from __future__ import annotations

import importlib
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import psutil

_GIB = 1024**3


@dataclass(frozen=True)
class TileParameters:
    """Engineering defaults; backends must benchmark tiles before estimating ETA."""

    tile_size: int
    overlap: int
    aggressive_offload: bool
    heavy_reconstruction_supported: bool


@dataclass(frozen=True)
class EnvironmentReport:
    """Detected resources in bytes, with unavailable accelerator fields left null."""

    gpu_model: str | None
    gpu_vram_bytes: int | None
    cuda_version: str | None
    pytorch_version: str | None
    system_ram_bytes: int
    free_disk_bytes: int
    cpu_cores: int
    tiles: TileParameters

    def to_dict(self) -> dict:
        return asdict(self)


class EnvironmentInspector:
    """Inspect the actual runtime without allocating tensors or requiring torch."""

    @staticmethod
    def derive_tile_parameters(vram_gib: float | None) -> TileParameters:
        """Choose conservative defaults within the PRD VRAM bands."""
        if vram_gib is not None and (not math.isfinite(vram_gib) or vram_gib < 0):
            raise ValueError("VRAM must be a finite, nonnegative GiB value or None")
        if not vram_gib:
            return TileParameters(192, 32, True, False)
        if vram_gib >= 40:
            return TileParameters(768, 96, False, True)
        if vram_gib >= 20:
            return TileParameters(512, 64, False, True)
        if vram_gib >= 12:
            return TileParameters(256, 64, False, True)
        return TileParameters(192, 32, True, True)

    def inspect(self, disk_path: str | Path = ".") -> EnvironmentReport:
        """Report the current CUDA device and resources on the output filesystem."""
        gpu_model = gpu_vram_bytes = cuda_version = pytorch_version = None
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            torch = None
        if torch is not None:
            pytorch_version = str(torch.__version__)
            cuda_version = torch.version.cuda
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                gpu_model = properties.name
                gpu_vram_bytes = int(properties.total_memory)
        return EnvironmentReport(
            gpu_model=gpu_model,
            gpu_vram_bytes=gpu_vram_bytes,
            cuda_version=cuda_version,
            pytorch_version=pytorch_version,
            system_ram_bytes=int(psutil.virtual_memory().total),
            free_disk_bytes=shutil.disk_usage(disk_path).free,
            cpu_cores=os.cpu_count() or 1,
            tiles=self.derive_tile_parameters(None if gpu_vram_bytes is None else gpu_vram_bytes / _GIB),
        )

    @staticmethod
    def print_report(report: EnvironmentReport, file: TextIO | None = None) -> None:
        """Print the seven startup diagnostics required by the notebook."""
        lines = [
            f"GPU model: {report.gpu_model or 'unavailable (CPU-only)'}",
            f"GPU VRAM: {report.gpu_vram_bytes / _GIB:.2f} GiB"
            if report.gpu_vram_bytes is not None
            else "GPU VRAM: unavailable",
            f"CUDA version: {report.cuda_version or 'unavailable'}",
            f"PyTorch version: {report.pytorch_version or 'not installed'}",
            f"System RAM: {report.system_ram_bytes / _GIB:.2f} GiB",
            f"Free disk: {report.free_disk_bytes / _GIB:.2f} GiB",
            f"CPU cores: {report.cpu_cores}",
            f"Tile size: {report.tiles.tile_size} px; overlap: {report.tiles.overlap} px",
        ]
        if not report.tiles.heavy_reconstruction_supported:
            lines.append("CPU-only: sensor/metadata stages only practical; heavy SR/diffusion disabled.")
        print("\n".join(lines), file=file)
