"""Runtime inspection, memory benchmarking and storage estimates (PRD sections 21.2 to 21.5, 28.1).

GPU details come from ``nvidia-smi`` rather than importing PyTorch, so the
inspection itself does not add hundreds of megabytes to the process and skew
the memory benchmark.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

import numpy as np
import psutil

MB = 1_000_000  # PRD 21.3 uses decimal megabytes

# (label, bytes per pixel, channels, dtype) from PRD section 21.3
REPRESENTATIONS = (
    ("RGB 8-bit", 3, 3, np.uint8),
    ("RGB 16-bit integer", 6, 3, np.uint16),
    ("RGB float16", 6, 3, np.float16),
    ("RGB float32", 12, 3, np.float32),
    ("RGBA float32", 16, 4, np.float32),
    ("Single-channel float32 mask", 4, 1, np.float32),
)

_PACKAGES = (
    "numpy",
    "scipy",
    "pillow",
    "opencv-python",
    "opencv-python-headless",
    "scikit-image",
    "tifffile",
    "OpenEXR",
    "PyYAML",
    "psutil",
    "matplotlib",
    "torch",
)


def inspect_runtime(path: str | Path = ".") -> dict:
    """Environment facts every run records (PRD 6.2 and 21.2)."""
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path(path).resolve() if Path(path).exists() else Path.cwd())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "in_colab": "google.colab" in sys.modules or _module_available("google.colab"),
        "ram_total_gb": round(memory.total / 1e9, 2),
        "ram_available_gb": round(memory.available / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "gpu": gpu_info(),
        "packages": package_versions(),
    }


def gpu_info() -> dict | None:
    """GPU name, memory and driver CUDA version from nvidia-smi, or None without an NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        query = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
        banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    if not query:
        return None
    name, total, free, driver = [part.strip() for part in query[0].split(",")]
    cuda = None
    if "CUDA Version:" in banner:
        cuda = banner.split("CUDA Version:")[1].split()[0]
    return {
        "name": name,
        "count": len(query),
        "memory_total_gb": round(float(total) / 1024, 2),
        "memory_free_gb": round(float(free) / 1024, 2),
        "driver_version": driver,
        "driver_cuda_version": cuda,
    }


def package_versions() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_revision(path: str | Path) -> dict | None:
    """Commit and dirty state of the checkout containing ``path``, if it is a Git checkout."""
    try:

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()

        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (subprocess.SubprocessError, OSError):
        return None


def estimate_memory(width: int, height: int) -> list[dict]:
    """PRD 21.3 base-image memory for the given dimensions, in decimal MB."""
    pixels = width * height
    return [
        {"representation": label, "bytes_per_pixel": bpp, "estimated_mb": round(pixels * bpp / MB, 1)}
        for label, bpp, _, _ in REPRESENTATIONS
    ]


def benchmark_representations(width: int, height: int) -> list[dict]:
    """Allocate and touch each PRD 21.3 representation in a fresh process and measure its resident memory.

    A separate interpreter per representation keeps allocator caching (which
    reuses freed pages inside one process) from hiding the true cost.
    """
    rows = []
    for (_, _, channels, dtype), estimate in zip(REPRESENTATIONS, estimate_memory(width, height), strict=True):
        result = subprocess.run(
            [sys.executable, "-c", _BENCHMARK_SCRIPT, str(height), str(width), str(channels), np.dtype(dtype).name],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        grown, seconds = result.stdout.split()
        rows.append(
            {**estimate, "measured_rss_mb": round(int(grown) / MB, 1), "allocate_seconds": round(float(seconds), 3)}
        )
    return rows


_BENCHMARK_SCRIPT = """
import sys, time
import numpy as np, psutil
height, width, channels, dtype = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
process = psutil.Process()
before = process.memory_info().rss
start = time.perf_counter()
array = np.ones((height, width, channels), dtype=dtype)  # ones() writes every page
elapsed = time.perf_counter() - start
print(process.memory_info().rss - before, elapsed)
"""


def estimate_storage(
    width: int,
    height: int,
    baselines: int,
    *,
    tiff: bool,
    exr: bool,
    exr_bytes_per_sample: int,
    source_pixels: int,
    save_intermediates: bool,
) -> dict:
    """Upper-bound output size (uncompressed) compared with free disk space (PRD 21.5)."""
    pixels = width * height
    per_baseline = (pixels * 6 if tiff else 0) + (pixels * 3 * exr_bytes_per_sample if exr else 0)
    intermediates = source_pixels * 12 if save_intermediates else 0
    return {
        "upper_bound_bytes": per_baseline * baselines + intermediates,
        "per_baseline_bytes": per_baseline,
        "intermediates_bytes": intermediates,
        "note": "uncompressed upper bound; lossless compression usually reduces it",
    }


@dataclass
class StageMemory:
    name: str
    seconds: float
    rss_start_mb: float
    rss_peak_mb: float
    rss_end_mb: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


@dataclass
class MemoryMonitor:
    """Samples resident memory on a background thread and attributes peaks to named stages."""

    interval: float = 0.005
    stages: list[StageMemory] = field(default_factory=list)
    peak_rss: int = 0

    def __post_init__(self) -> None:
        self._process = psutil.Process()
        self._lock = threading.Lock()
        self._window_peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> MemoryMonitor:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample, name="memory-monitor", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._record(self._process.memory_info().rss)
            self._stop.wait(self.interval)

    def _record(self, rss: int) -> int:
        with self._lock:
            self._window_peak = max(self._window_peak, rss)
            self.peak_rss = max(self.peak_rss, rss)
            return self._window_peak

    @contextmanager
    def stage(self, name: str):
        rss = self._process.memory_info().rss
        with self._lock:
            self._window_peak = rss
        start = time.perf_counter()
        try:
            yield
        finally:
            end_rss = self._process.memory_info().rss
            peak = self._record(end_rss)
            self.stages.append(StageMemory(name, time.perf_counter() - start, rss / MB, peak / MB, end_rss / MB))

    def __enter__(self) -> MemoryMonitor:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _module_available(name: str) -> bool:
    try:
        from importlib.util import find_spec

        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def process_peak_rss_mb() -> float:
    """Lifetime resident-memory high-water mark of this process, from the OS."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((peak if sys.platform == "darwin" else peak * 1024) / MB, 1)
