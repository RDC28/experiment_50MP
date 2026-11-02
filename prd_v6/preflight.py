"""Driver step 3: runtime tier, resources, versions and verified downloads (sections 8.1, 9.1).

Any failure raises PreflightError before heavy work, which makes the run INVALID with the reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
import urllib.request
from pathlib import Path

from prd_v6 import constants

MIN_RAM_GB = 48
MIN_DISK_GB = 80
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
ACCEPTED_GPUS = ("A100", "H100")
WARNED_GPUS = ("L4",)


class PreflightError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, *, size: int | None, digest: str | None, log=print, attempts: int = 3) -> dict:
    """Fetch with a browser-like user agent and retries; verify size and SHA-256 when pinned."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size) and (digest is None or sha256(dest) == digest):
        return {"url": url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": sha256(dest), "cached": True}
    last = None
    for attempt in range(1, attempts + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 22)
            got_size, got_sha = tmp.stat().st_size, sha256(tmp)
            if size is not None and got_size != size:
                raise PreflightError(f"{url}: {got_size} bytes, expected {size}")
            if digest is not None and got_sha != digest:
                raise PreflightError(f"{url}: SHA-256 {got_sha}, expected {digest}")
            tmp.replace(dest)
            log(f"downloaded {dest.name} ({got_size} bytes, sha256 {got_sha})")
            return {"url": url, "path": str(dest), "bytes": got_size, "sha256": got_sha, "cached": False}
        except PreflightError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as error:  # noqa: BLE001 - network errors are retried, then reported
            tmp.unlink(missing_ok=True)
            last = error
            log(f"download attempt {attempt} of {url} failed: {type(error).__name__}: {error}")
            time.sleep(5 * attempt)
    raise PreflightError(f"download failed after {attempts} attempts: {url}: {last}")


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("numpy", "scipy", "cv2", "PIL", "tifffile", "rawpy", "skimage", "matplotlib", "torch", "spandrel"):
        try:
            imported = __import__(module)
            out[module] = getattr(imported, "__version__", "unknown")
        except Exception as error:  # noqa: BLE001
            out[module] = f"not importable: {type(error).__name__}"
    try:
        import rawpy

        out["libraw"] = ".".join(str(v) for v in rawpy.libraw_version)
    except Exception:  # noqa: BLE001
        pass
    return out


def classify(gpu: str | None, ram_gb: float, disk_free_gb: float) -> tuple[list[str], list[str]]:
    """Section 9.1: (problems that stop the run, warnings that do not)."""
    gpu = gpu or ""
    problems, warnings = [], []
    if not gpu:
        problems.append("no GPU: CPU-only runtimes are rejected (PRD v6 section 9.1)")
    elif "T4" in gpu:
        problems.append(f"{gpu} is rejected: choose an A100 runtime (PRD v6 section 9.1)")
    elif any(name in gpu for name in WARNED_GPUS):
        warnings.append(f"{gpu} accepted with a warning: same results, slower SR than the stated A100 tier")
    elif not any(name in gpu for name in ACCEPTED_GPUS):
        warnings.append(f"{gpu} is not the stated A100 tier; continuing")
    if ram_gb < MIN_RAM_GB:
        problems.append(f"host RAM {ram_gb:.0f} GB < {MIN_RAM_GB} GB: choose High-RAM")
    if disk_free_gb < MIN_DISK_GB:
        problems.append(f"free disk {disk_free_gb:.0f} GB < {MIN_DISK_GB} GB")
    return problems, warnings


def runtime_tier(disk_path: Path) -> dict:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    tier: dict = {"ram_gb": ram / 1e9, "disk_free_gb": shutil.disk_usage(disk_path).free / 1e9, "gpu": None}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            tier.update(gpu=props.name, vram_gb=props.total_memory / 1e9, cuda=torch.version.cuda)
    except Exception as error:  # noqa: BLE001
        tier["gpu_error"] = f"{type(error).__name__}: {error}"
    problems, warnings = classify(tier["gpu"], tier["ram_gb"], tier["disk_free_gb"])
    tier.update(problems=problems, warnings=warnings)
    return tier


def run(run_dir: Path, settings: dict, log=print) -> dict:
    """Check the runtime, download and verify inputs, load HAT-L. Returns the inputs for step 4."""
    from prd_v5.dng_validate import ENV_VAR, find_dng_validate
    from prd_v6 import sr
    from prd_v6.install import DNG_VALIDATE_BUILD

    run_dir = Path(run_dir)
    tier = runtime_tier(run_dir)
    environment = {"tier": tier, "versions": versions()}
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, default=str))
    for warning in tier["warnings"]:
        log(f"preflight warning: {warning}")
    log(
        f"runtime: GPU {tier.get('gpu')} ({tier.get('vram_gb', 0):.0f} GB), RAM {tier['ram_gb']:.0f} GB, "
        f"free disk {tier['disk_free_gb']:.0f} GB"
    )
    if tier["problems"]:
        raise PreflightError("; ".join(tier["problems"]))
    build = json.loads((run_dir / DNG_VALIDATE_BUILD).read_text()) if (run_dir / DNG_VALIDATE_BUILD).exists() else {}
    tool = find_dng_validate(build.get("binary")) if build.get("status") == "built" else None
    if tool is None:
        raise PreflightError(f"dng_validate is not available: {build.get('reason') or 'setup did not build it'}")
    os.environ[ENV_VAR] = str(tool)
    inputs = run_dir / "inputs"
    arw = download(
        settings.get("REFERENCE_ARW_URL") or constants.REFERENCE_ARW_URL,
        inputs / constants.REFERENCE_ARW_NAME,
        size=constants.REFERENCE_ARW_SIZE,
        digest=constants.REFERENCE_ARW_SHA256,
        log=log,
    )
    checkpoint = download(
        constants.HAT_L_URL,
        inputs / constants.HAT_L_NAME,
        size=constants.HAT_L_SIZE,
        digest=constants.HAT_L_SHA256,
        log=log,
    )
    model, model_info = sr.load_hat_l(checkpoint["path"])
    log(f"HAT-L loaded: {model_info}")
    environment.update(inputs={"reference_arw": arw, "hat_l": checkpoint}, model=model_info, dng_validate=str(tool))
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, default=str))
    return {"arw": Path(arw["path"]), "model": model, "model_info": model_info, "dng_validate": str(tool)}
