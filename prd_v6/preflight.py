"""Driver step 3: runtime tier, resources, versions and verified downloads (sections 8.1, 9.1).

Any failure raises PreflightError before heavy work, which makes the run INVALID with the reason.
"""

from __future__ import annotations

import hashlib
import html
import http.cookiejar
import json
import os
import platform
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

from prd_v6 import constants

MIN_RAM_GB = 48
MIN_DISK_GB = 80
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
ACCEPTED_GPUS = ("A100", "H100")
WARNED_GPUS = ("L4",)


DRIVE_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com")
DRIVE_DIRECT = "https://drive.google.com/uc?export=download&id={}"
OWN_SOURCES = (("4k_source", "SOURCE_4K_URL"), ("2k_source", "SOURCE_2K_URL"))


class PreflightError(RuntimeError):
    pass


def drive_file_id(url: str) -> str | None:
    """The file id of a Google Drive share link (file/d/<id>/view, open?id=, uc?id=), else None."""
    parts = urllib.parse.urlparse(url)
    if parts.hostname not in DRIVE_HOSTS:
        return None
    found = re.search(r"/d/([\w-]+)", parts.path)
    if found:
        return found.group(1)
    ids = urllib.parse.parse_qs(parts.query).get("id")
    return ids[0] if ids else None


def own_image_dest(inputs: Path, key: str, url: str) -> Path:
    suffix = "" if drive_file_id(url) else Path(url.split("?")[0]).suffix
    return Path(inputs) / f"{key}{suffix or '.img'}"


def _confirm_url(page: str, base: str) -> str | None:
    """The large-file confirmation form (or link) of a Drive 'cannot scan for viruses' page."""
    form = re.search(r'<form[^>]*action="([^"]+)"[^>]*>(.*?)</form>', page, re.S)
    if form:
        fields = re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', form.group(2))
        query = urllib.parse.urlencode([(name, html.unescape(value)) for name, value in fields])
        return urllib.parse.urljoin(base, html.unescape(form.group(1))) + ("?" + query if query else "")
    link = re.search(r'href="(/uc\?export=download[^"]+)"', page)
    return urllib.parse.urljoin(base, html.unescape(link.group(1))) if link else None


def _fetch(url: str, tmp: Path) -> None:
    """Stream `url` to `tmp`; a Drive link goes through its direct download and large-file confirmation."""
    file_id = drive_file_id(url)
    target = DRIVE_DIRECT.format(file_id) if file_id else url
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    for _ in range(3 if file_id else 1):
        request = urllib.request.Request(target, headers={"User-Agent": USER_AGENT})
        with opener.open(request, timeout=120) as response:
            if file_id and response.headers.get_content_type() == "text/html":
                page = response.read(1 << 22).decode("utf-8", "replace")
                target = _confirm_url(page, response.geturl())
                if target is None:
                    raise PreflightError(
                        f"{url}: Google Drive returned a web page, not the file; share it as 'Anyone with the link'"
                    )
                continue
            with open(tmp, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 22)
            return
    raise PreflightError(f"{url}: Google Drive kept asking for confirmation instead of sending the file")


def check_image(path: Path, setting: str) -> None:
    """Fail early, naming the setting, when a downloaded source image does not decode (e.g. an HTML page)."""
    from PIL import Image

    try:
        with Image.open(path) as opened:
            opened.verify()
    except Exception as error:  # noqa: BLE001 - any decode failure means the bytes are not an image
        size = Path(path).stat().st_size
        Path(path).unlink(missing_ok=True)
        raise PreflightError(
            f"{setting}: the download ({size} bytes) is not a decodable image ({type(error).__name__}). "
            "Use a direct image URL or a Google Drive link shared as 'Anyone with the link'."
        ) from error


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
            _fetch(url, tmp)
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
    for key, setting in OWN_SOURCES:  # the captain's own images: fail now, not after the benchmark
        if settings.get(setting):
            dest = own_image_dest(inputs, key, settings[setting])
            download(settings[setting], dest, size=None, digest=None, log=log)
            check_image(dest, setting)
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
