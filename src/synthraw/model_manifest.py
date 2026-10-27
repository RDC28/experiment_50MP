"""Lazy, integrity-checked checkpoints for PRD_v3 checkpoint and licence policy."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen

UNPINNED = "UNPINNED - disabled until pinned"


@dataclass(frozen=True)
class ModelSpec:
    """A repository revision and checkpoint digest, never an inferred pin."""

    name: str
    repository: str | None
    commit: str | None
    checkpoint_url: str | None
    sha256: str | None
    licence: str
    licence_gate: bool = False

    @property
    def status(self) -> str:
        """Incomplete pins cannot be downloaded, even with licence acceptance."""
        if (
            self.repository
            and self.repository.startswith("https://")
            and self.commit
            and re.fullmatch(r"[0-9a-f]{7,40}", self.commit)
            and self.checkpoint_url
            and self.checkpoint_url.startswith("https://")
            and self.sha256
            and re.fullmatch(r"[0-9a-f]{64}", self.sha256)
        ):
            return "PINNED"
        return UNPINNED


# Revisions and checkpoint digests transcribed from cycle 1 docs/models.md.
# Seven-character revisions are the verified repository revisions recorded there.
MODEL_MANIFEST = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "SWINIR",
            "https://github.com/JingyunLiang/SwinIR",
            "6545850",
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
            "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215",
            "Apache-2.0",
        ),
        ModelSpec(
            "REAL_ESRGAN",
            "https://github.com/xinntao/Real-ESRGAN",
            "a4abfb2",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            "BSD-3-Clause",
        ),
        ModelSpec("HAT", "https://github.com/XPixelGroup/HAT", None, None, None, "Apache-2.0"),
        ModelSpec("SUPIR", "https://github.com/Fanghua-Yu/SUPIR", None, None, None, "Restricted/non-commercial", True),
        ModelSpec("EXPANDDIFF", None, None, None, None, "Unverified - acceptance required", True),
        ModelSpec("VID2HDR", None, None, None, None, "MIT code; restricted SVD dependency", True),
        ModelSpec("RAW_FLOW", None, None, None, None, "MIT code; weights unverified", True),
        ModelSpec("SPIRALDIFF", None, None, None, None, "MIT code; weights unverified", True),
        ModelSpec("SPECTRAL_FROM_AI_IMAGE", None, None, None, None, "No method selected - acceptance required", True),
        ModelSpec(
            "GEOCALIB", "https://github.com/cvg/GeoCalib", None, None, None, "Apache-2.0 code; CC-BY-4.0 weights", True
        ),
    )
}


def write_model_manifest(run_root: str | Path) -> Path:
    """Record every enabled and disabled checkpoint in logs/model_manifest.json."""
    path = Path(run_root) / "logs" / "model_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": [{**asdict(spec), "status": spec.status} for spec in MODEL_MANIFEST.values()]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _require_colab() -> None:
    try:
        importlib.import_module("google.colab")
    except ImportError as exc:
        raise RuntimeError("Checkpoint downloads are permitted only inside Colab") from exc


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint(
    name: str,
    *,
    cache_dir: str | Path = "/content/synthraw_cache",
    accept_licence: bool = False,
) -> Path:
    """Fetch only on explicit Colab runtime invocation; verify cached bytes too."""
    spec = MODEL_MANIFEST[name]
    if spec.status != "PINNED":
        raise ValueError(f"{name}: {spec.status}")
    if spec.licence_gate:
        if not accept_licence:
            raise PermissionError(f"{name} requires explicit licence acceptance: {spec.licence}")
        warnings.warn(f"{name} licence restrictions: {spec.licence}", UserWarning, stacklevel=2)
    _require_colab()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    # Keep configured caches out of Git even if placed inside a checkout.
    (cache / ".gitignore").write_text("*\n", encoding="utf-8")
    target = cache / f"{name.lower()}-{spec.sha256}.pth"
    if target.exists():
        if _digest(target) != spec.sha256:
            raise ValueError(f"SHA-256 mismatch in cached checkpoint: {name}")
        return target
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            with urlopen(spec.checkpoint_url, timeout=60) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if _digest(temporary) != spec.sha256:
            raise ValueError(f"SHA-256 mismatch in downloaded checkpoint: {name}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
