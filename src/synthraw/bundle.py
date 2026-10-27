"""Run-folder bundle zip and Colab delivery for the PRD_v3 output bundle."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from synthraw.run_layout import RUN_DIRECTORIES

# Disk-backed full-resolution arrays; every other PRD tree file is always bundled.
LARGE_ARRAY_DIRECTORIES = ("intermediates", "confidence")
# Larger zips are not offered as a browser download; the Drive copy still carries them.
MAX_BROWSER_DOWNLOAD_BYTES = 2 * 1024**3


def bundle_run(
    run_root: str | Path, *, exclude_large_arrays: bool = False, destination: str | Path | None = None
) -> Path:
    """Zip the whole run tree next to it; optionally omit intermediates/ and confidence/ arrays.

    ``logs/bundle_manifest.json`` records every bundled and omitted file. The zip is staged
    and renamed so a partial archive is never published.
    """
    root = Path(run_root)
    missing = [name for name in RUN_DIRECTORIES if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Not a run folder; missing {missing}: {root}")
    target = Path(destination) if destination is not None else root.with_name(root.name + ".zip")
    if target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Bundle destination must be outside the run folder")
    included, omitted = [], []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        large = exclude_large_arrays and relative.parts[0] in LARGE_ARRAY_DIRECTORIES and path.suffix == ".npy"
        (omitted if large else included).append(relative.as_posix())
    manifest_name = "logs/bundle_manifest.json"
    manifest = {
        "run": root.name,
        "exclude_large_arrays": exclude_large_arrays,
        "included": sorted({*included, manifest_name}),
        "omitted_large_arrays": omitted,
    }
    (root / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staged = target.with_name(f".{target.name}.partial")
    try:
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in manifest["included"]:
                archive.write(root / name, f"{root.name}/{name}")
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    return target


def deliver_bundle(
    bundle: str | Path,
    *,
    drive_dir: str | Path | None = None,
    run_root: str | Path | None = None,
    max_download_bytes: int = MAX_BROWSER_DOWNLOAD_BYTES,
) -> dict[str, Any]:
    """Offer the zip via google.colab.files and copy it, plus the full ``run_root`` tree, to mounted Drive.

    A zip larger than ``max_download_bytes`` is not offered to the browser. The Drive copy of
    ``run_root`` keeps every full-resolution intermediate a lightweight bundle omits. No delivery
    channel is required: unavailable channels are reported, not raised.
    """
    bundle = Path(bundle)
    size = bundle.stat().st_size
    status: dict[str, Any] = {"bundle": str(bundle), "download": "unavailable", "drive_copy": None}
    try:
        from google.colab import files
    except ImportError:
        files = None
    if files is not None:
        if size > max_download_bytes:
            status["download"] = (
                f"skipped: {size / 1e6:.0f} MB exceeds the {max_download_bytes / 1e6:.0f} MB browser download limit"
            )
        else:
            files.download(str(bundle))
            status["download"] = "offered"
    if drive_dir is not None:
        drive = Path(drive_dir)
        if drive.is_dir():
            status["drive_copy"] = str(shutil.copy2(bundle, drive / bundle.name))
            if run_root is not None:
                run_root = Path(run_root)
                status["drive_run_copy"] = str(shutil.copytree(run_root, drive / run_root.name, dirs_exist_ok=True))
        else:
            status["drive_copy"] = f"skipped: {drive} is not a mounted directory"
    return status
