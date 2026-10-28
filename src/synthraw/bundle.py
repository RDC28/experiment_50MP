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
RUNTIME_ONLY = "the full-resolution intermediates exist only on this Colab runtime and are lost when it recycles"


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


def drive_run_folder(mount_path: str | Path, *, mount: bool) -> tuple[Path | None, str | None]:
    """Return ``MyDrive/synthraw_runs`` under ``mount_path``, mounting Google Drive first when ``mount`` is set.

    An already mounted Drive is used as is. A declined or failed mount returns a warning instead of
    raising: the full-resolution intermediates then exist only on the Colab runtime.
    """
    my_drive = Path(mount_path, "MyDrive")
    if not my_drive.is_dir() and mount:
        try:
            from google.colab import drive

            drive.mount(str(mount_path))
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
        else:
            reason = None if my_drive.is_dir() else f"{my_drive} not found after mounting"
        if reason is not None:
            return None, f"Google Drive was not mounted ({reason}); {RUNTIME_ONLY}"
    if not my_drive.is_dir():
        return None, None
    folder = my_drive / "synthraw_runs"
    folder.mkdir(exist_ok=True)
    return folder, None


def copy_run_to_drive(run_root: str | Path, drive_dir: str | Path) -> tuple[Path | None, str | None]:
    """Copy the whole run folder, full-resolution intermediates included, into ``drive_dir``.

    Run it before the run report and the zip are written so a failure warning can go into both;
    ``sync_to_drive`` then copies the files written afterwards. A failed copy (quota, I/O) returns a
    warning instead of raising.
    """
    run_root = Path(run_root)
    try:
        return Path(shutil.copytree(run_root, Path(drive_dir) / run_root.name, dirs_exist_ok=True)), None
    except OSError as error:
        return None, f"Copying the run folder to Google Drive failed ({type(error).__name__}: {error}); {RUNTIME_ONLY}"


# Files written after ``copy_run_to_drive``: the run report, the bundle manifest and the finished log.
FINAL_RUN_FILES = ("validation/run_report.json", "logs/bundle_manifest.json", "logs/execution.log")


def sync_to_drive(run_root: str | Path, drive_dir: str | Path, bundle: str | Path) -> str | None:
    """Copy ``FINAL_RUN_FILES`` into the Drive run folder and the bundle zip beside it.

    A failed copy (quota, I/O) returns a warning instead of raising.
    """
    run_root, drive, bundle = Path(run_root), Path(drive_dir), Path(bundle)
    try:
        for name in FINAL_RUN_FILES:
            target = drive / run_root.name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_root / name, target)
        shutil.copy2(bundle, drive / bundle.name)
    except OSError as error:
        return (
            f"Copying the run report, logs and {bundle.name} to Google Drive failed ({type(error).__name__}: "
            f"{error}); download the zip from this Colab runtime before it recycles"
        )
    return None


def deliver_bundle(bundle: str | Path, *, max_download_bytes: int = MAX_BROWSER_DOWNLOAD_BYTES) -> dict[str, Any]:
    """Offer the zip via google.colab.files when present; report, never raise, when it is unavailable.

    A zip larger than ``max_download_bytes`` is not offered to the browser.
    """
    bundle = Path(bundle)
    size = bundle.stat().st_size
    status: dict[str, Any] = {"bundle": str(bundle), "download": "unavailable"}
    try:
        from google.colab import files
    except ImportError:
        files = None
    if files is not None:
        if size > max_download_bytes:
            status["download"] = (
                f"skipped: {size / 1024**2:.0f} MB exceeds the {max_download_bytes / 1024**2:.0f} MB "
                "browser download limit"
            )
        else:
            files.download(str(bundle))
            status["download"] = "offered"
    return status


# The run's final image: the LibRaw-developed 16-bit TIFF master and the synthetic DNG it was developed from.
FINAL_IMAGE_FILES = {"tiff16": "masters/developed_50mp_16bit.tif", "dng": "raw/synthetic_50mp.dng"}


def deliver_final_image(
    run_root: str | Path,
    *,
    drive_dir: str | Path | None = None,
    download: bool = True,
    max_download_bytes: int = MAX_BROWSER_DOWNLOAD_BYTES,
    log: Any = print,
) -> dict[str, Any]:
    """Hand the final TIFF and DNG to the person running the notebook, printing every path.

    With ``drive_dir`` (Google Drive mounted), each file is placed at ``drive_dir/<run>/<same path>``, the
    layout ``copy_run_to_drive`` uses, and copied only when that copy is missing or differs in size. With
    ``download`` in Colab, each file up to ``max_download_bytes`` is offered as a browser download. Failures
    are reported in the result and logged, never raised.
    """
    root = Path(run_root)
    result: dict[str, Any] = {"files": {}, "missing": [], "drive": {}, "downloads": {}, "warnings": []}
    for kind, name in FINAL_IMAGE_FILES.items():
        path = root / name
        if path.is_file():
            result["files"][kind] = str(path)
            log(f"Final {kind}: {path} ({path.stat().st_size / 1024**2:.1f} MB)")
        else:
            result["missing"].append(name)
            log(f"Final {kind} not written: {path} is missing (check WRITE_TIFF16 / WRITE_DNG and the run log)")
    if drive_dir is not None:
        for kind, local in result["files"].items():
            source = Path(local)
            target = Path(drive_dir) / root.name / source.relative_to(root)
            try:
                if not target.is_file() or target.stat().st_size != source.stat().st_size:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            except OSError as error:
                reason = f"{type(error).__name__}: {error}"
                result["warnings"].append(f"Copying {source.name} to Google Drive failed ({reason})")
                continue
            result["drive"][kind] = str(target)
            log(f"On Google Drive: {target}")
    if download:
        try:
            from google.colab import files
        except ImportError:
            files = None
            log("Not in Colab: no browser download; use the paths above")
        offered = result["files"].items() if files is not None else ()
        for kind, local in offered:
            size = Path(local).stat().st_size
            if size > max_download_bytes:
                result["downloads"][kind] = (
                    f"skipped: {size / 1024**2:.0f} MB exceeds the {max_download_bytes / 1024**2:.0f} MB "
                    "browser download limit"
                )
            else:
                files.download(local)
                result["downloads"][kind] = "offered"
            log(f"Browser download of {Path(local).name}: {result['downloads'][kind]}")
    for warning in result["warnings"]:
        log(f"WARNING: {warning}")
    return result
