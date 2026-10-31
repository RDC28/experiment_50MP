"""Adobe dng_validate conformance: locate or build the tool, run it, record the outcome.

The gate is "not measured" unless the real tool ran. A nonzero exit or an error
line in its output is "failed"; a clean run is "passed". Nothing here claims
conformance from the tool's mere presence.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ENV_VAR = "SYNTHRAW_DNG_VALIDATE"
SKIP_ENV_VAR = "SYNTHRAW_DNG_VALIDATE_SKIP"
BUILD_SCRIPT = Path(__file__).parent / "tools" / "build_dng_validate.sh"
# Adobe publishes no digest; this value is a local observation of one fetch of the
# conventional download path (see EVALUATION.md), pinned so a changed archive is refused.
SDK_URL = "https://download.adobe.com/pub/adobe/dng/dng_sdk_1_7_1.zip"
SDK_SHA256 = "5366d9abd0e623858573bfaf0291b5e352d704f51b322753a22f82af3ae6e0c2"


def not_measured(reason: str) -> dict:
    return {"status": "not measured", "reason": reason}


def find_dng_validate(path: str | os.PathLike | None = None) -> Path | None:
    """Resolve an explicit path, then the environment variable, then PATH."""
    for candidate in (path, os.environ.get(ENV_VAR)):
        if candidate:
            found = Path(candidate)
            return found if found.is_file() and os.access(found, os.X_OK) else None
    located = shutil.which("dng_validate")
    return Path(located) if located else None


def build_dng_validate(
    workdir: str | os.PathLike, *, script: str | os.PathLike = BUILD_SCRIPT, timeout_s: int = 3600
) -> dict:
    """Run the Linux build script; a failure is a visible skip, never an exception."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run(
            ["bash", str(script), str(workdir), SDK_URL, SDK_SHA256],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return not_measured(f"dng_validate build did not run to completion: {error}")
    binary = workdir / "dng_validate"
    if done.returncode != 0 or not binary.is_file():
        tail = (done.stderr or done.stdout).strip().splitlines()[-12:]
        return not_measured(f"dng_validate build failed (exit {done.returncode}): " + " | ".join(tail))
    return {"status": "built", "binary": str(binary), "sdk_sha256": SDK_SHA256}


def run_dng_validate(dng: str | os.PathLike, binary: str | os.PathLike | None = None, *, timeout_s: int = 1800) -> dict:
    """Validate one DNG with Adobe's tool and report what it said."""
    dng = Path(dng)
    tool = find_dng_validate(binary)
    if tool is None:
        return not_measured(
            os.environ.get(SKIP_ENV_VAR)
            or f"dng_validate not available: pass a prebuilt path, set {ENV_VAR}, or let the notebook setup cell "
            "build it"
        )
    try:
        done = subprocess.run([str(tool), "-v", str(dng)], capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as error:
        return not_measured(f"dng_validate did not complete: {error}")
    output = (done.stdout or "") + (done.stderr or "")
    errors = [line.strip() for line in output.splitlines() if "error" in line.lower()]
    record = {
        "status": "passed" if done.returncode == 0 and not errors else "failed",
        "tool": str(tool),
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "exit_code": done.returncode,
        "error_lines": errors[:20],
        "output_tail": output.strip().splitlines()[-20:],
    }
    return record
