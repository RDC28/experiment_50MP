"""Runtime evidence for analytic replay, without host identity or credentials."""

import platform

import numpy as np
import PIL
import rawpy
import tifffile
from PIL import features


def runtime_record() -> dict:
    """Record loaded numerical/codec versions, not a reproducibility guarantee.

    Native LibRaw and LittleCMS versions matter independently of their Python
    bindings. No hostname, username, environment variables or paths are captured.
    Missing optional LittleCMS support is represented as null, never guessed.
    """
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "libraries": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "littlecms": features.version("littlecms2"),
            "tifffile": tifffile.__version__,
            "rawpy": rawpy.__version__,
            "libraw": ".".join(str(part) for part in rawpy.libraw_version),
        },
        "reproducibility": (
            "Same-runtime tiny replay tested; cross-runtime bitwise reproducibility not validated. "
            "Versions describe the loaded runtime, not pinned dependencies or camera calibration."
        ),
    }


def software_identity():
    """Version and source revision, without host paths or credentials."""
    import os
    import re
    import subprocess
    from importlib.metadata import version
    from pathlib import Path

    supplied = os.environ.get("SYNTHRAW_SOURCE_REVISION", "")
    if re.fullmatch(r"[0-9a-f]{40}", supplied):
        return f"prd-v5 {version('synthetic-raw-50mp-lab')} git:{supplied}"
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        revision = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        revision = "unavailable"
    return f"prd-v5 {version('synthetic-raw-50mp-lab')} git:{revision}"
