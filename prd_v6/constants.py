"""Pinned inputs, camera-model constants and gate thresholds (PRD v6 sections 4, 5, 6 and 8)."""

from __future__ import annotations

import math

# Section 8.1: the reference ARW, fetched from the publisher at run time and never redistributed.
REFERENCE_ARW_URL = "https://img.photographyblog.com/reviews/sony_a1_ii/sample_images/sony_a1_ii_65.arw"
REFERENCE_ARW_NAME = "sony_a1_ii_65.arw"
REFERENCE_ARW_SIZE = 106_033_152
REFERENCE_ARW_SHA256 = "dc87580c1d6111cbaef530478fa1f7863a02a5482cd7cab9a902d731c20b4893"

# Section 8.1: HAT-L x4 from a third-party Hugging Face mirror, pinned by revision and hash.
HAT_L_NAME = "HAT-L_SRx4_ImageNet-pretrain.pth"
HAT_L_REVISION = "cf5cb1c188096b795e608629afd9a864646dbffa"
HAT_L_URL = f"https://huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain/resolve/{HAT_L_REVISION}/{HAT_L_NAME}"
HAT_L_SIZE = 165_774_123
HAT_L_SHA256 = "5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043"
# HAT-L has about 40.8 M parameters and HAT about 20.8 M; anything below this bound is not HAT-L.
HAT_L_MIN_PARAMETERS = 35_000_000

# Camera-model constants of the A1 II (docs/reference-arw-comparison.md). The colour matrix lives in
# synthraw.camera_profile so there is one source of truth for it.
CFA_PATTERN = "RGGB"
BLACK_14 = 512
CLIP_14 = 15360
SIGNAL_14 = CLIP_14 - BLACK_14  # 14,848 DN of signal
GAIN_E_PER_DN14 = 2.08  # green-plane photon transfer
READ_NOISE_E = 4.0  # single-frame upper bound
FULL_WELL_E = 33_000.0

# Section 4: every output stores the 14-bit scale multiplied by exactly 4.
SCALE_16 = 4
BLACK_16 = BLACK_14 * SCALE_16  # 2048
WHITE_16 = CLIP_14 * SCALE_16  # 61440
SIGNAL_16 = SIGNAL_14 * SCALE_16  # 59,392

# Section 5.4: output names, sizes (height, width) and their downsampling ratio from 50 MP.
VISIBLE_HW = (5760, 8640)
OUTPUT_RATIOS = {"2k": 4.5, "4k": 2.25, "50mp": 1.0}
OUTPUT_LABELS = {"2k": "2K", "4k": "4K", "50mp": "50 MP"}
OUTPUT_ORDER = ("2k", "4k", "50mp")

# Section 5.3: the illuminant prior is the Planckian point at the mired midpoint of this range.
CCT_LOW_K = 2850.0
CCT_HIGH_K = 7500.0
PRIOR_CCT_K = 2.0 / (1.0 / CCT_LOW_K + 1.0 / CCT_HIGH_K)

# Section 9.3: maximum quality settings.
SR_SCALE = 4
SR_TILE = 512
SR_OVERLAP = 128
BACK_PROJECTION_ITERATIONS = 3
NOISE_SEED = 6

# Section 6: gate thresholds. Fixed; the notebook never relaxes them.
P1_MIN = 0.95
P2_MIN = 0.95
P2_RELATIVE = 0.05
P2_SIGMAS = 3.0
CLIPPED_FRACTION_OF_CLIP = 0.98
MAX_CLIPPED_FRACTION = 0.01
BORDER_PHOTOSITES = 16
NOISE_FLOOR_P1_MIN = 0.97
NOISE_FLOOR_P2_MIN = 0.99
SENSITIVITY_MIN_DROP = 0.01
ABLATION_MIN_DELTA = -0.001
NOISE_GATE_TOLERANCE = 0.15
SEAM_RATIO_MAX = 1.25


def border_pixels(ratio: float) -> int:
    """The 16-photosite border, rounded up to whole output pixels (section 6.2)."""
    return math.ceil(BORDER_PHOTOSITES / ratio - 1e-9)


def output_shape(visible_hw: tuple[int, int], name: str) -> tuple[int, int]:
    """Exact output size for a visible area; the ratios must divide it (section 6.1)."""
    ratio = OUTPUT_RATIOS[name]
    height, width = (side / ratio for side in visible_hw)
    if not (height.is_integer() and width.is_integer()):
        raise ValueError(f"{name} ratio {ratio} does not divide the visible area {visible_hw}")
    return int(height), int(width)
