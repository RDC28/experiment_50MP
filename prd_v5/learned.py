"""Explicit experimental camera-space residual contract; no bundled model loader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from prd_v5.hypothesis import CameraHypothesis


def camera_domain(camera: CameraHypothesis) -> str:
    """Bind adapters to the exact shared camera hypothesis, including seeds."""
    return hashlib.sha256(json.dumps(asdict(camera), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ResidualAdapter:
    """Caller-loaded, deterministic tile predictor returning (residual, confidence).

    Inputs are encoded sRGB and clean camera RGB, both HxWx3; residual has
    identical shape in base-ISO-full-well units and confidence is HxW in [0,1].
    Predictors receive absolute (top,left) and the immutable camera. They must
    supply their own receptive-field handling and never add stochastic noise.
    No RAW-Flow/SpiralDiff compatibility or scientific calibration is implied.
    """

    predict: Callable
    checkpoint_path: str
    sha256: str
    camera_sha256: str
    source: str
    licence: str
    licence_accepted: bool = False
    experimental_accepted: bool = False
    normalization: str = "clean_camera_rgb_base_iso_full_well"

    def verify(self, camera: CameraHypothesis) -> dict:
        if not callable(self.predict) or self.experimental_accepted is not True or self.licence_accepted is not True:
            raise ValueError("Callable adapter and explicit experimental/licence acceptance required")
        if not self.source.strip() or not self.licence.strip():
            raise ValueError("Adapter source and licence required")
        if self.normalization != "clean_camera_rgb_base_iso_full_well" or self.camera_sha256 != camera_domain(camera):
            raise ValueError("Adapter normalization or camera domain mismatch")
        digest = hashlib.sha256()
        with Path(self.checkpoint_path).open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != self.sha256:
            raise ValueError("Residual checkpoint SHA-256 mismatch")
        return {
            "sha256": self.sha256,
            "camera_sha256": self.camera_sha256,
            "source": self.source,
            "licence": self.licence,
            "normalization": self.normalization,
            "integrity": "verified bytes; callable/checkpoint relationship is caller responsibility",
        }


def resolve_mode(mode, adapter, camera) -> dict:
    if mode not in {"ANALYTIC", "HYBRID", "LEARNED_RGB2RAW"}:
        raise ValueError("Unknown inverse ISP mode")
    if mode == "ANALYTIC" and adapter is not None:
        raise ValueError("ANALYTIC mode cannot execute a residual adapter")
    if mode == "LEARNED_RGB2RAW" and adapter is None:
        raise ValueError("Strict learned mode requires a compatible residual adapter")
    evidence = adapter.verify(camera) if adapter is not None else None
    return {
        "requested": mode,
        "effective": "HYBRID" if adapter is not None else "ANALYTIC",
        "status": "ready; not executed"
        if adapter is not None
        else "disabled"
        if mode == "ANALYTIC"
        else "skipped learned branch: no compatible camera-adapted checkpoint/loader supplied",
        "checkpoint": evidence,
        "validation": "experimental residual contract; scientific and Colab validation pending",
    }


def apply_residual(rgb, clean, camera, adapter, *, mode="HYBRID", tile=1024, threshold=0.5) -> dict:
    """Modify disk-backed clean expectation in bounded tiles before CFA/noise."""
    report = resolve_mode(mode, adapter, camera)
    if type(tile) is not int or tile < 1 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid residual tile or confidence threshold")
    if adapter is None:
        return report
    if rgb.shape != clean.shape or clean.ndim != 3 or clean.shape[-1] != 3:
        raise ValueError("Residual inputs must be matching HxWx3 arrays")
    accepted = changed = floored = 0
    for top in range(0, clean.shape[0], tile):
        for left in range(0, clean.shape[1], tile):
            region = np.s_[top : top + tile, left : left + tile]
            baseline = np.array(clean[region], copy=True)
            encoded = np.array(rgb[region], copy=True)
            baseline.setflags(write=False)
            encoded.setflags(write=False)
            residual, confidence = adapter.predict(encoded, baseline, (top, left), camera)
            residual, confidence = np.asarray(residual), np.asarray(confidence)
            if residual.shape != baseline.shape or confidence.shape != baseline.shape[:2]:
                raise ValueError("Residual/confidence shape mismatch")
            if not np.isfinite(residual).all() or not np.isfinite(confidence).all():
                raise ValueError("Nonfinite residual/confidence")
            if np.any(confidence < 0) or np.any(confidence > 1):
                raise ValueError("Confidence must be in [0,1]")
            gate = confidence >= threshold
            corrected = baseline.astype(np.float64) + np.where(gate[..., None], residual * confidence[..., None], 0)
            if not np.isfinite(corrected).all() or np.any(corrected > np.finfo(np.float32).max):
                raise ValueError("Residual exceeds clean float32 range")
            floored += int(np.count_nonzero(corrected < 0))
            corrected = np.maximum(corrected, 0).astype(np.float32)
            changed += int(np.count_nonzero(corrected != baseline))
            accepted += int(np.count_nonzero(gate))
            clean[region] = corrected
    clean.flush()
    return {
        **report,
        "status": "executed experimental residual",
        "accepted_pixels": accepted,
        "changed_samples": changed,
        "zero_effect": changed == 0,
        "floored_samples": floored,
        "tile": tile,
        "confidence_threshold": threshold,
        "sensor_noise_in_network": False,
    }
