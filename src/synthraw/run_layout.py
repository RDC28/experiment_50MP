"""Run directories and independent random streams for PRD_v3 output and seeds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

RUN_DIRECTORIES = (
    "source",
    "raw",
    "masters",
    "metadata",
    "intermediates",
    "confidence",
    "validation",
    "logs",
)


@dataclass(frozen=True)
class RunLayout:
    """Paths for the PRD_v3 exact output tree; existing runs are never reused."""

    root: Path

    @classmethod
    def create(cls, output_dir: str | Path, *, timestamp: datetime | None = None) -> RunLayout:
        """Create an empty run tree, rejecting same-second collisions explicitly."""
        instant = datetime.now() if timestamp is None else timestamp
        root = Path(output_dir) / instant.strftime("run_%Y%m%d_%H%M%S")
        root.mkdir(parents=True, exist_ok=False)
        for name in RUN_DIRECTORIES:
            (root / name).mkdir()
        return cls(root)

    def directory(self, name: str) -> Path:
        """Return a named output directory without permitting arbitrary paths."""
        if name not in RUN_DIRECTORIES:
            raise ValueError(f"Unknown run directory: {name}")
        return self.root / name


@dataclass(frozen=True)
class RandomStreams:
    """Separate scene, exposure noise and fixed sensor-body PRD_v3 generators."""

    scene: np.random.Generator
    sensor_noise: np.random.Generator
    sensor_body: np.random.Generator

    @classmethod
    def from_seeds(
        cls, scene_seed: int = 42, sensor_noise_seed: int = 42001, sensor_body_seed: int = 42002
    ) -> RandomStreams:
        """Initialize local generators without changing NumPy's global RNG state."""
        seeds = (scene_seed, sensor_noise_seed, sensor_body_seed)
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Seeds must be nonnegative integers")
        return cls(*(np.random.default_rng(seed) for seed in seeds))
