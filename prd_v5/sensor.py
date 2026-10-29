"""Bounded MVP Bayer sampling and temporal electron-domain sensor simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prd_v5.hypothesis import CameraHypothesis


@dataclass(frozen=True)
class BayerSensor:
    """Apply one shared hypothesis to clean camera RGB at final sensor geometry.

    Unity clean signal represents base-ISO full well. Shot noise precedes read
    noise, analogue gain, rounding and clipping. No full-sensor FPN/PRNU or
    optical effects are implied. CFA origin is the top-left of the supplied
    full sensor array; crops must be taken after this stage.
    """

    camera: CameraHypothesis = CameraHypothesis()

    def process(self, source: np.ndarray, destination: np.memmap, *, strip_rows: int = 64) -> dict:
        """Write uint16 ADC codes with row-bounded working memory.

        Row-indexed independent shot/read streams preserve draws across strip
        sizes and when noise switches change. Equal final sensor coordinates
        share random streams across source-resolution runs, but differing
        Poisson means need not yield identical residuals. Record NumPy version
        for reproducibility. Caller must discard partial output after failure.
        """
        if type(strip_rows) is not int or strip_rows < 1:
            raise ValueError("strip_rows must be a positive integer")
        if source.ndim != 3 or source.shape[-1] != 3 or min(source.shape) < 1 or source.dtype.kind != "f":
            raise ValueError("source must be nonempty floating point HxWx3 camera RGB")
        if not isinstance(destination, np.memmap) or destination.dtype != np.uint16:
            raise ValueError("destination must be a uint16 disk-backed memmap")
        if destination.shape != source.shape[:2] or not destination.flags.writeable:
            raise ValueError("destination must match sensor dimensions and be writable")
        if np.may_share_memory(source, destination) or (
            isinstance(source, np.memmap) and source.filename == destination.filename
        ):
            raise ValueError("source and destination must use separate nonoverlapping buffers")
        camera = self.camera
        tile = np.array(["RGB".index(c) for c in camera.cfa_pattern]).reshape(2, 2)
        shot_seed = camera.stream_seed("sensor_shot")
        read_seed = camera.stream_seed("sensor_read")
        low = high = 0
        for start in range(0, source.shape[0], strip_rows):
            block = source[start : start + strip_rows]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError("source must contain finite nonnegative camera samples")
            for offset, row in enumerate(block):
                y = start + offset
                # Select CFA samples before electrons or any stochastic operation.
                mosaic = np.empty(source.shape[1], dtype=np.float64)
                for parity in (0, 1):
                    mosaic[parity::2] = row[parity::2, tile[y % 2, parity]]
                electrons = mosaic * camera.full_well_e
                if not np.isfinite(electrons).all() or np.any(electrons > 1e15):
                    raise ValueError("electron expectation exceeds supported numerical range")
                if camera.shot_noise:
                    electrons = np.random.default_rng(np.random.SeedSequence([shot_seed, y])).poisson(electrons)
                if camera.read_noise and camera.read_noise_e > 0:
                    read = np.random.default_rng(np.random.SeedSequence([read_seed, y]))
                    electrons = electrons + read.normal(0, camera.read_noise_e, size=len(mosaic))
                codes = np.rint(electrons * camera.analogue_gain / camera.conversion_gain_e_per_dn + camera.black_level)
                low += int(np.count_nonzero(codes < 0))
                high += int(np.count_nonzero(codes > camera.white_level))
                destination[y] = np.clip(codes, 0, camera.white_level).astype(np.uint16)
        destination.flush()
        return {
            "method": "CFA_then_electrons_shot_read_gain_ADC",
            "camera_registry": camera.registry(),
            "cfa_origin": [0, 0],
            "shot_noise_effective": camera.shot_noise,
            "read_noise_effective": camera.read_noise and camera.read_noise_e > 0,
            "read_noise_zero_effect": camera.read_noise and camera.read_noise_e == 0,
            "codes_clipped_low": low,
            "codes_clipped_high": high,
            "rng": "NumPy default_rng; independent stage and absolute sensor row streams",
            "numpy_version": np.__version__,
            "strip_rows": strip_rows,
            "validation": "Synthetic temporal model only; physical camera calibration not measured",
        }
