"""Bounded MVP Bayer sampling and temporal electron-domain sensor simulation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from prd_v5.hypothesis import CameraField, CameraHypothesis


@dataclass(frozen=True)
class SensorEffects:
    """Uncalibrated CFA-body priors; zero amplitudes preserve the MVP exactly.

    Dark current doubles every temperature_doubling_c relative to 20 C.
    Body maps use camera seed and absolute sensor coordinates, never source size.
    """

    nonlinearity_compression: float = 0.0
    prnu_sigma: float = 0.0
    dsnu_e: float = 0.0
    row_fpn_e: float = 0.0
    column_fpn_e: float = 0.0
    dark_current_e_per_s: float = 0.0
    temperature_c: float = 20.0
    temperature_doubling_c: float = 6.0
    hot_pixel_fraction: float = 0.0
    dead_pixel_fraction: float = 0.0
    hot_pixel_e_per_s: float = 1000.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "temperature_c" and (value < 0 or value > 1e15):
                raise ValueError(f"{name} must be nonnegative and <= 1e15")
        if not -100 <= self.temperature_c <= 100 or self.temperature_doubling_c < 1:
            raise ValueError("Temperature must be -100..100 C and doubling interval >= 1 C")
        if not 0 <= self.nonlinearity_compression <= 1:
            raise ValueError("nonlinearity_compression must be in [0, 1]")
        if max(self.hot_pixel_fraction, self.dead_pixel_fraction) > 1 or self.prnu_sigma > 1:
            raise ValueError("Defect fractions and PRNU sigma must be <= 1")
        if self.dark_rate > 1e15 or self.hot_pixel_e_per_s > 1e15:
            raise ValueError("Dark/hot rates exceed supported numerical range")

    @property
    def dark_rate(self):
        return self.dark_current_e_per_s * 2 ** ((self.temperature_c - 20) / self.temperature_doubling_c)

    def registry(self):
        return {
            f"sensor.{name}": asdict(
                CameraField(
                    f"sensor.{name}",
                    value,
                    "C"
                    if name in ("temperature_c", "temperature_doubling_c")
                    else "electrons/s"
                    if name.endswith("per_s")
                    else "electrons RMS"
                    if name.endswith("_e")
                    else "ratio",
                    "AUTO",
                    "SYNTHETIC",
                    None,
                    None,
                    "Explicit uncalibrated sensor-body prior; shared camera seed",
                    "Parameter-domain checks only; physical calibration and full-scale validation pending",
                )
            )
            for name, value in asdict(self).items()
        }

    def states(self):
        return {
            name: {"effective": value > 0, "status": "experimental synthetic prior" if value > 0 else "zero-effect"}
            for name, value in asdict(self).items()
            if name not in ("temperature_c", "temperature_doubling_c", "hot_pixel_e_per_s")
        }


@dataclass(frozen=True)
class BayerSensor:
    """Apply one shared hypothesis to clean camera RGB at final sensor geometry.

    Unity clean signal represents ADC saturation: full well at base ISO and
    full well / analogue gain above it. Shot noise precedes read
    noise, analogue gain, rounding and clipping. Optional body effects are synthetic priors, not camera calibration.
    No optical effects are implied. CFA origin is the top-left of the supplied
    full sensor array; crops must be taken after this stage.
    """

    camera: CameraHypothesis = CameraHypothesis()
    effects: SensorEffects = SensorEffects()

    def process(
        self,
        source: np.ndarray,
        destination: np.memmap,
        *,
        strip_rows: int = 64,
        defect_mask: np.memmap | None = None,
    ) -> dict:
        """Write uint16 ADC codes with row-bounded working memory.

        Row-indexed independent shot/read streams preserve draws across strip
        sizes and when noise switches change. Equal final sensor coordinates
        share random streams across source-resolution runs, but differing
        Poisson means need not yield identical residuals. Record NumPy version
        for reproducibility. Optional defect_mask receives uint8 hot/dead flags
        in final sensor coordinates, using the same draws as the ADC simulation.
        Caller must discard all partial output buffers after failure.
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
        if defect_mask is not None:
            if (
                not isinstance(defect_mask, np.memmap)
                or defect_mask.dtype != np.uint8
                or defect_mask.shape != source.shape[:2]
                or not defect_mask.flags.writeable
            ):
                raise ValueError("defect_mask must be a writable uint8 disk-backed sensor-sized memmap")
            for buffer in (source, destination):
                if np.may_share_memory(buffer, defect_mask) or (
                    isinstance(buffer, np.memmap) and buffer.filename == defect_mask.filename
                ):
                    raise ValueError("defect_mask must use a separate nonoverlapping buffer")
        camera = self.camera
        effects = self.effects

        def body(component, row=0):
            return np.random.default_rng(np.random.SeedSequence([camera.stream_seed("sensor_body"), component, row]))

        width = source.shape[1]
        columns = body(4).normal(0, effects.column_fpn_e, width) if effects.column_fpn_e else None
        dark_mean = effects.dark_rate * camera.shutter_s
        if not math.isfinite(dark_mean) or dark_mean > 1e15:
            raise ValueError("Dark expectation exceeds supported numerical range")
        tile = np.array(["RGB".index(c) for c in camera.cfa_pattern]).reshape(2, 2)
        shot_seed = camera.stream_seed("sensor_shot")
        read_seed = camera.stream_seed("sensor_read")
        low = high = 0
        hot_sites = dead_sites = overlapping_sites = 0
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
                electrons = mosaic * camera.full_well_e / camera.analogue_gain
                if not np.isfinite(electrons).all() or np.any(electrons > 1e15):
                    raise ValueError("electron expectation exceeds supported numerical range")
                if effects.prnu_sigma:
                    electrons *= np.maximum(body(0, y).normal(1, effects.prnu_sigma, width), 0)
                if effects.dead_pixel_fraction:
                    dead = body(6, y).random(width) < effects.dead_pixel_fraction
                    dead_sites += int(np.count_nonzero(dead))
                    electrons[dead] = 0
                else:
                    dead = None
                hot = None
                if effects.hot_pixel_fraction:
                    hot = body(5, y).random(width) < effects.hot_pixel_fraction
                    hot_sites += int(np.count_nonzero(hot))
                    if dead is not None:
                        overlapping_sites += int(np.count_nonzero(hot & dead))
                    electrons[hot] += effects.hot_pixel_e_per_s * camera.shutter_s
                if defect_mask is not None:
                    # Bit flags retain both selections even when dead overrides hot.
                    defect_mask[y] = 0
                    if hot is not None:
                        defect_mask[y, hot] = 1
                    if dead is not None:
                        defect_mask[y, dead] |= 2
                electrons += dark_mean
                if dead is not None:
                    electrons[dead] = 0
                if not np.isfinite(electrons).all() or np.any(electrons > 1e15):
                    raise ValueError("Sensor expectation exceeds supported numerical range")
                if camera.shot_noise:
                    electrons = np.random.default_rng(np.random.SeedSequence([shot_seed, y])).poisson(electrons)
                if effects.nonlinearity_compression:
                    # Monotone charge-to-signal compression, before read electronics.
                    electrons = electrons / (1 + effects.nonlinearity_compression * electrons / camera.full_well_e)
                if camera.read_noise and camera.read_noise_e > 0:
                    read = np.random.default_rng(np.random.SeedSequence([read_seed, y]))
                    electrons = electrons + read.normal(0, camera.read_noise_e, size=len(mosaic))
                if effects.dsnu_e:
                    electrons = electrons + body(1, y).normal(0, effects.dsnu_e, width)
                if effects.row_fpn_e:
                    electrons = electrons + body(2, y).normal(0, effects.row_fpn_e)
                if columns is not None:
                    electrons = electrons + columns
                codes = np.rint(electrons * camera.analogue_gain / camera.conversion_gain_e_per_dn + camera.black_level)
                low += int(np.count_nonzero(codes < 0))
                high += int(np.count_nonzero(codes > camera.white_level))
                destination[y] = np.clip(codes, 0, camera.white_level).astype(np.uint16)
        destination.flush()
        if defect_mask is not None:
            defect_mask.flush()
        return {
            "method": "CFA_then_electrons_shot_read_gain_ADC",
            "camera_registry": camera.registry(),
            "body_effects": {
                "parameters": asdict(effects),
                "states": effects.states(),
                "dark_expectation_e": dark_mean,
                "defect_occupancy": {
                    "photosites": int(source.shape[0] * width),
                    "hot_selected": hot_sites,
                    "dead_selected": dead_sites,
                    "hot_and_dead": overlapping_sites,
                    "hot_surviving_dead_override": hot_sites - overlapping_sites,
                    "semantics": (
                        "Realized synthetic selections, not detected defects or necessarily changed ADC codes; "
                        "dead overrides hot charge, read/offset electronics remain"
                    ),
                },
                "defect_mask": {
                    "written": defect_mask is not None,
                    "encoding": "uint8 flags: bit 0 hot selected, bit 1 dead selected; 3 means overlap",
                    "origin": [0, 0],
                    "semantics": "Synthetic selections, not detected defects or proof of changed ADC codes",
                },
                "nonlinearity_model": "q_out = q / (1 + k*q/full_well_e); after shot, before read; k in [0,1]",
                "body_seed": camera.stream_seed("sensor_body"),
                "order": "CFA, PRNU, defects and dark expectations, shot, charge compression, read, DSNU/FPN, ADC",
                "noise_profile_limit": (
                    "DNG NoiseProfile and LinearResponseLimit describe the linear baseline only; "
                    "they exclude body maps, dark current and nonlinear charge compression"
                ),
                "validation": "Experimental uncalibrated priors; dead sites retain read/offset electronics",
            },
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
