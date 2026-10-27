"""PRD_v3 automated validation sequence applied to a finished pipeline run folder.

``ValidationRunner`` reads only the run tree written by ``run_pipeline`` (or the notebook),
applies every Validation-milestone validator and writes ``validation/validation_results.json``.
Arrays are memory-mapped and scanned in strips; developed renders are downsampled at once.
A failed hard gate makes the run status FAILED; QA targets and report-only rows never do.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import DNGFieldRecord
from synthraw.validation import (
    ADCContractValidator,
    CFAValidator,
    ColourValidator,
    DemosaicValidator,
    EditabilityValidator,
    HDRValidator,
    MetadataDependencyValidator,
    NoiseFit,
    NoiseValidator,
    NumericHealthValidator,
    RawTherapeeRenderer,
    RoundTripResult,
    RoundTripValidator,
    SeamValidator,
    SensorSeedValidator,
    ValidationCheck,
    ValidationThresholds,
    hard_failures,
    libraw_reopen_check,
)

RESULTS_NAME = "validation_results.json"
# Pipeline products compared by the optional two-run determinism check.
DETERMINISM_ARRAYS = ("raw/sensor_mosaic_uint16.npy", "intermediates/03_scene_linear.npy")
# Intermediates scanned for NaN/Inf, with physical lower/upper bounds where they exist. Noisy
# electrons keep negative read-noise excursions for the ADC pedestal, so they are finite-only.
HEALTH_ARRAYS: dict[str, tuple[float | None, float | None]] = {
    "03_scene_linear.npy": (None, None),
    "04_hdr_radiance.npy": (0.0, None),
    "05_camera_native_rgb.npy": (None, None),
    "06_optical_sensor_rgb.npy": (None, None),
    "07_expected_electrons.npy": (0.0, None),
    "08_noisy_electrons.npy": (None, None),
    "09_clean_cfa.npy": (0.0, None),
    "10_noisy_cfa.npy": (None, None),
}


@dataclass
class ValidationRun:
    """All rows of one validation pass plus the in-memory products the figures need."""

    root: Path
    checks: list[ValidationCheck]
    noise_fit: NoiseFit | None = None
    roundtrip: RoundTripResult | None = None
    demosaic_thumbs: dict[str, np.ndarray] = field(default_factory=dict)
    editability_thumbs: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "FAILED" if hard_failures(self.checks) else "PASSED"

    def to_dict(self) -> dict[str, Any]:
        failed_targets = [c.name for c in self.checks if not c.hard and c.passed is False]
        return {
            "status": self.status,
            "hard_failures": [check.name for check in hard_failures(self.checks)],
            "qa_targets_missed": failed_targets,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
        }


def load_dng_records(path: str | Path) -> dict[str, DNGFieldRecord]:
    """Rebuild DNG field records from metadata/dng_metadata.json."""
    return {item["field"]: DNGFieldRecord(**item) for item in json.loads(Path(path).read_text())}


def _group_value(groups: Mapping[str, Any], group: str, name: str) -> Any:
    for record in groups.get(group, []):
        if record["field"] == name:
            return record["value"]
    raise KeyError(f"{group}.{name} missing from metadata.json")


def _open_tiff(path: Path) -> np.ndarray:
    import tifffile

    try:
        return tifffile.memmap(path, mode="r")
    except ValueError:
        return tifffile.imread(path)


class ValidationRunner:
    """Run the PRD automated validation sequence on a run folder and write its JSON."""

    def __init__(
        self,
        thresholds: ValidationThresholds | None = None,
        *,
        noise_realisations: int = 32,
        noise_patch: int = 32,
        camera_samples: int = 4096,
        roundtrip_max_side: int = 2048,
        thumbnail_side: int = 512,
        rawtherapee: RawTherapeeRenderer | None = None,
    ) -> None:
        self.thresholds = thresholds or ValidationThresholds()
        self.noise_realisations = noise_realisations
        self.noise_patch = noise_patch
        self.camera_samples = camera_samples
        self.roundtrip_max_side = roundtrip_max_side
        self.thumbnail_side = thumbnail_side
        self.rawtherapee = rawtherapee or RawTherapeeRenderer()

    def run(
        self,
        root: str | Path,
        *,
        reference_run: str | Path | None = None,
        strip_rows: int = 128,
        log: Callable[[str], None] | None = None,
    ) -> ValidationRun:
        """Validate ``root``; ``reference_run`` is a second same-seed run for pipeline determinism."""
        root = Path(root)
        say = log or (lambda message: None)
        config = ExperimentConfig.from_dict(json.loads((root / "metadata" / "config.json").read_text()))
        groups = json.loads((root / "metadata" / "metadata.json").read_text())["groups"]
        records = load_dng_records(root / "metadata" / "dng_metadata.json")
        dng = root / "raw" / "synthetic_50mp.dng"
        inter = root / "intermediates"

        def load(path: Path) -> np.ndarray:
            return np.load(path, mmap_mode="r")

        mosaic = load(root / "raw" / "sensor_mosaic_uint16.npy")
        phase = _group_value(groups, "sensor_profile", "applied_cfa_phase")
        black = float(_group_value(groups, "sensor_profile", "applied_adc_black_level"))
        white = int(_group_value(groups, "sensor_profile", "applied_adc_white_level"))
        bits = int(_group_value(groups, "sensor_profile", "applied_adc_adc_bits"))
        thresholds = self.thresholds
        run = ValidationRun(root, [])
        checks = run.checks

        say("numeric health, CFA and ADC contracts")
        checks.append(
            NumericHealthValidator().validate(
                {name: load(inter / name) for name in HEALTH_ARRAYS if (inter / name).is_file()},
                ranges=HEALTH_ARRAYS,
                strip_rows=strip_rows,
            )
        )
        checks.append(
            CFAValidator().validate(
                load(inter / "09_clean_cfa.npy"),
                load(inter / "07_expected_electrons.npy"),
                phase,
                records,
                strip_rows=strip_rows,
            )
        )
        noisy_cfa = CFAValidator().validate(
            load(inter / "10_noisy_cfa.npy"), load(inter / "08_noisy_electrons.npy"), phase, strip_rows=strip_rows
        )
        noisy_cfa.name = "cfa_phase_noisy"
        checks.append(noisy_cfa)
        checks.append(
            ADCContractValidator().validate(
                mosaic, black_level=black, white_level=white, adc_bits=bits, strip_rows=strip_rows
            )
        )

        say("DNG reopen, metadata dependencies and colour profile")
        checks.extend(libraw_reopen_check(dng_path=dng, mosaic=mosaic, records=records))
        checks.append(MetadataDependencyValidator().validate(groups))
        camera = load(inter / "05_camera_native_rgb.npy")
        step = max(1, math.ceil(math.sqrt(camera.shape[0] * camera.shape[1] / self.camera_samples)))
        samples = np.asarray(camera[::step, ::step], dtype=np.float64).reshape(-1, 3)
        checks.extend(ColourValidator().validate(records, samples))

        say(f"flat-field noise test ({self.noise_realisations} realisations per level)")
        fit, profile = NoiseValidator(thresholds).measure(
            config,
            full_well_electrons=float(_group_value(groups, "sensor_profile", "full_well_electrons")),
            conversion_gain_e_per_dn=float(
                _group_value(groups, "sensor_profile", "applied_adc_conversion_gain_e_per_dn")
            ),
            read_noise_e=float(_group_value(groups, "sensor_profile", "applied_adc_read_noise_e")),
            analogue_gain=float(_group_value(groups, "sensor_profile", "applied_adc_analogue_gain")),
            black_level=black,
            white_level=white,
            realisations=self.noise_realisations,
            patch=self.noise_patch,
            phase=phase,
        )
        run.noise_fit = fit
        dng_profile = records["NoiseProfile"].value if "NoiseProfile" in records else None
        noise_checks = NoiseValidator(thresholds).validate(fit, dng_profile)
        if dng_profile is not None and profile is not None:
            # The flats must reproduce the run's own ADC noise model, not a different one.
            agree = bool(np.allclose(np.ravel(profile), np.ravel(dng_profile), rtol=1e-9, atol=0))
            noise_checks[0].measured["flat_profile_matches_dng"] = agree
            noise_checks[0].passed = bool(noise_checks[0].passed and agree)
            if not agree:
                noise_checks[0].detail = "flat-field ADC NoiseProfile differs from the DNG NoiseProfile"
        checks.extend(noise_checks)

        say("demosaic and editability renders")
        demosaic_checks, run.demosaic_thumbs = DemosaicValidator().develop(dng, thumbnail_side=self.thumbnail_side)
        checks.extend(demosaic_checks)
        edit_checks, run.editability_thumbs = EditabilityValidator(thresholds).validate(
            dng, thumbnail_side=self.thumbnail_side // 2
        )
        checks.extend(edit_checks)

        say("round trip, SR seams and HDR")
        tiff = root / "masters" / "developed_50mp_16bit.tif"
        if tiff.is_file():
            confidence = root / "confidence"
            supported = np.asarray(load(confidence / "sr_source_supported_mask.npy")) > 0.5
            supported &= np.asarray(load(confidence / "source_supported_mask.npy"))
            transfer = _group_value(groups, "colour_space", "input_colour_space")["transfer"]
            run.roundtrip = RoundTripValidator(thresholds).validate(
                load(inter / "02_sr_rgb.npy"),
                _open_tiff(tiff),
                supported,
                transfer=transfer,
                max_side=self.roundtrip_max_side,
            )
            checks.extend(run.roundtrip.checks)
        else:
            checks.append(
                ValidationCheck(
                    "roundtrip_supported_ssim",
                    "Round trip",
                    None,
                    detail="developed 16-bit TIFF not written (WRITE_TIFF16 off); round trip not measured",
                )
            )
        sr_report = json.loads((root / "metadata" / "sr_reconstruction.json").read_text())
        hdr_report = json.loads((root / "metadata" / "hdr_reconstruction.json").read_text())
        checks.append(SeamValidator(thresholds).validate(sr_report))
        checks.extend(HDRValidator().validate(load(inter / "04_hdr_radiance.npy"), hdr_report, strip_rows=strip_rows))

        say("determinism and seed checks")
        checks.extend(SensorSeedValidator().validate(config))
        if reference_run is not None:
            other = Path(reference_run)
            checks.append(
                SensorSeedValidator.compare_runs(
                    {name: load(root / name) for name in DETERMINISM_ARRAYS},
                    {name: load(other / name) for name in DETERMINISM_ARRAYS if (other / name).is_file()},
                )
            )
        checks.append(self.rawtherapee.render(dng, root / "validation" / "rawtherapee"))

        run.summary = {
            "dng": str(dng.relative_to(root)),
            "mosaic_shape": list(mosaic.shape),
            "cfa_phase": phase,
            "generated_hdr_area_percentage": hdr_report.get("generated_hdr_area_percentage"),
            "source_supported_fraction": (
                run.roundtrip.checks[0].measured.get("supported_fraction") if run.roundtrip else None
            ),
            "thresholds": asdict(thresholds),
        }
        path = root / "validation" / RESULTS_NAME
        path.write_text(json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n")
        say(f"validation {run.status}: {path.relative_to(root)}")
        return run
