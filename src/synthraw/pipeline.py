"""End-to-end orchestration for the PRD_v3 Integration milestone and exact output tree.

Each stage reads its inputs from disk checkpoints, writes its products into the run
layout and drops its array references before the next stage starts. Only small
records and paths survive between stages.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from synthraw.config import ExperimentConfig
from synthraw.environment import EnvironmentInspector, EnvironmentReport
from synthraw.geometry import TargetGeometryPlanner
from synthraw.libraw_validator import HardValidationError
from synthraw.provenance import FieldRecord, Provenance, ProvenanceWriter, write_c2pa
from synthraw.run_layout import RunLayout

T = TypeVar("T")

SOFTWARE_VERSION = "0.1.0"
SOURCE_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png"}
STAGES = (
    "input_audit",
    "geometry",
    "super_resolution",
    "hdr",
    "camera_exposure",
    "inverse_isp",
    "optics",
    "electrons_noise",
    "cfa",
    "adc",
    "metadata",
    "dng_export",
    "libraw_reopen",
    "raw_development_tiff",
    "exr_preview",
    "provenance",
)


@dataclass(frozen=True)
class PipelineResult:
    """Published products of a successful run; absent on any hard failure."""

    root: Path
    dng_path: Path
    tiff_path: Path | None
    exr_path: Path | None
    preview_path: Path
    stages: tuple[str, ...]
    fallbacks: tuple[str, ...] = field(default_factory=tuple)


def is_out_of_memory(error: BaseException) -> bool:
    """Recognize host and CUDA OOM without importing torch."""
    return isinstance(error, MemoryError) or (isinstance(error, RuntimeError) and "out of memory" in str(error).lower())


def run_with_oom_fallback(
    operation: Callable[[int, bool], T],
    *,
    tile: int,
    generative_enabled: bool,
    log: Callable[[str], None] = lambda message: None,
) -> T:
    """PRD failure hierarchy: on OOM halve the tile, retry once, then disable generative backends.

    ``operation(tile, disable_generative)`` must clean up its own partial outputs.
    Non-OOM failures and OOM without an optional generative backend propagate.
    """
    try:
        return operation(tile, False)
    except (MemoryError, RuntimeError) as error:
        if not is_out_of_memory(error):
            raise
        log(f"OOM at tile {tile}; retrying once at {max(1, tile // 2)}: {error}")
    tile = max(1, tile // 2)
    try:
        return operation(tile, False)
    except (MemoryError, RuntimeError) as error:
        if not is_out_of_memory(error) or not generative_enabled:
            raise
        log(f"OOM persisted at tile {tile}; disabling optional generative backend: {error}")
    return operation(tile, True)


def release_memory() -> None:
    """Drop unreachable arrays and cached accelerator blocks between stages."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    text = json.dumps(payload, indent=2, allow_nan=False, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _new_memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


class PipelineRun:
    """Mutable run state shared by ``run_pipeline`` and the notebook: paths, records and small scalars only.

    Each notebook section calls ``run_stage`` for its stage, so both drivers execute one definition of the chain.
    """

    def __init__(self, source: Path, config: ExperimentConfig, layout: RunLayout, options: dict[str, Any]) -> None:
        self.source = source
        self.config = config
        self.run = layout
        self.options = options
        self.fallbacks: list[str] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.outputs: dict[str, list[str]] = {}
        self.groups: dict[str, list[FieldRecord]] = {}
        self.logger = open_execution_log(layout, f"synthraw.pipeline.{layout.root.name}.{id(self)}")

    @classmethod
    def create(
        cls,
        source_path: str | Path,
        config: ExperimentConfig,
        output_root: str | Path,
        *,
        environment: EnvironmentReport | None = None,
        timestamp: datetime | None = None,
        fast_dev_max_side: int = 512,
        accept_licence: bool = False,
        scene_ev100: float = 10.0,
        manual_exposure: tuple[float, float, int] | None = None,
        manual_wb: dict[str, float] | None = None,
        manual_fov_degrees: float | None = None,
        strip_rows: int = 128,
        c2pa_signer: Any = None,
    ) -> PipelineRun:
        """Validate the request, create a new run_YYYYMMDD_HHMMSS tree and write its environment logs.

        ``options`` keeps the keyword settings under their ``run_pipeline`` names, so a second run
        can reuse them unchanged.
        """
        if not config.WRITE_DNG:
            raise ValueError("run_pipeline requires WRITE_DNG: the DNG is the primary deliverable")
        if type(strip_rows) is not int or strip_rows < 2:
            raise ValueError("strip_rows must be an integer of at least 2")
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        layout = RunLayout.create(output_root, timestamp=timestamp)
        options = {
            "fast_dev_max_side": fast_dev_max_side,
            "accept_licence": accept_licence,
            "scene_ev100": scene_ev100,
            "manual_exposure": manual_exposure,
            "manual_wb": manual_wb,
            "manual_fov_degrees": manual_fov_degrees,
            "strip_rows": strip_rows,
            "c2pa_signer": c2pa_signer,
        }
        state = cls(source, config, layout, options)
        try:
            state.environment = environment or EnvironmentInspector().inspect(layout.root)
            write_run_logs(layout, state.environment)
        except BaseException as error:
            state.logger.error(f"pipeline FAILED: {type(error).__name__}: {error}")
            state.close()
            raise
        state.log(f"synthraw {SOFTWARE_VERSION}; source {source.name}; geometry {config.OUTPUT_GEOMETRY}")
        return state

    @property
    def strip_rows(self) -> int:
        return self.options["strip_rows"]

    def path(self, directory: str, name: str) -> Path:
        return self.run.directory(directory) / name

    def log(self, message: str) -> None:
        self.logger.info(message)

    def close(self) -> None:
        _close_handlers(self.logger)

    def _files(self) -> set[Path]:
        return {path for path in self.run.root.rglob("*") if path.is_file()}

    def run_stage(self, name: str) -> None:
        """Run, checkpoint and free one ``STAGES`` entry in chain order; a failure stops the whole chain.

        The files each stage creates are recorded. Running a stage again (for example after editing a
        notebook hypothesis) moves its previous files aside and replaces them only when the stage succeeds;
        a failed attempt removes what it wrote and restores them. Later stages are marked stale (their
        checkpoints and fallbacks are dropped, so they must run again before a run report) but keep their
        files until they are re-run. Files this run did not create are never moved, so stage writers still
        refuse to overwrite them.
        """
        if name not in STAGES:
            raise ValueError(f"Unknown pipeline stage: {name}")
        index = STAGES.index(name)
        completed = [checkpoint["stage"] for checkpoint in self.checkpoints]
        if completed[:index] != list(STAGES[:index]):
            raise RuntimeError(f"Stage {name} needs {list(STAGES[:index])} completed first; completed {completed}")
        del self.checkpoints[index:]
        self.fallbacks = [message for checkpoint in self.checkpoints for message in checkpoint["fallbacks"]]
        recorded = len(self.fallbacks)
        root = self.run.root
        aside = root / f".rerun_{name}"
        previous = [relative for relative in self.outputs.pop(name, []) if (root / relative).is_file()]
        for relative in previous:
            (aside / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).replace(aside / relative)
        existing = self._files()
        self.log(f"stage {name}: start")
        started = time.perf_counter()
        try:
            getattr(self, name)()
        except BaseException as error:
            self.logger.error(f"stage {name}: FAILED: {type(error).__name__}: {error}")
            for path in self._files() - existing:
                path.unlink(missing_ok=True)
            for relative in previous:
                (aside / relative).replace(root / relative)
            self.outputs[name] = previous
            raise
        finally:
            shutil.rmtree(aside, ignore_errors=True)
            release_memory()
        seconds = time.perf_counter() - started
        self.outputs[name] = sorted(str(path.relative_to(root)) for path in self._files() - existing)
        self.checkpoints.append({"stage": name, "seconds": seconds, "fallbacks": self.fallbacks[recorded:]})
        _write_json(self.path("logs", "checkpoints.json"), {"completed": self.checkpoints})
        self.log(f"stage {name}: done in {seconds:.3f} s")

    def write_report(
        self, *, validation: Any = None, comparison: Any = None, warnings: Iterable[str] = ()
    ) -> dict[str, Any]:
        """Write validation/run_report.json for a completed chain; a failed validation hard gate is FAILED.

        No report is written unless every stage completed, so a hard failure never leaves a success report.
        ``warnings`` are recorded without changing the status.
        """
        completed = [checkpoint["stage"] for checkpoint in self.checkpoints]
        if completed != list(STAGES):
            raise RuntimeError(f"Run report requires every pipeline stage; completed {completed}")
        validation_status = "NOT RUN" if validation is None else validation.status
        report = {
            "status": "FAILED" if validation_status == "FAILED" else "SUCCESS",
            "validation": validation_status,
            "hard_failures": [] if validation is None else validation.to_dict()["hard_failures"],
            "comparison_2k_vs_4k": (
                None if comparison is None else {"status": comparison.status, "flags": comparison.flags}
            ),
            "stages": completed,
            "fallbacks": self.fallbacks,
            "warnings": list(warnings),
            "dng": "raw/synthetic_50mp.dng",
            "libraw_reopened": True,
        }
        _write_json(self.path("validation", "run_report.json"), report)
        self.log(f"run report {report['status']}; validation {validation_status}")
        return report

    # Stages follow PRD notebook order; each reloads disk checkpoints read-only.

    def input_audit(self) -> None:
        from PIL import Image

        from synthraw.input_audit import InputInspector

        self.audit = InputInspector().inspect(self.source)
        original = publish_source(self.run, self.source, self.audit)
        self.source_original = original
        _write_json(self.path("metadata", "config.json"), self.config.to_dict())
        _write_json(self.path("metadata", "input_audit.json"), self.audit.to_dict())
        with Image.open(original) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if rgb.shape != (self.audit.height, self.audit.width, 3):
            raise ValueError("Decoded source dimensions differ from the input audit")
        with self.path("intermediates", "01_input_rgb.npy").open("xb") as stream:
            np.save(stream, rgb, allow_pickle=False)

    def geometry(self) -> None:
        self.geometry_plan = TargetGeometryPlanner().plan(
            self.audit.width, self.audit.height, self.config, fast_dev_max_side=self.options["fast_dev_max_side"]
        )
        _write_json(self.path("metadata", "geometry_plan.json"), asdict(self.geometry_plan))
        self.log(f"geometry {self.geometry_plan.width} x {self.geometry_plan.height}")

    def super_resolution(self) -> None:
        from synthraw.super_resolution import SuperResolutionEngine

        source = np.load(self.path("intermediates", "01_input_rgb.npy"), mmap_mode="r")
        self.sr_result = SuperResolutionEngine().run(
            source,
            self.geometry_plan,
            self.config,
            self.environment,
            self.run.root,
            accept_licence=self.options["accept_licence"],
        )
        tiled = self.sr_result.tiled
        self.fallbacks.extend(f"SR {reason}" for reason in tiled.fallback_reasons + tiled.retry_reasons)
        _write_json(self.path("metadata", "sr_reconstruction.json"), asdict(self.sr_result))
        self.groups["sr_canvas"] = [self.sr_result.canvas.provenance]
        self.log(f"SR backend {tiled.backend}; fallbacks {list(tiled.fallback_reasons)}")

    def hdr(self) -> None:
        from synthraw.colour_space import resolve_colour_space
        from synthraw.hdr import HDRReconstructionEngine

        colour = resolve_colour_space(self.audit)
        self.groups["colour_space"] = colour.records()
        engine = HDRReconstructionEngine()
        encoded = np.load(self.sr_result.canvas.path, mmap_mode="r")
        linear = engine.linearize(
            encoded, self.path("intermediates", "03_scene_linear.npy"), colour_space=colour, undo_tone=True
        )
        enabled = self.config.HDR_ENABLED
        backend = self.config.HDR_BACKEND if enabled else "DETERMINISTIC"

        def expand(strip_rows: int, disable_generative: bool) -> dict:
            # Stage, then split the radiance and the masks into their PRD folders.
            with tempfile.TemporaryDirectory(prefix="hdr_", dir=self.run.directory("intermediates")) as staging:
                result = engine.expand(
                    encoded,
                    linear,
                    staging,
                    highlight_ev=self.config.HDR_HIGHLIGHT_EV if enabled else 0.0,
                    shadow_ev=self.config.HDR_SHADOW_EV if enabled else 0.0,
                    backend="DETERMINISTIC" if disable_generative else backend,
                    strip_rows=strip_rows,
                    negative_policy="CLIP",
                )
                for name, array in result.pop("arrays").items():
                    staged = Path(array.filename)
                    target = self.path("intermediates" if name == "04_hdr_radiance" else "confidence", staged.name)
                    staged.replace(target)
            if disable_generative:
                result["fallback_reason"] = "Optional generative HDR backend disabled after repeated OOM"
            return result

        report = run_with_oom_fallback(
            expand, tile=self.strip_rows, generative_enabled=backend != "DETERMINISTIC", log=self.fallback
        )
        if report["fallback_reason"]:
            self.fallbacks.append(f"HDR {report['fallback_reason']}")
        self.hdr_report = report
        _write_json(self.path("metadata", "hdr_reconstruction.json"), report)

    def fallback(self, message: str) -> None:
        self.fallbacks.append(message)
        self.logger.warning(message)

    def camera_exposure(self) -> None:
        from synthraw.camera_geometry import CameraGeometryEstimator
        from synthraw.camera_profile import CameraProfileManager
        from synthraw.exposure import ExposurePosteriorEstimator
        from synthraw.white_balance import WhiteBalanceEstimator

        self.camera_profile = CameraProfileManager().build(
            config=self.config, geometry=self.geometry_plan, reference_raw=self.config.REFERENCE_RAW_PATH
        )
        radiance = np.load(self.path("intermediates", "04_hdr_radiance.npy"), mmap_mode="r")
        step = max(1, int(np.ceil(max(radiance.shape[:2]) / 256)))
        self.white_balance = WhiteBalanceEstimator().estimate(
            radiance[::step, ::step], self.camera_profile, **(self.options["manual_wb"] or {})
        )
        self.exposure = ExposurePosteriorEstimator().estimate(
            self.camera_profile,
            mode=self.config.EXPOSURE_MODE,
            scene_ev100=self.options["scene_ev100"],
            manual=self.options["manual_exposure"],
        )
        camera_geometry = CameraGeometryEstimator().estimate(
            self.camera_profile, manual_fov_degrees=self.options["manual_fov_degrees"]
        )
        if camera_geometry.fallback_reason:
            self.fallbacks.append(f"Camera geometry {camera_geometry.fallback_reason}")
        for name, records in (
            ("white_balance", self.white_balance.records),
            ("exposure_posterior", self.exposure.records),
            ("camera_geometry", camera_geometry.records),
        ):
            self.groups[name] = list(records.values())
            ProvenanceWriter().write(self.groups[name], self.path("metadata", f"{name}.json"))

    def inverse_isp(self) -> None:
        from synthraw.camera_colour import CameraColourModel, InverseISPEngine
        from synthraw.experimental import spectral_inference_records

        self.colour_model = CameraColourModel()
        engine = InverseISPEngine()
        radiance = np.load(self.path("intermediates", "04_hdr_radiance.npy"), mmap_mode="r")
        output = _new_memmap(self.path("intermediates", "05_camera_native_rgb.npy"), np.float32, radiance.shape)
        for start in range(0, radiance.shape[0], self.strip_rows):
            result = engine.reconstruct(
                radiance[start : start + self.strip_rows],
                self.colour_model,
                white_balance_gains=self.white_balance.gains,
                use_raw_flow=self.config.USE_RAW_FLOW,
                use_spiraldiff=self.config.USE_SPIRALDIFF,
            )
            output[start : start + self.strip_rows] = result.camera_rgb
        output.flush()
        self.inverse_isp_backend = result.records["inverse_isp_backend"].value
        self.groups["inverse_isp"] = list(result.records.values())
        spectral = spectral_inference_records(
            self.config.INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE, accept_licence=self.options["accept_licence"]
        )
        self.groups["colour_model"] = list(self.colour_model.metadata().values()) + spectral
        for label, record in (
            ("Inverse ISP", result.records["inverse_isp_fallback_reason"]),
            ("Spectral inference", spectral[-1]),
        ):
            if record.value:
                self.fallbacks.append(f"{label} {record.value}")
        ProvenanceWriter().write(self.groups["inverse_isp"], self.path("metadata", "inverse_isp.json"))

    def optics(self) -> None:
        from synthraw.optics import LensSimulator, OpticalBlurParameters

        camera_rgb = np.load(self.path("intermediates", "05_camera_native_rgb.npy"), mmap_mode="r")
        output = _new_memmap(
            self.path("intermediates", "06_optical_sensor_rgb.npy"), camera_rgb.dtype, camera_rgb.shape
        )
        output, records = LensSimulator().apply(
            camera_rgb,
            self.config,
            out=output,
            strip_rows=self.strip_rows,
            work_dir=self.run.directory("intermediates"),
            blur=OpticalBlurParameters(
                lens_blur=True,
                f_number=self.exposure.records["FNumber"].value,
                pixel_pitch_um=self.camera_profile.camera["sensor_width_mm"].value * 1000 / self.geometry_plan.width,
            ),
        )
        output.flush()
        self.optical_records = records
        self.groups["optics"] = [record for stage in records.values() for record in stage.values()]
        ProvenanceWriter().write(self.groups["optics"], self.path("metadata", "optical_parameters.json"))

    def electrons_noise(self) -> None:
        from synthraw.exposure import exposure_and_gain
        from synthraw.sensor import ElectronSignalSimulator, SensorNoiseSimulator

        sensor = self.camera_profile.sensor
        full_well = sensor["full_well_electrons"].value
        # Photon-transfer calibration supplies conversion gain when available; otherwise
        # map full well to the available DN range as a labelled simulation prior.
        if "conversion_gain_e_per_dn" not in sensor:
            sensor["conversion_gain_e_per_dn"] = FieldRecord(
                "conversion_gain_e_per_dn",
                full_well / (sensor["WhiteLevel"].value - sensor["BlackLevel"].value),
                Provenance.FABRICATED,
                "Full-well to ADC-range simulation prior",
                unit="electrons/DN",
            )
        self.conversion_gain = sensor["conversion_gain_e_per_dn"].value
        exposure = self.exposure.records
        gain_records = exposure_and_gain(self.config, exposure, self.camera_profile.camera)
        sensor.update(gain_records)
        self.analogue_gain = gain_records["applied_analogue_gain"].value
        self.noise_parameters = {
            "read_noise_e": sensor["read_noise_electrons"].value,
            "exposure_time_s": exposure["ExposureTime"].value,
            "prnu_sigma": sensor["prnu_sigma"].value,
            "dsnu_e": sensor["dsnu_sigma_dn"].value * self.conversion_gain,
            "row_fpn_e": 0.5,
            "column_fpn_e": 0.5,
            "dark_current_e_per_s": 0.1,
            "hot_pixel_fraction": 0.00001,
            "dead_pixel_fraction": 0.00001,
            "hot_pixel_e_per_s": 1000.0,
        }
        optical = np.load(self.path("intermediates", "06_optical_sensor_rgb.npy"), mmap_mode="r")
        expected = _new_memmap(self.path("intermediates", "07_expected_electrons.npy"), np.float64, optical.shape)
        ElectronSignalSimulator().simulate(
            optical,
            full_well,
            exposure_scale=gain_records["applied_exposure_scale"].value,
            strip_rows=self.strip_rows,
            out=expected,
        )
        expected.flush()
        noisy = _new_memmap(self.path("intermediates", "08_noisy_electrons.npy"), np.float64, optical.shape)
        SensorNoiseSimulator().simulate(
            expected, self.config, **self.noise_parameters, strip_rows=self.strip_rows, out=noisy
        )
        noisy.flush()
        for applied, original in (
            ("read_noise_e", "read_noise_electrons"),
            ("prnu_sigma", "prnu_sigma"),
            ("exposure_time_s", "ExposureTime"),
        ):
            record = (exposure if original == "ExposureTime" else sensor)[original]
            sensor[f"applied_{applied}"] = replace(
                record, field=f"applied_{applied}", value=self.noise_parameters[applied]
            )
        sensor["applied_dsnu_e"] = FieldRecord(
            "applied_dsnu_e",
            self.noise_parameters["dsnu_e"],
            Provenance.DETERMINISTIC,
            "DSNU in DN multiplied by applied conversion gain",
            unit="electrons",
            dependencies=["dsnu_sigma_dn", "conversion_gain_e_per_dn"],
        )
        for name in (
            "row_fpn_e",
            "column_fpn_e",
            "dark_current_e_per_s",
            "hot_pixel_fraction",
            "dead_pixel_fraction",
            "hot_pixel_e_per_s",
        ):
            sensor[f"applied_{name}"] = FieldRecord(
                f"applied_{name}",
                self.noise_parameters[name],
                Provenance.FABRICATED,
                "Hypothetical sensor simulation prior; not measured from a Sony sensor",
            )
        toggles = (
            "SENSOR_BODY_SEED",
            "SENSOR_NOISE_SEED",
            "SHOT_NOISE",
            "READ_NOISE",
            "PRNU",
            "DSNU",
            "ROW_FPN",
            "COLUMN_FPN",
            "DARK_CURRENT",
            "HOT_PIXELS",
            "DEAD_PIXELS",
        )
        for name, value in {name: getattr(self.config, name) for name in toggles}.items():
            sensor[f"applied_{name}"] = FieldRecord(
                f"applied_{name}", value, Provenance.DETERMINISTIC, "Explicit simulation configuration"
            )

    def cfa(self) -> None:
        from synthraw.cfa import CFAMosaicer

        self.cfa_phase = self.camera_profile.sensor["cfa_pattern"].value
        mosaicer = CFAMosaicer(self.cfa_phase)
        for source_name, target_name in (
            ("07_expected_electrons.npy", "09_clean_cfa.npy"),
            ("08_noisy_electrons.npy", "10_noisy_cfa.npy"),
        ):
            electrons = np.load(self.path("intermediates", source_name), mmap_mode="r")
            mosaic = _new_memmap(self.path("intermediates", target_name), electrons.dtype, electrons.shape[:2])
            mosaicer.mosaic(electrons, strip_rows=self.strip_rows, out=mosaic)
            mosaic.flush()
            del electrons, mosaic
        self.camera_profile.sensor["applied_cfa_phase"] = FieldRecord(
            "applied_cfa_phase",
            self.cfa_phase,
            Provenance.DETERMINISTIC,
            "Bayer channel sampling from the selected sensor profile",
            dependencies=["cfa_pattern"],
        )

    def adc(self) -> None:
        from synthraw.adc import ADCSimulator

        sensor = self.camera_profile.sensor
        noisy = np.load(self.path("intermediates", "10_noisy_cfa.npy"), mmap_mode="r")
        codes = _new_memmap(self.path("intermediates", "11_adc_codes.npy"), np.uint16, noisy.shape)
        self.adc_result = ADCSimulator().simulate(
            noisy,
            self.config,
            conversion_gain_e_per_dn=self.conversion_gain,
            read_noise_e=self.noise_parameters["read_noise_e"],
            analogue_gain=self.analogue_gain,
            black_level=sensor["BlackLevel"].value,
            white_level=sensor["WhiteLevel"].value,
            strip_rows=self.strip_rows,
            out=codes,
        )
        codes.flush()
        public = _new_memmap(self.path("raw", "sensor_mosaic_uint16.npy"), np.uint16, codes.shape)
        for start in range(0, codes.shape[0], self.strip_rows):
            public[start : start + self.strip_rows] = codes[start : start + self.strip_rows]
        public.flush()
        # Keep only the small result fields; the mosaic itself is reloaded from disk.
        self.adc_result = replace(self.adc_result, codes=None)
        for name, value in self.adc_result.applied_parameters.items():
            sensor[f"applied_adc_{name}"] = FieldRecord(
                f"applied_adc_{name}",
                value,
                Provenance.DETERMINISTIC,
                "Exact parameter applied by ADCSimulator; source priors/calibration retained in sensor profile",
            )
        sensor["applied_adc_noise_profile"] = FieldRecord(
            "applied_adc_noise_profile",
            self.adc_result.noise_profile,
            Provenance.DETERMINISTIC,
            "Normalised Poisson/read model from applied ADC parameters; omitted when shot noise is disabled",
            dependencies=[f"applied_adc_{name}" for name in self.adc_result.applied_parameters],
        )

    def metadata(self) -> None:
        from synthraw.dng_metadata import EXPOSURE_FIELDS, DNGMetadataBuilder
        from synthraw.sony_emulation import SonyARWEmulationBuilder, capture_inputs
        from synthraw.sony_registry import SonyMetadataRegistry

        colour = self.colour_model.metadata()
        self.dng_records = DNGMetadataBuilder().build(
            self.geometry_plan,
            self.config,
            cfa_phase=self.cfa_phase,
            black_level=self.adc_result.applied_parameters["black_level"],
            white_level=self.adc_result.applied_parameters["white_level"],
            color_matrix1=colour["ColorMatrix1"].value,
            color_matrix2=colour["ColorMatrix2"].value,
            as_shot_neutral=tuple(1.0 / gain for gain in self.white_balance.gains),
            noise_profile=self.adc_result.noise_profile,
            software_version=SOFTWARE_VERSION,
            # BaselineExposure stays a DNG profile prior; the posterior copy is kept in metadata.json.
            exposure={name: record for name, record in self.exposure.records.items() if name in EXPOSURE_FIELDS},
        )
        metadata_dir = self.run.directory("metadata")
        registry = SonyMetadataRegistry().build()
        emulation = SonyARWEmulationBuilder().build(
            registry,
            self.config,
            width=self.geometry_plan.width,
            height=self.geometry_plan.height,
            inputs=capture_inputs(
                registry,
                exposure=self.exposure.records,
                white_balance=self.white_balance.records,
                camera=self.camera_profile.camera,
            ),
            optical_records=self.optical_records,
        )
        _write_json(metadata_dir / "sony_tag_registry.json", registry)
        if self.config.WRITE_SONY_ARW_EMULATION_JSON:
            _write_json(metadata_dir / "sony_arw_emulation.json", emulation)
        ProvenanceWriter().write(list(self.dng_records.values()), metadata_dir / "dng_metadata.json")
        self.camera_profile.write(metadata_dir)
        fresh = {
            "dng": list(self.dng_records.values()),
            "camera_profile": list(self.camera_profile.camera.values()),
            "sensor_profile": list(self.camera_profile.sensor.values()),
        }
        self.groups = fresh | {name: records for name, records in self.groups.items() if name not in fresh}
        if self.config.WRITE_METADATA_JSON:
            ProvenanceWriter().write_metadata(self.groups, metadata_dir / "metadata.json")

    def dng_export(self) -> None:
        from synthraw.dng_writer import DNGWriter

        mosaic = np.load(self.path("raw", "sensor_mosaic_uint16.npy"), mmap_mode="r")
        destination = self.path("raw", "synthetic_50mp.dng")
        # Stage on the same filesystem; only a LibRaw-validated file is published.
        self.staging = tempfile.TemporaryDirectory(prefix=".dng_export_", dir=self.run.directory("raw"))
        try:
            self.staged_dng = DNGWriter().write(
                Path(self.staging.name) / destination.name, mosaic, self.dng_records, self.config
            )
        except BaseException:
            self.staging.cleanup()
            raise
        self.dng_path = destination

    def libraw_reopen(self) -> None:
        from synthraw.libraw_validator import LibRawValidator

        if not getattr(self, "staged_dng", None) or not self.staged_dng.is_file():
            raise RuntimeError("LibRaw reopen needs a newly staged DNG: re-run the DNG export cell first")
        mosaic = np.load(self.path("raw", "sensor_mosaic_uint16.npy"), mmap_mode="r")
        try:
            LibRawValidator().validate(self.staged_dng, mosaic, self.dng_records, develop=True)
            self.staged_dng.replace(self.dng_path)
        finally:
            self.staging.cleanup()
        _write_json(
            self.path("validation", "libraw_roundtrip.json"),
            {
                "dng": self.dng_path.name,
                "libraw_reopened": True,
                "pixels_identical": True,
                "metadata_matched": ["dimensions", "CFAPattern", "BlackLevel", "WhiteLevel", "AsShotNeutral"],
            },
        )

    def raw_development_tiff(self) -> None:
        from synthraw.exporters import TIFFWriter

        self.tiff_path = None
        if self.config.WRITE_TIFF16:
            # Developed by LibRaw from the published DNG, not from any pre-sensor RGB stage.
            self.tiff_path = TIFFWriter().write(
                self.path("masters", "developed_50mp_16bit.tif"), self.dng_path, rows_per_strip=self.strip_rows
            )

    def exr_preview(self) -> None:
        from synthraw.exporters import EXRWriter, PreviewWriter

        radiance = np.load(self.path("intermediates", "04_hdr_radiance.npy"), mmap_mode="r")
        self.exr_path = None
        if self.config.WRITE_EXR:
            self.exr_path = EXRWriter().write(self.path("masters", "scene_linear_50mp.exr"), radiance)
        self.preview_path = PreviewWriter().write(self.path("masters", "preview_8bit.png"), radiance)

    def provenance(self) -> None:
        tiled = self.sr_result.tiled
        self.c2pa = write_run_provenance(
            self.run,
            self.config,
            source_original=self.source_original,
            source_sha256=self.audit.sha256,
            dng_records=self.dng_records,
            groups=self.groups,
            backends={
                "super_resolution": tiled.backend,
                "hdr": self.hdr_report["backend"],
                "inverse_isp": self.inverse_isp_backend,
            },
            fallbacks=self.fallbacks,
            generated_hdr_area_percentage=self.hdr_report["generated_hdr_area_percentage"],
            learned_models=tiled.backend not in (None, "LANCZOS"),
            c2pa_targets=[path for path in (self.tiff_path, self.preview_path) if path is not None],
            c2pa_signer=self.options["c2pa_signer"],
        )
        self.log(f"C2PA: {self.c2pa['status']}")


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def open_execution_log(run: RunLayout, name: str = "synthraw.notebook") -> logging.Logger:
    """Return a logger appending to logs/execution.log; rebinding a name closes its previous run's log."""
    logger = logging.getLogger(name)
    _close_handlers(logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run.directory("logs") / "execution.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def publish_source(run: RunLayout, source: str | Path, audit: Any) -> Path:
    """Copy the audited input to source/source_original.ext and write source_sha256.txt (PRD tree)."""
    original = run.directory("source") / ("source_original" + SOURCE_EXTENSIONS[audit.image_format])
    shutil.copy2(source, original)
    if _sha256(original) != audit.sha256:
        raise RuntimeError("Source copy differs from the audited input")
    (run.directory("source") / "source_sha256.txt").write_text(f"{audit.sha256}  {original.name}\n", "utf-8")
    return original


def write_run_logs(run: RunLayout, environment: EnvironmentReport) -> tuple[Path, Path]:
    """Write logs/environment.json and logs/model_manifest.json for a run."""
    from synthraw.model_manifest import write_model_manifest

    return _write_json(run.directory("logs") / "environment.json", environment.to_dict()), write_model_manifest(
        run.root
    )


def write_run_provenance(
    run: RunLayout,
    config: ExperimentConfig,
    *,
    source_original: Path,
    source_sha256: str,
    dng_records: Mapping[str, FieldRecord],
    groups: Mapping[str, Iterable[FieldRecord]],
    backends: Mapping[str, str | None],
    fallbacks: Iterable[str],
    generated_hdr_area_percentage: float,
    learned_models: bool,
    c2pa_targets: Iterable[Path],
    c2pa_signer: Any = None,
) -> dict[str, Any]:
    """Apply optional C2PA, then write metadata/provenance.json with output hashes; returns the C2PA status.

    JSON provenance is mandatory and is written whether or not C2PA is available.
    """
    c2pa = write_c2pa(
        list(c2pa_targets),
        enabled=config.WRITE_C2PA,
        signer=c2pa_signer,
        learned_models=learned_models,
        version=SOFTWARE_VERSION,
    )
    outputs = sorted(
        path
        for directory in ("source", "raw", "masters", "metadata")
        for path in run.directory(directory).iterdir()
        if path.is_file() and path.name != "provenance.json"
    )
    counts = {cls.value: 0 for cls in Provenance}
    for records in groups.values():
        for record in records:
            counts[Provenance(record.provenance).value] += 1
    ProvenanceWriter().write_provenance(
        {
            "run": run.root.name,
            "software": {"name": "synthraw", "version": SOFTWARE_VERSION},
            "source": {"file": str(Path(source_original).relative_to(run.root)), "sha256": source_sha256},
            "dng_identity": {
                name: dng_records[name].value
                for name in ("Make", "Model", "UniqueCameraModel", "ImageDescription")
                if name in dng_records
            },
            "chain": list(STAGES),
            "backends": dict(backends),
            "fallbacks": list(fallbacks),
            "field_provenance_counts": counts,
            "generated_hdr_area_percentage": generated_hdr_area_percentage,
            "c2pa": c2pa,
            "outputs": {str(path.relative_to(run.root)): _sha256(path) for path in outputs},
        },
        run.directory("metadata") / "provenance.json",
    )
    return c2pa


def run_pipeline(
    source_path: str | Path,
    config: ExperimentConfig,
    output_root: str | Path,
    *,
    environment: EnvironmentReport | None = None,
    timestamp: datetime | None = None,
    fast_dev_max_side: int = 512,
    accept_licence: bool = False,
    scene_ev100: float = 10.0,
    manual_exposure: tuple[float, float, int] | None = None,
    manual_wb: dict[str, float] | None = None,
    manual_fov_degrees: float | None = None,
    strip_rows: int = 128,
    c2pa_signer: Any = None,
) -> PipelineResult:
    """Execute the full PRD_v3 chain from a JPEG/PNG into a new run_YYYYMMDD_HHMMSS tree.

    A DNG that LibRaw cannot reopen is a hard failure: the exception propagates, the
    DNG is not published and no success report is written. ``validation/run_report.json``
    exists only for a completed run.
    """
    state = PipelineRun.create(
        source_path,
        config,
        output_root,
        environment=environment,
        timestamp=timestamp,
        fast_dev_max_side=fast_dev_max_side,
        accept_licence=accept_licence,
        scene_ev100=scene_ev100,
        manual_exposure=manual_exposure,
        manual_wb=manual_wb,
        manual_fov_degrees=manual_fov_degrees,
        strip_rows=strip_rows,
        c2pa_signer=c2pa_signer,
    )
    try:
        for name in STAGES:
            state.run_stage(name)
        report = state.write_report()
        state.log("pipeline SUCCESS")
        return PipelineResult(
            state.run.root,
            state.dng_path,
            state.tiff_path,
            state.exr_path,
            state.preview_path,
            tuple(report["stages"]),
            tuple(state.fallbacks),
        )
    except HardValidationError as error:
        state.logger.error(f"HARD FAILURE: {error}; no success report written")
        raise
    except BaseException as error:
        state.logger.error(f"pipeline FAILED: {type(error).__name__}: {error}")
        raise
    finally:
        state.close()
