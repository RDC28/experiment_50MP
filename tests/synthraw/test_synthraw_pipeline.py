import hashlib
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rawpy
import tifffile
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.exporters import adobe_rgb_icc
from synthraw.hdr import HDRReconstructionEngine
from synthraw.libraw_validator import HardValidationError, LibRawValidator
from synthraw.pipeline import STAGES, run_pipeline, run_with_oom_fallback
from synthraw.provenance import (
    DIGITAL_SOURCE_TYPES,
    SCIENTIFIC_BOUNDARY,
    FieldRecord,
    Provenance,
    ProvenanceWriter,
    write_c2pa,
)

PRD_TREE = {
    "source/source_original.png",
    "source/source_sha256.txt",
    "raw/synthetic_50mp.dng",
    "raw/sensor_mosaic_uint16.npy",
    "masters/developed_50mp_16bit.tif",
    "masters/scene_linear_50mp.exr",
    "masters/preview_8bit.png",
    *(
        f"metadata/{name}.json"
        for name in (
            "metadata",
            "sony_arw_emulation",
            "sony_tag_registry",
            "sensor_profile",
            "camera_profile",
            "provenance",
            "config",
        )
    ),
    *(
        f"intermediates/{name}.npy"
        for name in (
            "01_input_rgb",
            "02_sr_rgb",
            "03_scene_linear",
            "04_hdr_radiance",
            "05_camera_native_rgb",
            "06_optical_sensor_rgb",
            "07_expected_electrons",
            "08_noisy_electrons",
            "09_clean_cfa",
            "10_noisy_cfa",
            "11_adc_codes",
        )
    ),
    *(
        f"confidence/{name}.npy"
        for name in (
            "generated_detail_mask",
            "source_supported_mask",
            "highlight_fabricated_mask",
            "shadow_fabricated_mask",
            "hdr_confidence",
        )
    ),
    "logs/environment.json",
    "logs/model_manifest.json",
    "logs/execution.log",
}


@pytest.fixture
def source_png(tmp_path):
    """A 48 x 20 synthetic gradient with clipped highlights and crushed shadows."""
    rng = np.random.default_rng(7)
    ramp = np.linspace(0, 1, 48)[None, :, None] * np.array([255, 230, 200])[None, None, :]
    pixels = np.clip(ramp + rng.integers(-20, 20, (20, 48, 3)), 0, 255).astype(np.uint8)
    path = tmp_path / "tiny source.png"
    Image.fromarray(pixels).save(path)
    return path


def run_tiny(source, output, geometry="SONY_A1II_NATIVE", **settings):
    config = ExperimentConfig(OUTPUT_GEOMETRY=geometry, FAST_DEV_RUN=True, **settings)
    return run_pipeline(
        source, config, output, timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=96, strip_rows=16
    )


@pytest.mark.parametrize(("geometry", "shape"), [("SONY_A1II_NATIVE", (64, 96)), ("CINEMA_50MP_VIRTUAL", (40, 96))])
def test_tiny_end_to_end_run_produces_prd_tree(source_png, tmp_path, geometry, shape):
    result = run_tiny(source_png, tmp_path / "runs", geometry)
    root = result.root
    assert root == tmp_path / "runs" / "run_20260102_030405"
    files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    assert PRD_TREE <= files
    assert not [name for name in files if ".dng_export" in name or name.startswith("intermediates/hdr_")]
    assert result.stages == STAGES

    digest = hashlib.sha256(source_png.read_bytes()).hexdigest()
    assert (root / "source/source_sha256.txt").read_text() == f"{digest}  source_original.png\n"
    assert (root / "source/source_original.png").read_bytes() == source_png.read_bytes()

    # The published DNG reopens independently with the exact stored mosaic.
    mosaic = np.load(root / "raw/sensor_mosaic_uint16.npy")
    assert mosaic.shape == shape
    assert np.array_equal(mosaic, np.load(root / "intermediates/11_adc_codes.npy"))
    with rawpy.imread(str(result.dng_path)) as raw:
        assert np.array_equal(raw.raw_image, mosaic)
    with tifffile.TiffFile(result.dng_path) as dng:
        tags = dng.pages[0].tags
        assert tags["Make"].value == "Synthetic"
        assert 37500 not in tags  # No MakerNote.
        assert tags[34855].value > 0  # PhotographicSensitivity from the exposure posterior.

    with tifffile.TiffFile(result.tiff_path) as tiff:
        page = tiff.pages[0]
        assert page.shape == (*shape, 3) and page.dtype == np.uint16
        assert page.tags[34675].value == adobe_rgb_icc()
        developed = page.asarray()
    assert 0 < developed.mean() < 65535 and developed.std() > 0

    import OpenEXR

    with OpenEXR.File(str(result.exr_path), separate_channels=True) as exr:
        red = exr.channels()["R"].pixels
    radiance = np.load(root / "intermediates/04_hdr_radiance.npy")
    assert np.array_equal(red, radiance[..., 0]) and red.shape == shape
    with Image.open(result.preview_path) as preview:
        assert preview.mode == "RGB" and preview.size == shape[::-1]

    metadata = json.loads((root / "metadata/metadata.json").read_text())
    assert metadata["scientific_boundary"] == SCIENTIFIC_BOUNDARY
    records = [record for group in metadata["groups"].values() for record in group]
    assert {"dng", "camera_profile", "sensor_profile", "exposure_posterior", "white_balance"} <= set(metadata["groups"])
    keys = {"field", "value", "unit", "provenance", "method", "confidence", "alternatives", "dependencies"}
    assert all(keys | {"user_override"} <= record.keys() for record in records)
    assert {record["provenance"] for record in records} <= {"D", "I", "F", "M"}
    fabricated = [record for record in records if record["provenance"] == "F"]
    assert fabricated and all(record["confidence"] is None and record["disclaimer"] for record in fabricated)

    provenance = json.loads((root / "metadata/provenance.json").read_text())
    assert provenance["synthetic_capture"] is True
    assert "No physical camera capture occurred" in provenance["disclosure"]
    assert provenance["dng_identity"]["Make"] == "Synthetic"
    assert provenance["source"]["sha256"] == digest
    assert provenance["c2pa"]["status"] in {"C2PA unavailable", "C2PA disabled"}
    assert provenance["outputs"]["raw/synthetic_50mp.dng"] == hashlib.sha256(result.dng_path.read_bytes()).hexdigest()
    # Default SWINIR must fall back through the PRD chain off Colab without downloads.
    assert provenance["backends"]["super_resolution"] == "LANCZOS"
    assert any(reason.startswith("SR SWINIR") for reason in provenance["fallbacks"])

    assert json.loads((root / "metadata/config.json").read_text())["OUTPUT_GEOMETRY"] == geometry
    assert json.loads((root / "validation/run_report.json").read_text())["status"] == "SUCCESS"
    assert json.loads((root / "logs/model_manifest.json").read_text())["models"]
    assert "system_ram_bytes" in json.loads((root / "logs/environment.json").read_text())
    log = (root / "logs/execution.log").read_text()
    assert all(f"stage {name}: done" in log for name in STAGES) and "pipeline SUCCESS" in log


def test_libraw_failure_is_hard_failure_without_success_report(source_png, tmp_path, monkeypatch):
    def reject(self, path, mosaic, records, *, develop=False):
        raise HardValidationError("LibRaw cannot reopen file")

    monkeypatch.setattr(LibRawValidator, "validate", reject)
    with pytest.raises(HardValidationError, match="cannot reopen"):
        run_tiny(source_png, tmp_path / "runs")
    root = tmp_path / "runs" / "run_20260102_030405"
    assert not (root / "validation/run_report.json").exists()
    assert not (root / "raw/synthetic_50mp.dng").exists()
    assert not (root / "masters/developed_50mp_16bit.tif").exists()
    assert not list((root / "raw").glob(".dng_export*"))
    log = (root / "logs/execution.log").read_text()
    assert "HARD FAILURE" in log and "pipeline SUCCESS" not in log


def test_hdr_oom_halves_strip_and_retries(source_png, tmp_path, monkeypatch):
    original = HDRReconstructionEngine.expand
    strips = []

    def flaky(self, *args, strip_rows, **kwargs):
        strips.append(strip_rows)
        if len(strips) == 1:
            raise RuntimeError("CUDA out of memory")
        return original(self, *args, strip_rows=strip_rows, **kwargs)

    monkeypatch.setattr(HDRReconstructionEngine, "expand", flaky)
    result = run_tiny(source_png, tmp_path / "runs", SR_BACKEND="LANCZOS")
    assert strips == [16, 8]
    assert any("OOM at tile 16" in reason for reason in result.fallbacks)
    assert (result.root / "intermediates/04_hdr_radiance.npy").exists()


def test_pipeline_requires_dng_and_existing_source(source_png, tmp_path):
    with pytest.raises(ValueError, match="WRITE_DNG"):
        run_pipeline(source_png, ExperimentConfig(WRITE_DNG=False), tmp_path)
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "missing.png", ExperimentConfig(), tmp_path)
    assert not list(tmp_path.glob("run_*"))


def oom():
    return RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")


def test_oom_fallback_halves_once_then_disables_generative():
    calls = []

    def operation(tile, disable_generative):
        calls.append((tile, disable_generative))
        if not disable_generative:
            raise oom()
        return "deterministic"

    messages = []
    assert run_with_oom_fallback(operation, tile=256, generative_enabled=True, log=messages.append) == "deterministic"
    assert calls == [(256, False), (128, False), (128, True)]
    assert len(messages) == 2


def test_oom_fallback_without_generative_backend_raises():
    calls = []

    def operation(tile, disable_generative):
        calls.append(tile)
        raise MemoryError("host")

    with pytest.raises(MemoryError):
        run_with_oom_fallback(operation, tile=64, generative_enabled=False)
    assert calls == [64, 32]


def test_oom_fallback_propagates_other_errors_immediately():
    calls = []

    def operation(tile, disable_generative):
        calls.append(tile)
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_with_oom_fallback(operation, tile=64, generative_enabled=True)
    assert calls == [64]


def test_metadata_json_adds_fabricated_disclaimer(tmp_path):
    records = [
        FieldRecord("full_well_electrons", 52000, Provenance.FABRICATED, "virtual_a1ii_profile_prior", unit="e"),
        FieldRecord("ExposureTime", 0.008, Provenance.INFERRED, "posterior", unit="s", confidence=0.42),
    ]
    path = ProvenanceWriter().write_metadata({"sensor": records}, tmp_path / "metadata.json")
    payload = json.loads(path.read_text())
    fabricated, inferred = payload["groups"]["sensor"]
    assert fabricated["confidence"] is None and fabricated["disclaimer"]
    assert inferred["confidence"] == 0.42 and inferred["disclaimer"] is None
    assert set(payload["provenance_classes"]) == {"D", "I", "F", "M"}


def test_provenance_json_boundary_cannot_be_overridden(tmp_path):
    with pytest.raises(ValueError, match="override"):
        ProvenanceWriter().write_provenance({"synthetic_capture": False}, tmp_path / "provenance.json")
    path = ProvenanceWriter().write_provenance({"run": "x"}, tmp_path / "provenance.json")
    payload = json.loads(path.read_text())
    assert payload["scientific_boundary"] == SCIENTIFIC_BOUNDARY and payload["synthetic_capture"] is True


def test_c2pa_disabled_or_unavailable_never_fails(tmp_path, monkeypatch):
    target = tmp_path / "preview.png"
    target.write_bytes(b"png")
    assert write_c2pa([target], enabled=False)["status"] == "C2PA disabled"
    monkeypatch.setitem(sys.modules, "c2pa", None)
    assert write_c2pa([target], enabled=True, signer=object())["status"] == "C2PA unavailable"
    monkeypatch.setitem(sys.modules, "c2pa", types.SimpleNamespace(Builder=None))
    status = write_c2pa([target], enabled=True, signer=None)
    assert status["status"] == "C2PA unavailable" and "signer" in status["reason"]
    assert target.read_bytes() == b"png"


def test_c2pa_signs_digital_source_type_with_configured_signer(tmp_path, monkeypatch):
    manifests = []

    class Builder:
        def __init__(self, manifest):
            manifests.append(json.loads(manifest))

        def sign(self, signer, mime_type, source, destination):
            assert signer == "signer" and mime_type == "image/tiff"
            destination.write(source.read() + b"+c2pa")

    monkeypatch.setitem(sys.modules, "c2pa", types.SimpleNamespace(Builder=Builder))
    target = tmp_path / "developed.tif"
    target.write_bytes(b"tiff")
    status = write_c2pa([target], enabled=True, signer="signer", learned_models=True)
    assert status["status"] == "C2PA signed" and status["signed"] == [str(target)]
    assert target.read_bytes() == b"tiff+c2pa"
    action = manifests[0]["assertions"][0]["data"]["actions"][0]
    assert action["digitalSourceType"] == DIGITAL_SOURCE_TYPES["trained"]
    assert "No physical camera capture occurred" in action["description"]
    assert not list(tmp_path.glob(".*.c2pa"))


def test_c2pa_signing_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    class Builder:
        def __init__(self, manifest):
            pass

        def sign(self, *args):
            raise RuntimeError("certificate expired")

    monkeypatch.setitem(sys.modules, "c2pa", types.SimpleNamespace(Builder=Builder))
    target = tmp_path / "preview.png"
    target.write_bytes(b"png")
    status = write_c2pa([target], enabled=True, signer="signer")
    assert status["status"] == "C2PA unavailable" and "certificate expired" in status["reason"]
    assert target.read_bytes() == b"png" and not list(tmp_path.glob(".*.c2pa"))


def _records(path):
    return {item["field"]: item for item in json.loads(path.read_text())}


def test_hdr_highlight_headroom_is_not_clipped_in_the_dng(source_png, tmp_path):
    """Regression: exposure scale 1 put HDR radiance above 1 at or past full well, so the DNG clipped it."""
    from synthraw.cfa import CFAMosaicer

    config = ExperimentConfig(FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    result = run_tiny(source_png, tmp_path / "runs", SR_BACKEND="LANCZOS")
    root = result.root
    sensor = _records(root / "metadata/sensor_profile.json")
    assert sensor["applied_exposure_scale"]["value"] == pytest.approx(
        2.0**-config.HDR_HIGHLIGHT_EV / sensor["applied_analogue_gain"]["value"]
    )
    optical = CFAMosaicer(sensor["applied_cfa_phase"]["value"]).mosaic(
        np.load(root / "intermediates/06_optical_sensor_rgb.npy")
    )
    mosaic = np.load(root / "raw/sensor_mosaic_uint16.npy")
    white = sensor["applied_adc_white_level"]["value"]
    headroom = (optical > 1.0) & (optical < 0.8 * 2.0**config.HDR_HIGHLIGHT_EV)
    assert headroom.sum() > 0
    assert mosaic[headroom].max() < white
    # Brighter reconstructed radiance stays brighter in the stored codes.
    assert np.corrcoef(optical[headroom], mosaic[headroom])[0, 1] > 0.9
    with tifffile.TiffFile(result.tiff_path) as tiff:
        assert np.mean(tiff.pages[0].asarray() == 65535) < 0.01


def test_posterior_iso_drives_analogue_gain_and_noise(source_png, tmp_path):
    def run_at(iso):
        config = ExperimentConfig(FAST_DEV_RUN=True, SR_BACKEND="LANCZOS", EXPOSURE_MODE="MANUAL")
        result = run_pipeline(
            source_png,
            config,
            tmp_path / f"iso{iso}",
            fast_dev_max_side=96,
            strip_rows=16,
            manual_exposure=(1 / 125, 5.6, iso),
        )
        root = result.root
        sensor = {name: item["value"] for name, item in _records(root / "metadata/sensor_profile.json").items()}
        with tifffile.TiffFile(result.dng_path) as dng:
            tags = dng.pages[0].tags
            dng_iso, noise_profile = tags[34855].value, tags[51041].value
        mosaic = np.load(root / "raw/sensor_mosaic_uint16.npy").astype(float)
        k = sensor["applied_adc_analogue_gain"] / sensor["applied_adc_conversion_gain_e_per_dn"]
        clean = np.load(root / "intermediates/09_clean_cfa.npy") * k + sensor["applied_adc_black_level"]
        unclipped = mosaic < sensor["applied_adc_white_level"]
        noise = float(np.std((mosaic - clean)[unclipped]))
        return sensor, dng_iso, noise_profile, noise, float(mosaic.mean())

    base, base_iso, base_profile, base_noise, base_mean = run_at(100)
    high, high_iso, high_profile, high_noise, high_mean = run_at(1600)
    assert (base_iso, high_iso) == (100, 1600)
    assert base["applied_analogue_gain"] == base["applied_adc_analogue_gain"] == 1
    assert high["applied_analogue_gain"] == high["applied_adc_analogue_gain"] == 16
    assert high["applied_exposure_scale"] == pytest.approx(base["applied_exposure_scale"] / 16)
    # Same DN brightness from 16x fewer photoelectrons, so the DNG NoiseProfile and the noise both rise.
    assert high_profile[0] == pytest.approx(16 * base_profile[0])
    assert high_mean == pytest.approx(base_mean, rel=0.05)
    assert high_noise > 3 * base_noise


def test_reference_raw_profile_runs_end_to_end(source_png, tmp_path):
    """Regression: REFERENCE_RAW stored LibRaw's 4-channel BlackLevel list and crashed the sensor stage."""
    from dataclasses import replace

    from synthraw.dng_metadata import DNGMetadataBuilder
    from synthraw.dng_writer import DNGWriter
    from synthraw.geometry import TargetGeometryPlanner

    geometry = replace(
        TargetGeometryPlanner().plan(64, 48),
        width=64,
        height=48,
        active_area=(0, 0, 48, 64),
        default_crop_origin=(0, 0),
        default_crop_size=(64, 48),
    )
    records = DNGMetadataBuilder().build(geometry, cfa_phase="GRBG", as_shot_neutral=(0.5, 1, 0.75))
    reference = DNGWriter().write(tmp_path / "reference.dng", np.full((48, 64), 1024, dtype=np.uint16), records)
    result = run_tiny(
        source_png,
        tmp_path / "runs",
        SR_BACKEND="LANCZOS",
        CAMERA_PROFILE_SOURCE="REFERENCE_RAW",
        REFERENCE_RAW_PATH=str(reference),
    )
    sensor = _records(result.root / "metadata/sensor_profile.json")
    assert sensor["BlackLevel"]["value"] == 512.0 and sensor["BlackLevel"]["provenance"] == "M"
    assert sensor["black_level_per_channel"]["value"] == [512] * 4
    assert sensor["applied_cfa_phase"]["value"] == "GRBG"
    with rawpy.imread(str(result.dng_path)) as raw:
        assert list(raw.black_level_per_channel) == [512] * 4
        assert np.array_equal(raw.raw_image, np.load(result.root / "raw/sensor_mosaic_uint16.npy"))
    # Reference measurements resolve as metadata dependencies and never contradict the DNG AsShotNeutral.
    from synthraw.validation_runner import ValidationRunner

    run = ValidationRunner().run(result.root)
    assert run.status == "PASSED", [check.to_dict() for check in run.checks if check.hard and not check.passed]


def test_sony_sidecar_carries_run_records_and_honours_its_toggle(source_png, tmp_path):
    result = run_tiny(source_png, tmp_path / "runs", SR_BACKEND="LANCZOS")
    sidecar = json.loads((result.root / "metadata/sony_arw_emulation.json").read_text())
    values = {entry["field_name"]: entry for entry in sidecar["fields"]}
    posterior = _records(result.root / "metadata/exposure_posterior.json")
    assert values["SonyISO"]["generated_value"] == posterior["PhotographicSensitivity"]["value"]
    assert values["ExposureTime"]["generated_value"] == posterior["ExposureTime"]["value"]
    assert values["SonyFNumber"]["provenance"] == posterior["FNumber"]["provenance"] == "I"
    for name in ("ShutterType", "WhiteBalance", "ColorTemperature", "BaseISO", "StopsAboveBaseISO"):
        assert values[name]["implementation_state"] == "EMULATED" and values[name]["provenance"] in "DIF", name
    disabled = run_tiny(source_png, tmp_path / "off", SR_BACKEND="LANCZOS", WRITE_SONY_ARW_EMULATION_JSON=False)
    assert not (disabled.root / "metadata/sony_arw_emulation.json").exists()
    assert (disabled.root / "metadata/sony_tag_registry.json").is_file()


def test_run_report_requires_every_stage(source_png, tmp_path):
    from synthraw.pipeline import PipelineRun

    config = ExperimentConfig(FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    pipeline = PipelineRun.create(source_png, config, tmp_path / "runs", fast_dev_max_side=96)
    try:
        for name in ("input_audit", "geometry"):
            pipeline.run_stage(name)
        with pytest.raises(RuntimeError, match="every pipeline stage"):
            pipeline.write_report()
        with pytest.raises(ValueError, match="Unknown pipeline stage"):
            pipeline.run_stage("validation")
    finally:
        pipeline.close()
    assert not (pipeline.run.root / "validation/run_report.json").exists()


def test_rerunning_stages_after_editing_a_hypothesis_refreshes_the_run(source_png, tmp_path):
    """Regression: notebook reruns duplicated checkpoints, kept stale metadata groups and fallbacks."""
    from synthraw.pipeline import PipelineRun
    from synthraw.validation import MetadataDependencyValidator

    config = ExperimentConfig(FAST_DEV_RUN=True)
    pipeline = PipelineRun.create(source_png, config, tmp_path / "runs", fast_dev_max_side=96, strip_rows=16)
    root = pipeline.run.root
    try:
        for name in STAGES:
            pipeline.run_stage(name)
        first_fallbacks = list(pipeline.fallbacks)
        assert any(message.startswith("SR SWINIR") for message in first_fallbacks)
        first_neutral = _records(root / "metadata/dng_metadata.json")["AsShotNeutral"]["value"]

        pipeline.options["manual_wb"] = {"manual_cct": 3200.0}
        pipeline.run_stage("camera_exposure")
        with pytest.raises(RuntimeError, match="every pipeline stage"):
            pipeline.write_report()
        with pytest.raises(RuntimeError, match="completed first"):
            pipeline.run_stage("optics")
        for name in STAGES[STAGES.index("inverse_isp") :]:
            pipeline.run_stage(name)
        report = pipeline.write_report()
    finally:
        pipeline.close()
    assert report["stages"] == list(STAGES) and report["fallbacks"] == first_fallbacks
    neutral = _records(root / "metadata/dng_metadata.json")["AsShotNeutral"]["value"]
    assert neutral != pytest.approx(first_neutral)
    groups = json.loads((root / "metadata/metadata.json").read_text())["groups"]
    dng = {record["field"]: record["value"] for record in groups["dng"]}
    assert dng["AsShotNeutral"] == pytest.approx(neutral)
    check = MetadataDependencyValidator().validate(groups)
    assert check.passed, check.measured["violations"]


@pytest.mark.parametrize("name", STAGES)
def test_rerunning_any_stage_keeps_every_published_output(source_png, tmp_path, name):
    """Regression: reruns hit create-only writers, deleted published outputs first, or consumed the staged DNG."""
    from synthraw.pipeline import PipelineRun

    config = ExperimentConfig(FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    pipeline = PipelineRun.create(source_png, config, tmp_path / "runs", fast_dev_max_side=96, strip_rows=16)
    root = pipeline.run.root

    def published():
        return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}

    try:
        for stage in STAGES:
            pipeline.run_stage(stage)
        pipeline.write_report()
        tree = published()
        sr_mask = (root / "confidence/sr_source_supported_mask.npy").read_bytes()
        first = name
        if name == "libraw_reopen":
            # Its staged DNG was published by the earlier run, so it asks for DNG export instead.
            with pytest.raises(RuntimeError, match="re-run the DNG export cell"):
                pipeline.run_stage(name)
            assert published() == tree
            first = "dng_export"
        pipeline.run_stage(first)
        # The rerun replaced its own outputs; later stages are stale but keep their files.
        assert published() >= tree
        if first != STAGES[-1]:
            with pytest.raises(RuntimeError, match="every pipeline stage"):
                pipeline.write_report()
        for stage in STAGES[STAGES.index(first) + 1 :]:
            pipeline.run_stage(stage)
        report = pipeline.write_report()
    finally:
        pipeline.close()
    assert report["stages"] == list(STAGES)
    assert published() == tree
    assert not list(root.glob(".rerun_*"))
    assert (root / "confidence/sr_source_supported_mask.npy").read_bytes() == sr_mask
    with rawpy.imread(str(root / "raw/synthetic_50mp.dng")) as raw:
        assert np.array_equal(raw.raw_image, np.load(root / "raw/sensor_mosaic_uint16.npy"))


def test_failed_rerun_restores_the_previous_outputs(source_png, tmp_path, monkeypatch):
    from synthraw.pipeline import PipelineRun

    config = ExperimentConfig(FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    pipeline = PipelineRun.create(source_png, config, tmp_path / "runs", fast_dev_max_side=96, strip_rows=16)
    root = pipeline.run.root
    try:
        for stage in STAGES:
            pipeline.run_stage(stage)
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        pipeline.run_stage("dng_export")

        def reject(self, path, mosaic, records, *, develop=False):
            raise HardValidationError("LibRaw cannot reopen file")

        monkeypatch.setattr(LibRawValidator, "validate", reject)
        with pytest.raises(HardValidationError):
            pipeline.run_stage("libraw_reopen")
    finally:
        pipeline.close()
    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    for relative in ("raw/synthetic_50mp.dng", "validation/libraw_roundtrip.json", "masters/developed_50mp_16bit.tif"):
        assert after[Path(relative)] == before[Path(relative)]
    assert not list(root.glob(".rerun_*")) and not list((root / "raw").glob(".dng_export*"))


def test_stage_refuses_to_overwrite_files_it_did_not_create(source_png, tmp_path):
    from synthraw.pipeline import PipelineRun

    config = ExperimentConfig(FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    pipeline = PipelineRun.create(source_png, config, tmp_path / "runs", fast_dev_max_side=96, strip_rows=16)
    foreign = pipeline.run.root / "intermediates/03_scene_linear.npy"
    try:
        for name in ("input_audit", "geometry", "super_resolution"):
            pipeline.run_stage(name)
        np.save(foreign, np.zeros(3))
        with pytest.raises(FileExistsError):
            pipeline.run_stage("hdr")
    finally:
        pipeline.close()
    np.testing.assert_array_equal(np.load(foreign), np.zeros(3))
    assert not (pipeline.run.root / "intermediates/04_hdr_radiance.npy").exists()


def _developed(path):
    with tifffile.TiffFile(path) as tiff:
        return tiff.pages[0].asarray()


def test_lossless_run_publishes_a_compressed_dng_that_develops_like_the_uncompressed_one(source_png, tmp_path):
    lossless = run_tiny(source_png, tmp_path / "lossless", DNG_COMPRESSION="LOSSLESS")
    plain = run_tiny(source_png, tmp_path / "plain")
    root = lossless.root
    mosaic = np.load(root / "raw/sensor_mosaic_uint16.npy")
    assert np.array_equal(mosaic, np.load(plain.root / "raw/sensor_mosaic_uint16.npy"))
    with tifffile.TiffFile(lossless.dng_path) as dng:
        page = dng.pages[0]
        assert page.compression == 7 and page.is_tiled and page.bitspersample == 16
    with rawpy.imread(str(lossless.dng_path)) as raw:
        assert np.array_equal(raw.raw_image, mosaic)
    # The LibRaw-developed master cannot tell the two containers apart.
    assert np.array_equal(_developed(lossless.tiff_path), _developed(plain.tiff_path))
    roundtrip = json.loads((root / "validation/libraw_roundtrip.json").read_text())
    assert roundtrip["compression"] == "LOSSLESS" and roundtrip["pixels_identical"] is True
    assert roundtrip["dng_bytes"] == lossless.dng_path.stat().st_size and roundtrip["mosaic_bytes"] == mosaic.nbytes
    report = json.loads((root / "validation/run_report.json").read_text())
    assert report["dng_compression"] == "LOSSLESS"
    plain_report = json.loads((plain.root / "validation/run_report.json").read_text())
    assert plain_report["dng_compression"] == "UNCOMPRESSED"
    # A single realisation is still recorded, with the configured seed.
    realisations = json.loads((plain.root / "validation/sensor_realisations.json").read_text())
    assert realisations["count"] == 1 and plain.realisation_dngs == ()
    assert realisations["realisations"][0]["sensor_noise_seed"] == 42001
    assert realisations["realisations"][0]["dng"] == "raw/synthetic_50mp.dng"


def test_multiple_sensor_realisations_are_validated_published_and_reproducible(source_png, tmp_path):
    from synthraw.sensor import realisation_noise_seeds

    result = run_tiny(source_png, tmp_path / "runs", N_SENSOR_REALISATIONS=3, DNG_COMPRESSION="LOSSLESS")
    root = result.root
    seeds = realisation_noise_seeds(42001, 3)
    names = [
        "raw/synthetic_50mp.dng",
        "raw/synthetic_50mp_realisation_002.dng",
        "raw/synthetic_50mp_realisation_003.dng",
    ]
    assert result.realisation_dngs == tuple(root / name for name in names[1:])
    assert not [path for path in root.rglob("*") if ".realisation_" in str(path)]
    report = json.loads((root / "validation/sensor_realisations.json").read_text())
    assert report["count"] == 3 and report["sensor_body_seed"] == 42002
    entries = report["realisations"]
    assert [entry["sensor_noise_seed"] for entry in entries] == list(seeds)
    assert [entry["dng"] for entry in entries] == names
    run_report = json.loads((root / "validation/run_report.json").read_text())
    assert run_report["sensor_realisations"] == [
        {"index": index, "sensor_noise_seed": seed, "dng": name}
        for index, (seed, name) in enumerate(zip(seeds, names, strict=True), 1)
    ]
    provenance = json.loads((root / "metadata/provenance.json").read_text())
    metadata = json.loads((root / "metadata/metadata.json").read_text())
    recorded = {record["field"]: record for record in metadata["groups"]["sensor_profile"]}
    assert recorded["applied_sensor_noise_realisation_seeds"]["value"] == list(seeds)
    primary = np.load(root / "raw/sensor_mosaic_uint16.npy")
    decoded = []
    for entry in entries:
        path = root / entry["dng"]
        assert provenance["outputs"][entry["dng"]] == entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with rawpy.imread(str(path)) as raw:
            decoded.append(raw.raw_image.copy())
    assert np.array_equal(decoded[0], primary)
    for entry, mosaic in zip(entries[1:], decoded[1:], strict=True):
        difference = mosaic.astype(np.float64) - primary
        assert entry["rms_difference_from_primary_dn"] == pytest.approx(np.sqrt(np.mean(difference**2)))
        assert entry["mean_difference_from_primary_dn"] == pytest.approx(difference.mean())
        # Independent noise, same scene: the realisations differ but are not offset from each other.
        assert entry["rms_difference_from_primary_dn"] > 0
        assert abs(entry["mean_difference_from_primary_dn"]) < entry["rms_difference_from_primary_dn"] / 4
    assert not np.array_equal(decoded[1], decoded[2])
    # Running with a recorded seed reproduces that realisation exactly.
    rerun = run_tiny(source_png, tmp_path / "rerun", SENSOR_NOISE_SEED=seeds[2])
    assert np.array_equal(np.load(rerun.root / "raw/sensor_mosaic_uint16.npy"), decoded[2])


def test_without_synthetic_exif_capture_settings_stay_out_of_the_dng_and_sidecar(source_png, tmp_path, monkeypatch):
    monkeypatch.setattr("synthraw.sony_registry.shutil.which", lambda name: None)
    result = run_tiny(source_png, tmp_path / "runs", GENERATE_SYNTHETIC_EXIF=False)
    root = result.root
    with tifffile.TiffFile(result.dng_path) as dng:
        tags = dng.pages[0].tags
        # ExposureTime, FNumber, PhotographicSensitivity, ShutterSpeedValue, ApertureValue.
        assert not {33434, 33437, 34855, 37377, 37378} & {tag.code for tag in tags}
        assert tags["Make"].value == "Synthetic"
    sidecar = {
        entry["field_name"]: entry
        for entry in json.loads((root / "metadata/sony_arw_emulation.json").read_text())["fields"]
    }
    for name in ("SonyISO", "SonyFNumber", "ExposureTime", "WhiteBalance"):
        assert sidecar[name]["generated_value"] is None, name
        assert sidecar[name]["unsupported_reason"].startswith("GENERATE_SYNTHETIC_EXIF is False"), name
    # The hypotheses still drive the simulation and stay in the provenance record.
    posterior = json.loads((root / "metadata/exposure_posterior.json").read_text())
    assert "PhotographicSensitivity" in {record["field"] for record in posterior}
    report = json.loads((root / "validation/run_report.json").read_text())
    assert report["status"] == "SUCCESS"
    assert "Sony tag registry: bundled seed only (ExifTool is not installed)" in report["fallbacks"]

    manual = run_pipeline(
        source_png,
        ExperimentConfig(FAST_DEV_RUN=True, GENERATE_SYNTHETIC_EXIF=False, EXPOSURE_MODE="MANUAL"),
        tmp_path / "manual",
        timestamp=datetime(2026, 1, 2, 3, 4, 5),
        fast_dev_max_side=96,
        strip_rows=16,
        manual_exposure=(1 / 125, 5.6, 400),
    )
    with tifffile.TiffFile(manual.dng_path) as dng:
        assert dng.pages[0].tags[34855].value == 400  # Explicit user exposure is still written.
