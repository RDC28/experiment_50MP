import hashlib
import json
import sys
import types
from datetime import datetime

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
