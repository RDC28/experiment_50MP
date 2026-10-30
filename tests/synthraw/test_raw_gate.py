"""Real independent RAW milestone gate, using only tiny Bayer fixtures."""

from dataclasses import replace

import numpy as np
import pytest

from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import CFA_PATTERNS, DNGMetadataBuilder
from synthraw.dng_writer import DNGWriter
from synthraw.geometry import TargetGeometryPlanner
from synthraw.libraw_validator import HardValidationError, LibRawValidator


def tiny_raw(tmp_path, phase="RGGB", bits=14):
    geometry = replace(
        TargetGeometryPlanner().plan(64, 48),
        outpaint_bands=(),  # The bare test sensor has no planned outpaint bands.
        width=64,
        height=48,
        active_area=(0, 0, 48, 64),
        default_crop_origin=(0, 0),
        default_crop_size=(64, 48),
    )
    config = ExperimentConfig(ADC_BITS=bits)
    records = DNGMetadataBuilder().build(
        geometry,
        config,
        cfa_phase=phase,
        as_shot_neutral=(0.5, 1, 0.75),
        color_matrix2=((0.8, 0.1, 0.05), (0.05, 1, 0.1), (0.02, 0.1, 0.9)),
    )
    mosaic = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + 512
    path = DNGWriter().write(tmp_path / "gradient.dng", mosaic, records, config)
    return path, mosaic, records


@pytest.mark.parametrize("phase", CFA_PATTERNS)
@pytest.mark.parametrize("bits", (14, 16))
def test_real_raw_gate(tmp_path, phase, bits):
    LibRawValidator().validate(*tiny_raw(tmp_path, phase, bits), develop=True)


@pytest.mark.parametrize("field", ("CFAPattern", "BlackLevel", "WhiteLevel", "AsShotNeutral", "ColorMatrix2", "pixels"))
def test_gate_rejects_mismatch(tmp_path, field):
    path, mosaic, records = tiny_raw(tmp_path)
    if field == "pixels":
        mosaic[0, 0] += 1
    else:
        records[field].value = {
            "CFAPattern": CFA_PATTERNS["BGGR"],
            "BlackLevel": 1024,
            "WhiteLevel": 8191,
            "AsShotNeutral": (1, 1, 1),
            "ColorMatrix2": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        }[field]
    with pytest.raises(HardValidationError):
        LibRawValidator().validate(path, mosaic, records)


def test_decoder_failure_is_hard(tmp_path):
    path, mosaic, records = tiny_raw(tmp_path)
    path.write_bytes(b"not a RAW")
    with pytest.raises(HardValidationError, match="Independent LibRaw"):
        LibRawValidator().validate(path, mosaic, records)
