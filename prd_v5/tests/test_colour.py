"""Independent gray-world arithmetic and executable analytic gain evidence."""

import json

import numpy as np
import pytest

from prd_v5.analytic import AnalyticISP
from prd_v5.colour import resolve_gray_world
from prd_v5.fields import validate_selections


def test_gray_world_arithmetic_exclusion_and_strip_invariance():
    source = np.array([[[0.2, 0.4, 0.6], [0, 0.4, 0.6]], [[0.4, 0.6, 0.8], [1, 0.5, 0.5]]])
    samples = np.array([[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]])
    means = (((samples + 0.055) / 1.055) ** 2.4).mean(axis=0)
    isp, records, report = resolve_gray_world(AnalyticISP(exposure_gain=2), source, strip_rows=1)
    assert np.allclose(isp.white_balance, means[1] / means)
    assert np.allclose(report["as_shot_neutral"], means / means[1])
    assert report["eligible_pixels"] == report["excluded_endpoint_pixels"] == 2
    assert isp.exposure_gain == 2
    assert resolve_gray_world(AnalyticISP(exposure_gain=2), source, strip_rows=2)[0] == isp
    selected = validate_selections(isp, records, isp=True)["isp.white_balance"]
    assert selected["provenance"] == "ESTIMATED" and selected["confidence"] is None
    json.dumps(report, allow_nan=False)


def test_estimated_gains_drive_actual_inverse_pixels(tmp_path):
    source = np.full((2, 2, 3), [0.2, 0.4, 0.6], dtype=np.float32)
    base = AnalyticISP()
    isp, _, _ = resolve_gray_world(base, source)
    outputs = []
    for index, hypothesis in enumerate((base, isp)):
        dest = np.memmap(tmp_path / str(index), mode="w+", shape=source.shape, dtype=np.float32)
        try:
            assert hypothesis.process(source, dest)["forward_consistency"]["status"] == "passed"
            outputs.append(dest.copy())
        finally:
            dest._mmap.close()
    assert np.allclose(outputs[1], outputs[0] / isp.white_balance)
    assert np.all(outputs[1][..., 0] < outputs[0][..., 0])
    assert np.all(outputs[1][..., 2] > outputs[0][..., 2])


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((2, 2, 3)),
        np.ones((2, 2, 3)),
        np.full((2, 2, 3), np.nan),
        np.full((2, 2, 3), 1.1),
        np.ones((2, 2)),
        np.ones((2, 2, 3), dtype=np.uint8),
    ],
)
def test_invalid_or_empty_evidence_fails(source):
    with pytest.raises(ValueError):
        resolve_gray_world(AnalyticISP(), source)


@pytest.mark.parametrize("strip_rows", [0, True, 1.5])
def test_invalid_strip_rows_fail(strip_rows):
    with pytest.raises(ValueError):
        resolve_gray_world(AnalyticISP(), np.full((1, 1, 3), 0.5), strip_rows=strip_rows)
