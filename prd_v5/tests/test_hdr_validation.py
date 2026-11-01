"""Truth-claim policy fixtures, without HDR models or inferred truth."""

import json

import numpy as np
import pytest

from prd_v5.hdr_validation import validate_hdr_claims


@pytest.mark.parametrize("rows", [1, 3, 64])
def test_claims_in_endpoints_and_margins_fail(rows):
    clipped = np.zeros((5, 7), dtype=bool)
    support = np.ones_like(clipped)
    clipped[1, 2] = clipped[4, 6] = True
    support[0, :] = False
    # One endpoint also in the margin: forbidden union must not double count.
    clipped[0, 0] = True
    claims = np.ones_like(clipped)
    result = validate_hdr_claims(clipped, support, claims, strip_rows=rows)
    assert result["status"] == result["claim_policy"] == "failed"
    assert result["forbidden_claim_pixels"] == 9
    assert result["source_endpoint_pixels"] == 3
    assert result["unsupported_pixels"] == 7
    assert result["claimed_truth_pixels"] == 35
    assert result["numerical_hdr_fidelity"] == "not measured"
    json.dumps(result, allow_nan=False)


def test_nonendpoint_claims_do_not_establish_truth():
    clipped = np.array([[True, False], [False, False]])
    claims = ~clipped
    result = validate_hdr_claims(clipped, np.ones_like(clipped), claims)
    assert result["status"] == "not measured"
    assert result["claim_policy"].startswith("passed")
    assert result["claimed_truth_pixels"] == 3
    assert result["forbidden_claim_pixels"] == 0


def test_memmaps_are_read_only_and_missing_claims_remain_unmeasured(tmp_path):
    paths = [tmp_path / name for name in ("clipped.npy", "support.npy")]
    for path, value in zip(paths, (True, False), strict=True):
        np.save(path, np.full((3, 4), value, dtype=bool))
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    try:
        result = validate_hdr_claims(*arrays)
        assert result["status"] == "not measured"
        assert result["source_endpoint_pixels"] == result["unsupported_pixels"] == 12
        assert result["claimed_truth_pixels"] == result["forbidden_claim_pixels"] == 0
        assert np.all(arrays[0]) and not np.any(arrays[1])
    finally:
        for array in arrays:
            array._mmap.close()


@pytest.mark.parametrize("shape", [(0, 2), (2,), (1, 2, 3)])
def test_invalid_dimensions_fail(shape):
    array = np.zeros(shape, dtype=bool)
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_hdr_claims(array, array)


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros((2, 3), dtype=bool)])
def test_invalid_masks_fail(bad):
    good = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="aligned boolean"):
        validate_hdr_claims(good, bad)
    with pytest.raises(ValueError, match="aligned boolean"):
        validate_hdr_claims(good, good, bad)


@pytest.mark.parametrize("rows", [0, -1, True, 1.5])
def test_invalid_strip_budget_fails(rows):
    good = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="positive integer"):
        validate_hdr_claims(good, good, strip_rows=rows)
