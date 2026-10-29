# PRD v5 implementation status

The immutable specification is `PRD_v5.md`; scientific limits and the ordered
implementation plan are in `EVALUATION.md`.

## Built

The first MVP foundation is `hypothesis.py`: an immutable synthetic Sony-like
14-bit camera hypothesis in a uint16 storage envelope, seeded electronics
priors, resolution-independent stage seeds, linked gain/saturation/NoiseProfile
arithmetic, and a JSON-compatible initial registry. Identity fields remain null
and user-only. Registry values are labeled synthetic or derived, never recovered
camera measurements. Sampling is reproducible within the Python runtime; record
the runtime version and serialize the actual hypothesis for cross-runtime reuse.

The initial registry is a subset, not full PRD parameter coverage. Null confidence
and posterior mean no inferred confidence or exposure posterior was measured.
Stream seeds alone do not yet establish spatially consistent full-frame noise;
the future strip-based sensor implementation must preserve draw ordering.

## Pending

The remaining MVP config, complete registry, geometry, input inspection, bounded
pixel pipeline, DNG/rawpy/TIFF roundtrip, reports and new Colab notebook are not
implemented yet. Fidelity, learned RAW, full sensor, full optics and final
metadata follow MVP in that order. No HAT, HDR model or learned RAW checkpoint is
executed by this foundation. Upstream checkpoint and scientific limitations in
`EVALUATION.md` still apply.

## Lightweight validation

From the repository root, with pytest and ruff in an external environment:

```sh
python -m pytest prd_v5/tests -q
python -m ruff check .
```

These tests allocate no image arrays and download no models. Run any full suite
only through `/Users/rohit/Projects/firstmate/data/exp50mp-heavy-lock.sh`.

## Colab execution

No v5 notebook is runnable yet. Full-scale validation must wait for the new
notebook and execute in Colab, never on this 8 GB development Mac. No 50 MP,
checkpoint compatibility, RAW decoder, performance or scientific acceptance
claim follows from the current arithmetic tests.
