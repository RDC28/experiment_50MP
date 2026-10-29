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
Sensor stage/row streams preserve temporal draw ordering across strip sizes.
Fixed spatial sensor maps and full-scale consistency validation remain pending.

`geometry.py` adds an allocation-free `GeometryPlanner`: aspect-preserving
50 MP planning, fixed 8640x5760 native geometry, uniform source transforms,
explicit source crops and sensor margins, CFA-aligned DNG crop metadata and
individual buffer byte budgets. Outpaint plans identify required generated
bands as not executed. This planner performs no image reconstruction or padding.

`inputs.py` adds header-only JPEG/PNG source inspection with streaming SHA-256,
encoded dimensions/precision, ICC hash/description and explicit colour policy.
Unprofiled RGB needs an explicit sRGB assumption; embedded ICC requires a future
transform and malformed ICC is blocked. Alpha and unsupported pixel modes are
rejected. EXIF presence is recorded without copying private identity. This is
not pixel decoding, clipping diagnostics, or source integrity certification.

`analytic.py` adds strip-based sRGB inverse transfer, optional inverse smoothstep,
inverse colour matrix, white-balance and exposure removal into a float32 memmap.
It accepts normalized sRGB arrays, including disk-backed inputs, and reports
clipped-source uncertainty and floored negative camera samples. Its identity
profile is virtual sRGB, not Sony calibration; it performs no highlight recovery,
source decoding or ICC conversion. Caller-owned partial output must be discarded
if processing fails.

`sensor.py` adds CFA-first electron-domain Poisson shot and Gaussian read noise,
analogue gain, pedestal and rounded/clipped 14-bit codes in a uint16 memmap.
All four Bayer patterns support odd dimensions. Stage/absolute-row seeds preserve
strip invariance at a shared final sensor geometry. Reports include clipping,
zero-effect read noise and NumPy version. Synthetic flat/dark tests check noise
variance; no measured camera calibration or full sensor effects are claimed.

`raster.py` applies geometry with an explicit bilinear encoded-sRGB baseline,
pixel-center coordinates and row-bounded float32 memmap output. A disk-backed
boolean source-support mask distinguishes zero-signal sensor margins. Plans
requiring unavailable outpainting fail before writing. This is not HAT or a
silent SR fallback; learned fidelity remains pending.

`decode.py` decodes 8-bit RGB JPEG/PNG to normalized sRGB float32 memmaps,
with explicit unprofiled assumptions or LittleCMS relative-colorimetric ICC
conversion. It reports endpoint clipping indicators, uses stored raster orientation
and rejects over-budget sources before decode. Pillow holds source pixels in RAM;
the configurable eight-byte-per-pixel gate is a decoder estimate, not a peak RSS
guarantee. Strip buffers are additional. ICC output remains 8-bit; this is not HDR
or a precision-preserving 16-bit source decoder.

## Pending

The remaining MVP config, complete registry, pipeline orchestration, DNG/rawpy/TIFF roundtrip, reports and new Colab notebook are not
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

These tests use tiny source fixtures, analytic transforms and camera/geometry arithmetic, with no model downloads. Run any full suite
only through `/Users/rohit/Projects/firstmate/data/exp50mp-heavy-lock.sh`.

## Colab execution

No v5 notebook is runnable yet. Full-scale validation must wait for the new
notebook and execute in Colab, never on this 8 GB development Mac. No 50 MP,
checkpoint compatibility, RAW decoder, performance or scientific acceptance
claim follows from the current arithmetic tests.
