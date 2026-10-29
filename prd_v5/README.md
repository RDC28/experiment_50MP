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

The remaining MVP config, complete registry and new Colab notebook are not
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
checkpoint compatibility, full-scale RAW decoder, performance or scientific acceptance
claim follows from the current arithmetic tests.

## MVP DNG export

`prd_v5.dng.write_dng` connects the shared camera, geometry and analytic ISP
parameters to the existing uncompressed strip writer. It preserves 14-bit ADC
values in uint16 storage, CFA phase, crop, black/white levels, exposure and the
same temporal NoiseProfile used by the sensor. Colour matrices are derived from
the analytic camera-to-sRGB transform through a declared virtual D65 profile;
white balance supplies AsShotNeutral. This is not calibrated Sony colour.
Returned tag records use the legacy writer contract; the full v5 registry is
still pending. Exact sample/tag checks with tifffile pass on tiny fixtures.
The writer alone reports independent validation as not measured. Call
`roundtrip.develop_roundtrip` separately to validate with rawpy/LibRaw and write
an uncompressed 16-bit sRGB TIFF, checking exact TIFF samples in strips. Tiny
fixtures pass for all four Bayer patterns and analytic-to-ADC output. The
independent gate checks samples, dimensions, CFA, levels, WB, colour matrix and
orientation. DefaultCrop stays DNG metadata; the TIFF uses LibRaw visible raster.
LibRaw allocates whole-frame buffers: a configurable 64-byte/photosite admission
estimate defaults to 512 MiB, rejecting 50 MP until explicitly raised in Colab.
This is not a peak RSS guarantee. Adobe conformance, calibrated colour accuracy
and full-size Colab validation remain unmeasured.

## Analytic MVP runner

`prd_v5.pipeline.run_analytic(source_path, new_output_dir, camera, ...)` now
connects decoding, bilinear geometry, inverse ISP, CFA/electron noise/ADC, DNG
and independent LibRaw TIFF validation. Supply the same `CameraHypothesis` to
sequential input runs. The new output directory contains DNG, TIFF and a JSON
completion manifest with hashes, stage reports and the current partial registry.
Temporary memmaps are closed and removed even on failure; failed runs have no
completion manifest. Existing output directories are rejected. This explicit
analytic API is not the forthcoming full PRD configuration or a learned fallback.
It defaults to 50 MP but rejects that size under its conservative development
memory budget. Full-scale calls require explicit Colab budgets. The source must
remain unchanged during a call. The bundle also retains `source_support.npy`, a boolean full-sensor geometry mask
written in strips and included in manifest hashes. Load it with
`numpy.load(path, mmap_mode="r", allow_pickle=False)` to avoid a full-frame
allocation. True means source-supported pixel-center geometry; false means a
sensor margin. This is not a clipping mask or confidence in recovered detail.
Its coordinates precede DNG crop and TIFF development. The bundle also includes
`source_clipped.npy`, a boolean mask of source-supported resampled sRGB pixels
with any channel at 1, with a count and hash in the manifest. It flags highlight
uncertainty without claiming recovery. ICC conversion can change endpoints and
bilinear interpolation can dilute them; below-endpoint clipping is undetected.
False values must be interpreted alongside source support, not as proof of known
radiance.

Each successful bundle also includes a self-contained offline `report.html`,
using the existing synthraw report's system typography and neutral tables.
It shows artifact hashes, stage evidence, mask semantics, shared hypotheses,
provenance and pending work. Metadata is HTML-escaped and no scripts, remote
assets or image allocations are needed. Scientific acceptance remains explicitly
not measured, separate from file roundtrip success. The manifest hashes the
report; the report lists the image/mask artifacts rather than its own hash.
Report export must succeed before the completion manifest is written. Further
uncertainty masks, complete configuration/registry and the notebook remain pending.

The analytic registry now also records observed source hashes/header dimensions,
explicit source colour policy, supplied inverse-ISP hypotheses and derived sensor
geometry in the same field schema as camera values. Namespaces `source.`, `isp.`
and `geometry.` prevent collisions. Confidence/posteriors stay null; synthetic
colour/exposure assumptions are not inferred capture truth. Source paths and
private EXIF values are excluded from these added field records. This remains a
partial registry, not the complete PRD field-mode configuration.

`config.AnalyticRunConfig` provides strict JSON serialization and execution for
this implemented analytic subset. Use `AnalyticRunConfig.from_json(text)` and
`config.run(source_path, new_output_dir)`; `config.to_json()` stores actual camera
and ISP hypotheses for reuse across sequential inputs. Unknown/duplicate fields,
unsupported schema versions and invalid numeric values are rejected. Source and
output paths are supplied separately. Defaults retain 50 MP and conservative
memory admission, so full-scale use requires explicit Colab budgets. This is not
the full PRD settings panel: HAT, HYBRID, HDR, learned, full sensor/optics and
field-mode orchestration remain pending rather than silently ignored switches.

The analytic JSON config and runner also accept `geometry_mode` with
`PRESERVE_SOURCE_ASPECT_50MP` (unchanged default) or `SONY_A1II_NATIVE`, and
`native_fit_policy` with `LETTERBOX_SENSOR_MARGIN` (default), `CROP_HORIZONTAL`
or `OUTPAINT_VERTICAL`. Native mode fixes the raster at 8640x5760 regardless of
`target_megapixels`; its effective policy, transform and crop appear in the
manifest and registry. Aspect-preserving mode uses letterbox fitting as before.
An outpaint request that needs generated bands fails before output creation or
frame allocation because no generator is integrated. An exact 3:2 input needs
no generated bands. Native processing still requires an explicit Colab memory
budget and has not been executed at full scale.
