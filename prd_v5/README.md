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

Successful analytic bundles record loaded Python, NumPy, Pillow, LittleCMS,
tifffile, rawpy and native LibRaw versions in `manifest.json` and `report.html`.
Only OS family and machine architecture are included, without host identity,
paths or environment variables. Missing optional LittleCMS support is null.
This runtime evidence helps diagnose replay differences; it does not pin an
environment or establish cross-runtime bitwise reproducibility.

Complete MVP configuration and executable field coverage remain pending.
The initial Colab notebook is available as described below. Fidelity, learned RAW, full sensor, full optics and final
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

Installing this repository with `pip install .` includes the `prd_v5` library
and its analytic JSON schema alongside the legacy packages. Read the installed
schema with
`importlib.resources.files("prd_v5").joinpath("analytic-config.schema.json")`;
it does not require a checkout-relative file path. Installation does not supply
learned checkpoints or enable unavailable integrations.

Open `notebooks/synthetic_50mp_v5.ipynb` in Colab with a matching checkout at
`/content/experiment_50MP`. Supply both source paths and a fresh output path.
The notebook inspects hardware, saves requests and effective states, and exposes
all PRD parameter-table entries as an editable request catalogue. Supported camera catalogue overrides execute through field-mode resolution;
unsupported overrides are rejected. Supported camera/ISP values
are editable separately. Maximum defaults request HAT, HYBRID, HDR, learned RAW
and extended sensor/optics; these integrations are visibly skipped.
Set `ALLOW_ANALYTIC_BASELINE=True` explicitly to execute the existing sequential
50 MP analytic pair, after reviewing skips and resource budgets. Unprofiled
inputs require explicit `ASSUME_UNPROFILED_SRGB=True`. Output is not a full-stack
completion claim. Strict learned mode always fails until a backend exists.
Full-scale validation must execute in Colab, never on this 8 GB development Mac. No 50 MP,
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

`analytic-config.schema.json` is a portable JSON Schema 2020-12 contract for
this implemented subset, including nested camera/ISP settings and geometry
domains. Omitted fields use `AnalyticRunConfig` defaults. Validate saved JSON
with a standard schema validator, then always call `AnalyticRunConfig.from_json`
before execution: structural validation cannot certify matrix conditioning,
finite arithmetic, duplicate keys or runtime memory admission. JSON Schema also
treats integral JSON numbers such as `1.0` as integers; the parser requires
integer tokens for integer settings. Tests use `jsonschema` (also installed by
the existing `nbformat` development dependency) and check schema field coverage
against the actual dataclasses. This schema does not expose or claim unbuilt
PRD features.

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

Each successful analytic bundle now includes a hashed `config.json` containing
all executed analytic settings, actual camera/ISP hypotheses and resource budgets.
Replay with `AnalyticRunConfig.from_json(Path("bundle/config.json").read_text())`,
then call `.run(source_path, new_output_dir)` with the original source. Source
and output paths remain separate; check the source SHA-256 against the manifest.
The offline HTML report lists the configuration artifact and digest. Tiny tests
verify identical DNG, TIFF and mask hashes when replayed within the same runtime;
this is not a cross-version or cross-platform bitwise reproducibility guarantee.
The saved settings cover only the implemented analytic subset, not later models.

Analytic stage reports include `forward_consistency`: the stored clean float32
camera RGB is rendered through the declared exposure/WB, matrix, display clip,
optional smoothstep and sRGB transfer, then compared with resampled input.
Strip-bounded maximum absolute error and RMSE use normalized encoded-sRGB units;
maximum error above 1e-5 marks this diagnostic failed. Gamut flooring can fail
the check without preventing bundle export. This measures internal consistency,
not calibrated colour, recovered highlights or scene radiance truth.

The analytic registry includes `dng.*` records for the intended metadata produced
by the same builder used for export. This exposes derived colour matrices,
AsShotNeutral, baseline exposure, crop, orientation and explicitly synthetic
identity without duplicating writer constants. These records describe intended
tags; actual independent decode evidence stays in the roundtrip stage report.
Values precede DNG rational rounding. With shot noise disabled, the registry
retains a null/UNSPECIFIED `dng.NoiseProfile` and the DNG omits that tag because
its slope must be positive. This remains the analytic subset of the full registry.

For sequential source-resolution runs, call
`run_analytic_pair(source_2k, source_4k, new_output_dir, config)` from
`prd_v5.paired`. The same frozen `AnalyticRunConfig` supplies actual camera/ISP
hypotheses, seeds and settings to both runs. Each child closes working arrays
before the next starts. Outputs live in `2k/` and `4k/`; `pair_manifest.json`
links their completed manifests with hashes, observed source dimensions and
output geometry. It appears only after both runs succeed. A failed second run
retains the first bundle for diagnosis without claiming pair completion; retry
in a new directory. Labels are caller-supplied and do not verify scene identity,
registration or exact resolution. Shared seeds do not imply identical noise for
different pixel signals. Pair comparison metrics remain unimplemented. Full-size
paired processing requires Colab and explicit resource budgets; never run it on
the development Mac.

Pass `sr_checkpoint=SRCheckpoint(...)` to `run_analytic_pair` or `config.run`
to opt into verified checkpoint SR before analytic unprocessing. Both paired
children receive the same frozen checkpoint specification and camera/ISP config;
execution is sequential and each child releases working arrays before the next.
Pair preflight verifies checkpoint integrity before creating outputs. The pair
manifest retains `shared_checkpoint` evidence and explicitly requires those
checkpoint bytes in addition to the analytic config for replay. Each child
verifies and loads the checkpoint independently; a failure never silently falls
back or creates a pair completion marker. Real inference remains Colab-only.
The notebook now dispatches this option, and scientific fidelity is not
established by successful output roundtrips.

For explicit field-mode selection, `fields.resolve_field(name, units, mode,
candidates)` selects from supplied `CameraField` records and returns a detached
record retaining the requested mode. AUTO prefers USER_SET, OBSERVED, DERIVED,
ESTIMATED, PRIOR_SAMPLED, then SYNTHETIC. ESTIMATE excludes synthetic/prior
fallbacks; FABRICATE accepts only priors/synthetic values; USER accepts only
USER_SET. Missing evidence returns null/UNSPECIFIED. Candidate values must still
pass the camera/ISP domain checks before execution. No estimator runs implicitly.
Identity fields require USER mode. For other unobservable metadata, callers must
set `unobservable_metadata=True`; synthetic/prior values then require
`fabricate_unobservable_metadata=True` (default false), even in AUTO. This API
is not yet wired into the analytic JSON configuration or a notebook panel.

`fields.resolve_camera(base_camera, modes, candidates)` resolves executable
camera inputs together and returns `(validated_camera, selection_records)`.
Modes are keyed by `CameraHypothesis` input names; evidence must use the units
in `base_camera.registry()`. Unspecified modes use AUTO and the base synthetic
hypothesis. Supplied evidence lists replace that field's base candidate.
Unresolved USER/ESTIMATE inputs fail, as do invalid domains and attempts to set
derived electronics. Use the returned camera in `AnalyticRunConfig(camera=...)`;
pass selection records through the runner keywords below to preserve provenance.
JSON/panel integration remains pending.

`fields.resolve_isp(base_isp, modes, candidates)` applies the same atomic selection
contract to all four `AnalyticISP` inputs. Names are unprefixed; white balance and
exposure gain use `ratio` units, matrix and tone switch use null units. Missing
evidence, unknown inputs, invalid matrices/gains and nonboolean tone switches
fail before returning a hypothesis. Use the result in `AnalyticRunConfig(isp=...)`.
Pass selection records through the runner keywords below to preserve provenance.
This does not perform WB/colour/exposure estimation or wire field modes into
JSON/the notebook panel.


To preserve selection evidence in executed bundles, pass resolver records to
`config.run(source, output, camera_selections=camera_records,
isp_selections=isp_records)` (also supported by `run_analytic`). The runner
checks field names, units, mode eligibility and equality with effective camera/
ISP values before source I/O. Conflicting, duplicate, unresolved or non-input
records fail. Accepted selections replace input provenance in the manifest and
HTML; derived records remain derived. Supply evidence only for supported input
fields. This retains caller-supplied claims, not independently verified estimates.
`config.json` still stores effective hypotheses only: replaying it alone does
not recreate selection provenance. Retain the manifest for that evidence.
JSON field-mode requests and the notebook panel remain pending.


The notebook catalogue now executes CFA pattern, ADC depth, black level,
full-well capacity, ISO, shutter time, aperture and seed selections through
`panel.resolve_camera_panel`. Explicit values require USER mode. AUTO and
FABRICATE without values use declared synthetic camera hypotheses; ESTIMATE
fails without evidence. Other catalogue overrides remain unsupported and fail.
The resolved immutable camera and selection records are reused in both paired
runs, with mode/provenance retained in each child manifest and HTML report.
Pair selection validation occurs before creating output directories. This does
not infer camera measurements or implement learned, sensor or optics models.

The notebook also executes `AsShotNeutral` catalogue overrides through
`panel.resolve_analytic_panel`: set its mode to `USER` and supply three positive
values with green equal to 1 (for example `[0.5, 1, 0.25]`). Their reciprocals
become analytic WB gains, affect camera RGB before CFA/noise, and drive the DNG
neutral. Both paired manifests retain USER selection provenance. AUTO/FABRICATE
without a value retain the explicit ISP hypothesis; ESTIMATE requires evidence
and currently fails. ColourMatrix1/2 are DNG matrices, not camera-to-sRGB inputs;
those catalogue overrides still fail rather than being incorrectly mapped.

Fidelity increment: `prd_v5.fidelity.SRCheckpoint` accepts a user-provided HAT
or Real-ESRGAN x4 checkpoint, expected SHA-256, full repository revision, source,
licence and explicit acceptance. `fidelity.reconstruct(source_rgb, output_path,
spec)` is a Colab CUDA-only SR API producing a disk-backed float32 encoded-sRGB
NPY at x4 with 512/64 halo tiles. It verifies bytes and Spandrel architecture,
scale and RGB channels before inference, uses safe tensor-only loading and
reports any OOM tile reduction. There is no download or implicit fallback.
Install the package's `sr` extra in Colab to supply torch/Spandrel. This API is
connected to optional notebook paired RAW execution. Actual checkpoint loading,
full-scale GPU execution, seam/fidelity acceptance and scientific metrics remain
unvalidated; local tests use explicit model doubles only.

Opt-in single-run SR integration: pass `sr_checkpoint=SRCheckpoint(...)` to
`prd_v5.pipeline.run_analytic` in Colab with a verified, accepted user checkpoint.
Its disk-backed x4 output feeds the normal geometry and analytic RAW stages.
Failures stop execution without an analytic fallback. The manifest retains
checkpoint evidence; `config.json` alone replays only analytic settings, so SR
replay also needs the same checkpoint and explicit argument. Paired/notebook SR
dispatch is available with explicit checkpoint provenance. Local tests use tiny model doubles only; real model
quality and full-scale execution are not validated.

Notebook SR: set `SR_CHECKPOINT` to a dictionary containing all required
`SRCheckpoint` fields and select matching `SR_BACKEND` (`HAT` or `REAL_ESRGAN`).
Preflight verifies integrity and explicit licence acceptance without tensor loading.
The execution cell installs the optional `[sr]` dependencies in Colab and passes
one shared checkpoint to the sequential pair. `ALLOW_ANALYTIC_BASELINE=True` is
still required because inverse ISP remains analytic. Missing weights require
that explicit baseline opt-in; invalid supplied weights always fail. No model
is downloaded and no backend is silently substituted. SR tile/overlap edits
beyond 512/64 currently fail explicitly. Preflight records readiness rather than
execution; only a completed pair updates SR effective state to executed. Actual
checkpoint compatibility, memory use and scientific fidelity remain Colab-only
unvalidated work.

Paired runs now record full-resolution, unaligned developed-TIFF disagreement
in `pair_manifest.json`: normalized MAE, RMSE, maximum error, channel bias,
PSNR and per-render saturation fractions. The comparator memory-maps uint16 RGB
TIFFs and reads row strips. It includes the whole visible raster, including
sensor margins, without applying DefaultCrop or fitting exposure. Unequal shapes
skip metrics explicitly. Identical samples have null PSNR with an infinite-PSNR
explanation so JSON remains finite. These metrics measure render disagreement,
not recovered detail or scientific fidelity; equal shapes do not prove scene
identity or registration. Alignment and benchmark-reference evaluation remain
pending. A comparison read failure prevents pair completion publication.

Joint exposure hypotheses can be selected before constructing the shared run
configuration, without decoding images or allocating frame buffers:

```python
from prd_v5.exposure import resolve_exposure

camera, camera_selections, exposure_report = resolve_exposure(
    CameraHypothesis(), scene_ev100=4.0, candidate_count=8
)
config = AnalyticRunConfig(camera=camera)
# Pass camera_selections to config.run or run_analytic_pair.
```

This reuses the Sony-like standard-stop envelope and heuristic joint ranking.
The caller supplies the scene EV100 prior; weights are relative support among
retained candidates, not calibrated probabilities or inferred capture truth.
Selected ISO drives gain/noise. Shutter/aperture remain metadata hypotheses and
leave analytic ISP exposure gain unchanged. The registry retains joint alternatives
and the EV prior. Automatic image metering and notebook posterior controls remain
pending; existing defaults and panel selections are unchanged.
