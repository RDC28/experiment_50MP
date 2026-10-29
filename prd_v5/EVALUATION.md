# PRD v5 evaluation and build plan

Evaluated 2026-10-29 before implementation. `PRD_v5.md` is the supplied document copied byte for byte. This evaluation does not rewrite it. The deliverable is one new Colab notebook under `prd_v5/notebooks/`, backed by composable, tested Python. No notebook, model download, full-frame inference, or training is permitted on the 8 GB development Mac. Full-suite tests must use `/Users/rohit/Projects/firstmate/data/exp50mp-heavy-lock.sh`.

## Implementation evidence: initial Colab deliverable (iteration 28)

`notebooks/synthetic_50mp_v5.ipynb` is the single new v5 notebook. It contains
Colab-only setup, hardware/RAM/disk inspection, saved requested/effective reports,
the complete PRD boolean panel with full-scale defaults and all parameter-table
entries with modes, priors and unresolved provenance. Supported camera/ISP inputs
feed one immutable configuration into the executable sequential paired runner.
The baseline requires explicit opt-in; strict learned mode fails. Unsupported
catalogue overrides fail rather than silently alter or fail to alter pixels.

This is an initial executable MVP notebook, not complete PRD implementation.
HAT/HYBRID/learned HDR and extended sensor/optics requests remain explicitly
skipped pending ordered integrations. Catalogue field modes beyond the supported
hypotheses, source diagnostic plots, scientific comparison/editability metrics,
EXR and final metadata coverage remain pending. Full-scale execution, peak memory,
GPU compatibility and scientific acceptance require Colab validation. No notebook
cells, inference, downloads or full-scale work ran locally. Static notebook format,
Python syntax, defaults, parameter coverage and hypothesis contracts are tested;
215 existing/new lightweight tests passed under the shared lock before the final
parameter-coverage test was added. Repository-wide ruff passes.

## Implementation evidence: twenty-seventh MVP increment

The analytic runner and `AnalyticRunConfig.run` now accept camera/ISP selection
records from the field resolvers. Before source I/O or output creation, records
must match executable input names, units, field-mode provenance eligibility and
the effective validated hypotheses. Duplicates, unresolved evidence, derived or
identity overrides and inconsistent values fail. Accepted records replace only
the corresponding input registry entries in the manifest and offline HTML;
derived electronics and DNG records retain derived provenance. Defaults and JSON
schema remain unchanged. Saved config replays effective values; selection claims
are retained in the manifest, not replayed as inferred evidence from config.json.

Validation: 213 lightweight v5 tests pass under the shared lock. Tiny actual
PNG-to-independent-LibRaw-TIFF output retains USER ISO and ESTIMATED exposure
records in JSON and HTML alongside matching effective settings. Invalid claims
fail before even opening a nonexistent source. Repository-wide ruff and changed
Python formatting pass. JSON/panel selection integration, complete MVP notebook,
later ordered phases and full-scale Colab/scientific validation remain pending.
No model, full-frame processing, notebook, commit or background service ran.

## Implementation evidence: twenty-sixth MVP increment

`fields.resolve_isp` now resolves all four executable analytic ISP inputs from
supplied field evidence before constructing a validated hypothesis. Base AUTO
hypotheses remain explicit; candidate lists replace them, and missing evidence
fails without fallback. Gain units match the analytic registry. Matrix
conditioning, positive gains and boolean tone controls remain executable gates.
Selection records retain modes/provenance separately; JSON/panel and runner
selection-provenance integration remain pending. No estimator is implied.

Validation: 206 lightweight v5 tests pass under the shared lock, including a
selected estimated exposure gain changing tiny disk-backed unprocessing by the
independently calculated factor, with passing forward reconstruction. Tests also
cover detached WB evidence, missing values, invalid domains/units and unknown
inputs. Repository-wide ruff and changed Python formatting pass. Full-scale
Colab validation, remaining MVP panel/notebook and later ordered phases remain
unresolved. No model, notebook, full-frame job, commit or background service ran.

## Implementation evidence: twenty-fifth MVP increment

`fields.resolve_camera` bridges supplied field evidence to a validated frozen
camera hypothesis. It resolves every executable camera input atomically, retains
selection records and recomputes coupled gain, saturation and NoiseProfile.
Unspecified modes preserve explicit base hypotheses through AUTO. Explicit
candidate lists replace base evidence; missing USER/ESTIMATE values fail without
fallback. Unknown/derived inputs, mismatched units and invalid physical domains
are rejected. No estimator or sampler is implied. Selection records remain a
separate return value; runner provenance and JSON/panel integration are pending.

Validation: all 197 lightweight v5 tests pass under the shared lock, including
resolved ISO driving a tiny disk-backed Bayer/ADC output, missing-evidence
rejection, domain checks and preserved base state. Repository-wide ruff and
changed Python formatting pass. Full-scale Colab validation, complete MVP panel
and notebook, and later ordered phases remain unresolved. No model, notebook,
full-frame job, commit or background service ran.

## Implementation evidence: twenty-fourth MVP increment

`fields.resolve_field` now executes all four PRD field modes against supplied
`CameraField` evidence. AUTO uses a declared precedence, ESTIMATE excludes priors,
FABRICATE selects declared prior/synthetic candidates, and USER selects only user
input. Unavailable evidence yields null/UNSPECIFIED without invented confidence.
Identity fields require USER mode; fields declared unobservable metadata require
the explicit fabrication gate for synthetic/prior values, including in AUTO.
Duplicate provenance and mismatched candidates fail rather than depend on order.
Returned nested values are detached. Candidates still require domain validation;
this API does not estimate, sample, or alter the current runner/config defaults.
Panel wiring and complete field catalogue remain pending MVP work.

Validation: all 189 lightweight v5 tests pass under the shared lock, including
resolved ISO driving actual gain/saturation, missing estimates, privacy gates,
metadata fabrication policy and detached uncertainty records. Repository-wide
ruff and changed Python formatting pass. Later ordered phases and full-scale
Colab/scientific validation remain unresolved. No model, notebook, full-frame
job, commit or background service ran.

## Implementation evidence: twenty-third MVP increment

`paired.run_analytic_pair` executes caller-labelled 2K and 4K sources sequentially
using one frozen analytic configuration, including actual camera/ISP hypotheses
and stream seeds. Working arrays close between children. A pair completion
manifest links both child manifests with SHA-256, observed source dimensions and
output dimensions; a failed child leaves no pair completion marker. Existing
directories are protected. Labels do not establish scene identity, registration
or resolution, and shared streams do not guarantee equal noise residuals.
Scientific comparison metrics remain pending.

Validation: tiny 16/32-pixel sources execute both independent DNG/TIFF roundtrips,
retain identical configuration and reproduce identical DNGs for constant equal
signals. Failure coverage verifies sequential completion, identical config object
use, released working arrays and retention of the first completed bundle without
pair completion. All 171 lightweight v5 tests pass under the shared lock;
repository-wide ruff and changed Python formatting pass. No full-size job, model,
notebook or background service ran. Complete panel/registry and notebook remain
pending before later phases; full-scale paired Colab validation remains unresolved.

## Implementation evidence: twenty-second MVP increment

The analytic registry now includes `dng.*` intended tag records derived directly
from the writer's metadata builder, including colour matrices, AsShotNeutral,
baseline exposure, orientation, synthetic identity and crop. These are derived
from synthetic hypotheses, not observed capture metadata or decoder evidence.
Disabled shot noise retains an explicit null/UNSPECIFIED NoiseProfile record
while the incompatible tag remains absent from the DNG.

Validation: tiny DNG tests compare registry values with serialized profile,
identity, geometry and orientation tags, allowing rational rounding, and exercise
both enabled and disabled shot noise. All 169 lightweight v5 tests pass under
the shared lock; repository-wide ruff passes. Full-scale Colab, complete field-mode
configuration, notebook and later ordered phases remain pending. No model,
full-frame job or background service was started.

## Implementation evidence: twenty-first MVP increment

The distribution now includes the additive `prd_v5` package and its analytic
configuration JSON schema, preserving both legacy installed packages. Previously
the explicit setuptools package list omitted all v5 modules. The schema is
available through `importlib.resources` without a source checkout.

Validation: built a wheel and installed it without dependencies into an external
temporary target. All seven pipeline tests passed against that installed package
outside the checkout, including tiny PNG-to-DNG-to-independent-TIFF execution,
hashes, replay and failure cleanup. The installed schema passes meta-validation
and validates the default saved configuration. All 167 lightweight v5 tests pass
under the shared lock. No full-size processing, model, notebook, commit or
background service ran. Complete MVP panel/registry and notebook remain pending
before later ordered phases; full-scale Colab and scientific validation remain
unresolved.

## Scope and interpretation

Build the supplied 50 MP experiment. The earlier request mentioned 100/200 MP outputs and diffusion/transformer training from scratch; this document instead requests 50 MP, existing pretrained models and analytic physical simulation. That is a scope divergence, not permission to implement the earlier request.

Maximum settings means full 50 MP, HAT preferred, hybrid inverse ISP, HDR and learned residual requested, full sensor and optics controls exposed and requested, all deliverables enabled, no fast-development preset. It does not mean maximizing noise, ISO, distortion or invented detail, nor allocating the largest possible GPU tile. Retain the PRD's 512/64 SR tiles, 1024/128 RAW tiles, mixed precision and memmap accumulation as quality/resource defaults. Preserve Sony-like 14-bit ADC in a 16-bit storage container. Preserve aspect ratio by default. Identity-sensitive GPS/serials stay user-only, unobservable metadata fabrication remains false. Expose every scalar, profile, coefficient, uncertainty/prior, field mode and experimental switch through the panel or an editable structured configuration section.

The PRD's all-features request conflicts with its own final rule that unvalidated features be experimental and disabled, and with zero optics/disabled sensor defaults. Resolve operationally by recording requested versus effective state separately. Request the full feature set; mark unavailable dependencies as skipped, zero coefficients as enabled but no-op, and experimental effects explicitly experimental. Never invent checkpoints, silently swap HAT for Lanczos, or claim that exposing a switch implements its algorithm. In strict learned-only mode an unavailable model must fail clearly; hybrid may retain the analytic branch with a visible skip. Licence acceptance is separate from enabling a feature.

## Feasibility and resource budget

The last supplied runtime was A100 40 GB with approximately 89 GB host RAM; inspect actual GPU/VRAM/RAM/disk each run. [Colab resource FAQ](https://research.google.com/colaboratory/faq.html) explicitly makes resource availability variable. Its session envelopes are limits, not throughput guarantees.

At 50 million photosites, uint16 Bayer is 100 MB, float32 Bayer 200 MB, float64 Bayer 400 MB, float32 RGB 600 MB, float64 RGB 1.2 GB, uint16 RGB 300 MB, and a float32 mask 200 MB (decimal units). Several simultaneous RGB intermediates and masks can consume 5-15 GB before decoder/model memory. The existing pipeline uses float64 RGB electron intermediates; v5 should mosaic before stochastic sensor simulation as its roadmap specifies, reducing three-channel allocations. Stage arrays must be disk-backed, flushed, and released. Process 2K and 4K runs sequentially with one immutable camera hypothesis and stable random streams. Reserve roughly 15-30 GB free disk for two runs plus reports and packaging, more if every intermediate and editability render is retained. Measure actual RSS, peak CUDA allocation and stage times; these are estimates.

| Phase | Colab feasibility | Resource/runtime qualification |
| --- | --- | --- |
| MVP analytic, Bayer, noise, DNG, TIFF | High on CPU or GPU runtime | Expect minutes for full-frame CPU transforms/LibRaw/I/O; no measured timing yet. Small executable fixtures can verify semantics locally. |
| Fidelity HAT, HDR, comparison | Medium/high on A100 | Tiled inference only; 512 input pixels at x4 produces 2048 output pixels per tile, with substantial attention activations. GPU OOM must reduce tiles visibly, never resolution silently. Tens of minutes is a planning allowance, not a benchmark. |
| Learned RAW | Experimental | Patch-based camera adaptation and dependencies may require training and camera-specific weights. A100 does not prove arbitrary checkpoint compatibility; diffusion can exceed a session. No 4-15 minute guarantee. |
| Full sensor | High as simulation | Seeded strip/CFA operations; memory and time grow linearly. Calibration truth is absent. |
| Full optics | Medium/high | Bounded strips with correct halos/global coordinates; large spatial PSFs and motion kernels increase time. Avoid full-frame coordinate grids and discontinuous per-tile fields. |
| Metadata and validation | High for software; conditional for science | Full-resolution LibRaw development/edit sweeps and two-input runs increase I/O and memory. Ground-truth metrics require user-provided licensed data. |

For a 4K source, an x4 model's intermediate can exceed 100 MP even though the target is 50 MP. Accumulate/resample bounded tiles directly to the planned output where possible. Two input resolutions must use equivalent transforms, not different implicit SR backends. A100 40 GB is plausible for tiled inference, not whole-frame attention or training from scratch. Budget hours for the first dependency/benchmark session and capture empirical timings before promising runtime.

## Verified upstream availability and licences

Checked official repository pages, READMEs and GitHub licence endpoints, without downloading weights. A README download link is evidence of advertised availability only; it is not a verified checkpoint hash or a proven runnable camera adapter. Code licences do not automatically establish dataset/checkpoint rights. Record repository revision, checkpoint source, SHA-256, architecture, camera domain and licence acceptance in the runtime manifest.

| Component | Actual source and licence | Availability and integration decision |
| --- | --- | --- |
| RAW-Diffusion | [SonyResearch/RAW-Diffusion](https://github.com/SonyResearch/RAW-Diffusion), Apache-2.0 | Public training/sampling code for FiveK Canon/Nikon and NOD Sony/Nikon. README documents training rather than a verified ready-made alpha1 II checkpoint. Treat pretrained compatibility as unverified; supply an explicit adapter contract and skip reason until a compatible pinned model is supplied. |
| SpiralDiff / CamLoRA | [Chuancy-TJU/SpiralDiff](https://github.com/Chuancy-TJU/SpiralDiff), [LICENSE](https://github.com/Chuancy-TJU/SpiralDiff/blob/main/LICENSE) contains MIT | README/badge says Apache-2.0, contradicting the actual MIT file and the PRD. README advertises Hugging Face pretrained weights and processed data. Neither camera adaptation nor native 50 MP inference is established by that claim. Preserve licence discrepancy in manifest; do not auto-approve model terms. |
| RAW-Flow | [liuzhen03/RAW-Flow](https://github.com/liuzhen03/RAW-Flow), MIT | Official AAAI 2026 code. README links Google Drive weights for Nikon D700, Canon EOS 5D and PASCALRAW, about 350 MB each. Those are not calibrated Sony alpha1 II models. Adapter must verify exact format, normalization and camera conditioning before use. |
| HAT / HAT-L / Real-HAT | [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT), Apache-2.0 | Official Google Drive/Baidu model links and tile example are present. HAT-L ImageNet-pretrained x4 is a sensible maximum-fidelity selection; Real-HAT GAN/sharper variants are perceptual alternatives, not evidence of greater truth. Existing repo HAT is unpinned, so provide user checkpoint + expected digest or a verifiable upstream pin, never a made-up hash. |
| Real-ESRGAN | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), BSD-3-Clause | Official x4plus release checkpoint and tiled inference exist. Existing `synthraw/model_manifest.py` already pins URL/revision/digest. Reuse its verified download policy. Report an explicit fallback if selected. |
| CycleISP | [swz30/CycleISP](https://github.com/swz30/CycleISP), [Academic Public License](https://github.com/swz30/CycleISP/blob/master/LICENSE.md) | Noncommercial restriction, attribution and same-licence distribution conditions. Public synthesis and denoising code/models exist, but an older environment and task-specific weights are not turnkey Sony calibration. Reference/comparison only, no automatic vendoring. |
| Brooks unprocessing | [google-research/unprocessing](https://github.com/google-research/google-research/tree/master/unprocessing), repository Apache-2.0 | Public analytic implementation; does not require neural weights for inversion. Reimplement equations using existing tested colour/ISP code. |
| ELD | [Vandermode/ELD](https://github.com/Vandermode/ELD) | Physical-noise research reference. Inspect the selected revision's licence before code reuse; no ELD checkpoint required by this build. A paper reference is not a licence grant. |
| Learned ITM / AIM 2025 | PRD selects no concrete repository/checkpoint | No model availability/licence can be verified without selecting one. Implement analytic inverse tone with clipping uncertainty; explicitly defer learned ITM. |
| NTIRE reverse ISP | [challenge report](https://openaccess.thecvf.com/CVPR2025_workshops/NTIRE) | Benchmark context, not a selected compatible pretrained model. Do not invent a backend from a challenge name. |
| DNG SDK | [Adobe official SDK and specification](https://www.adobe.com/support/downloads/dng/dng_sdk.html) | Official page confirms 1.7.1 build 2768, 2026-10-13, and reading/writing support. Adobe SDK terms apply, not generic open-source assumptions. Existing tifffile CFA writer plus independent LibRaw is the direct implementation; SDK conformance is additional Colab work. |
| rawpy / LibRaw | [rawpy](https://github.com/letmaik/rawpy), [LibRaw](https://www.libraw.org/) | Independent decoder/developer, not a reconstruction checkpoint. Respect rawpy MIT and LibRaw LGPL/CDDL build terms. Reopening alone is not full Adobe conformance. |

Existing ancillary backends (SwinIR, SUPIR, GeoCalib, VID2HDR, EXPANDDIFF, spectral reconstruction) are not required new v5 dependencies. Keep existing manifests intact; do not quietly activate them as replacements for specified models. LPIPS is an optional [official metric](https://github.com/richzhang/PerceptualSimilarity), with its own weights/dependency provenance; missing LPIPS must yield not measured, never zero error.

## Benchmark data

[FiveK](https://data.csail.mit.edu/graphics/fivek/) supplies real DNG and retouchings with file-specific licence files; preserve its Adobe/MIT file lists and inspect terms before redistribution. [RAISE](https://loki.disi.unitn.it/RAISE/) describes 8,156 camera-native RAW images and separate Nikon D7000 flat fields, requests citation, and is not automatically an unrestricted commercial training licence. [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) provides Sony alpha7S II and Fuji low-light data; its MIT code licence does not establish blanket rights to every dataset file. [Zurich RAW-to-RGB/PyNET](https://github.com/aiff22/PyNET) is phone RAW paired with DSLR rendering, not identical-scene full-frame RAW truth. Check actual download terms and alignment before use. No datasets or ARW files enter Git. Support user-provided local benchmark paths and licence records; absent benchmarks remain not measured.

## Acceptance gates and scientific limits

| Gate | Executable evidence and limit |
| --- | --- |
| DNG parse/CFA | Tiny generated files reopen through rawpy; check raw dimensions, Bayer phase, samples, black/white levels, crop/orientation. Colab must repeat on every full-size output. |
| Metadata coherence | Registry and DNG/manifest compare values with explicit rational rounding tolerance and tag mapping. Not every registry concept has a DNG tag; preserve JSON records and distinguish serialization from a populated standard tag. |
| Seams <=1 DN equivalent | Evaluate overlap disagreement on clean camera-space output before noise; use actual ADC scale. Nonlinear model/context changes may violate it, so record measured maximum and fail the gate. A smoothed preview cannot certify this. |
| 2K/4K | Same sampled hypothesis, geometry, camera/body maps and random streams; full-resolution aligned windows, pre/post-noise RAW metrics, spectra and source reconstruction. Upsampling one input twice is not independent input evidence. |
| SR value | Real held-out RAW-derived pairs and Lanczos baseline, optional LPIPS. No genuine truth means not measured, never a claim that added texture equals improvement. |
| HDR | Clipped-source uncertainty and generated-area masks enforce no recovered-truth claim; highlight/shadow numerical accuracy needs real HDR truth. |
| Noise | Flat and dark sweeps measure variance slope/intercept versus model with declared tolerances. Include ADC rounding/clipping and fixed versus temporal terms. These prove self-consistency, not measured Sony physics. |
| Colour | Matrix forward-render consistency can run without truth. DeltaE2000/WB angular accuracy require aligned colour-calibrated truth and camera/profile pairs. |
| Provenance | Every field has value/units/mode/provenance/confidence/posterior/method/validation, even null unknowns. Check dependency coherence and forbidden fabricated identity. |
| No silent no-op | Every boolean has requested/effective state and reason, using enabled/disabled/skipped dependency/enabled but no-op statuses. Experimental does not equal accepted. |
| Outputs/editability | DNG, independently developed uint16 TIFF, hashes, JSON/HTML, masks and comparison reports; EV -3,-2,0,+2,+3 and WB/demosaic/highlight/shadow sweeps record behavior. Plausibility still needs visual review. |

Pixel-accurate original RAW, real serial/GPS/history, actual spectral sensitivities and unique ISO/shutter/aperture cannot be established from rendered RGB. Missing truth must not turn into a passing score. Validation status should distinguish passed, failed, skipped and not measured, with overall scientific acceptance conditional when required truth is absent.

## Ambiguities, contradictions and lessons

The preserved PRD contains ChatGPT citation residue (`turn...search...`, `turn...academia...`, `filecite`/`turn0file0` markers). Those opaque citations resolve to no bibliography or attached source in this repository. The links in this evaluation are independently checked replacements for feasibility decisions, not edits to the PRD.

There is no exact even raster of precisely 50 million pixels for every aspect. Report actual rounded dimensions/MP and distinguish Sony native 49.7664 MP from marketing 50.1 MP. The virtual aspect-preserving geometry does not retain exactly the same physical pixel pitch as native Sony on the same sensor mm. Native letterbox sensor margins and source-supported crop must remain explicit.

Existing `src/synthraw` has geometry, profiles, colour, CFA, ADC, sensor, optics, DNG, rawpy, tiled SR, provenance, comparison and reports; reuse stable APIs without editing earlier notebooks/defaults. Its D/I/F/M provenance lacks v5's richer schema and field modes, so add a v5 registry rather than relabeling facts. Its sensor stage currently precedes CFA and uses RGB float64 electron buffers; v5 should use the requested CFA-first order. Its learned stubs/manifests do not establish working HAT/RAW inference.

Read-only lessons from `fm/exp50mp-prd2-realism-fixes:docs/reference-arw-comparison.md`: one ILCE-1M2 sample has raw 8672x5784, visible 8640x5760, RGGB and black 512. Camera-space WB and a real matrix matter; a virtual sRGB matrix is not a calibrated Sony model. Empty reserved HDR headroom made earlier RAW too dark. Single-frame noise estimates are scene-contaminated ceilings, not full calibration. Do not import that branch or its ARW. Full-resolution crops are necessary because earlier heavily downsampled comparison hid high-frequency differences; generated scene extension must be distinct from SR detail. Keep unaligned round-trip metrics next to aligned ones and report alignment gain.

## Concrete phase-by-phase build plan

1. MVP: add v5 package/config/registry with every parameter represented and validated, immutable seeded camera hypothesis, source/ICC/hash inspector and geometry planner. Implement small executable end-to-end RGB -> analytic inverse ISP -> optics interface -> Bayer -> electrons/noise -> ADC -> DNG -> LibRaw TIFF -> registry/report/hash bundle. Add one Colab notebook with setup, hardware, full config and readable stage cells. Use tiny fixtures in tests, no local notebooks or inference.
2. Fidelity: add tiled HAT integration accepting an explicitly verified compatible checkpoint, explicit Real-ESRGAN/Lanczos selection and fallback reporting, bounded memmaps, HDR uncertainty, joint exposure posterior, WB/profile selection, genuine 2K/4K inputs and full-resolution metrics/benchmark adapter. Preserve identical hypotheses and seeds. Test with lightweight model doubles plus real analytic small fixtures; defer performance claims to Colab.
3. Learned RAW: implement camera-aware residual adapter contract, confidence gating and analytic/hybrid/learned dispatch. Support only actual compatible model outputs with hashes, normalization and camera-domain validation. Unavailable upstream adapters/checkpoints must be visibly deferred/skipped, not identity functions presented as working diffusion. Keep physical stochastic noise outside the network.
4. Full sensor: CFA-domain PRNU/DSNU/body maps, row/column FPN, temperature/exposure dark current, defects, nonlinear response and dual conversion gain with coherent NoiseProfile limitations. Add flat/dark, code, defect, deterministic stream and gain-transition tests. Priors remain synthetic without calibration.
5. Full optics: pre-CFA radial/tangential warps, vignetting, lateral/axial CA, spatial PSF, diffraction/defocus/motion with nonzero-coefficient reporting and bounded halo handling. Expose all coefficients and uncertainty; test observable effects and zero/no-op reporting.
6. Final metadata: complete every registry field and standard-tag/manifest mapping, user-only identity, checkpoint/source/repro hashes, all deliverables and acceptance status with not-measured truth gates. Validate notebook JSON and Colab setup/config semantics, run ruff and full suite through the shared lock, review requested/effective feature coverage. Update this evaluation with actual implemented/deferred evidence before the committed handoff. No 50 MP run is claimed until a separate Colab execution supplies evidence.

## Implementation evidence: first MVP increment

`prd_v5/hypothesis.py` now provides a frozen camera hypothesis, local seeded
electronics-prior sampling, independent stage seeds, and the initial JSON field
schema. Conversion gain, ISO analogue gain, electron saturation and temporal DNG
NoiseProfile derive from the same state. Shot-noise-off returns no NoiseProfile
rather than inventing a positive slope. Unknown identity remains null/user-only.
This registry is explicitly partial and does not yet implement field selection,
exposure inference, or a complete notebook configuration.

Validation: 26 lightweight tests in `prd_v5/tests/test_hypothesis.py` passed;
repository-wide `python -m ruff check .` passed. Tests verify arithmetic variance
against the Poisson/read/rounding equation, deterministic sampling without global
RNG mutation, immutable shared state, JSON records and invalid parameter rejection.
No full suite, image pipeline, decoder roundtrip, model, notebook or full-frame
job ran. The remaining MVP and all subsequent phases remain pending; Colab-only
validation and the unavailable research integrations above remain unresolved.


## Implementation evidence: second MVP increment

`prd_v5/geometry.py` implements allocation-free geometry planning with a 50.0 MP
aspect-preserving target, even Bayer dimensions, native 8640x5760 dimensions,
uniform source-to-output transforms, supported source crops, sensor-margin
bands and DNG crop/ActiveArea coordinates. Requested and actual megapixels are
separate, and buffer payload estimates are explicit rather than peak RAM claims.
Native crop and outpaint policies are represented; outpaint bands have status
`required_not_executed`, never a claim that a generator ran. Native center-cover
may crop vertically for portrait sources despite the PRD policy name
`CROP_HORIZONTAL`; portrait outpainting may require side bands.

Validation: 20 new lightweight geometry tests pass, including equivalent 2K/4K
mapping, area accounting, CFA crop support, JSON serialization and invalid input
rejection. All 46 v5 camera/geometry tests and repository-wide ruff pass. Existing
pipeline defaults are unchanged. No images, models, notebooks or full-suite tests
ran. Rasterization, bounded pixel processing, DNG/rawpy/TIFF roundtrip, complete
configuration/registry and the Colab notebook remain pending within MVP; all
later phases and Colab-only scientific/performance checks remain pending.

## Implementation evidence: MVP source headers and identity

`prd_v5/inputs.py` provides bounded 1 MiB streaming file hashing and JPEG/PNG
header inspection without pixel decode. Source records contain dimensions,
encoded precision, byte count, ICC digest/description, EXIF presence and an
explicit colour policy. Unprofiled RGB remains unresolved unless the caller
explicitly assumes sRGB. Embedded ICC always requires a transform; malformed
profiles are blocked. Alpha, non-RGB modes and 16-bit PNG are rejected until
explicit compositing/precision-preserving decoding exists. No private EXIF
values are copied. Header inspection does not certify file integrity, clipping,
scene radiance or camera metadata, and is not an implemented ICC transform.

Validation: all 56 lightweight v5 tests passed, including 10 new source tests
for JPEG/PNG identity, JSON records, valid/invalid ICC, unsupported modes,
explicit assumptions, no pixel decode and private EXIF omission.
Repository-wide ruff passed. Python formatting is checked separately from the
immutable PRD. Source decode/diagnostics and full-scale Colab validation remain
pending, along with the rest of the MVP and later phases.

## Implementation evidence: MVP bounded analytic unprocessing

`prd_v5/analytic.py` adds an executable strip-based clean camera-RGB stage with
piecewise sRGB inverse transfer, optional inverse smoothstep tone, inverse colour
matrix, white-balance removal and exposure-gain removal. Parameters are immutable
synthetic/user hypotheses; the identity default is virtual sRGB, never Sony
calibration. Output is a caller-owned float32 memmap, flushed after processing;
intermediates scale with strip height rather than frame height. Input may also
be disk-backed. Invalid/nonfinite pixels, ill-conditioned matrices, invalid gains
and overlapping buffers are rejected. A failure may leave partial output, which
the caller must discard. Reports count clipped-source pixels and floored negative
camera samples, explicitly leaving highlight recovery and radiance accuracy
unmeasured. This API requires explicitly sRGB-encoded normalized pixels; it does
not implement source decoding, ICC conversion, exposure inference or HDR.

Validation: 13 new lightweight tests cover exact transfer anchors, a nontrivial
matrix/WB/exposure/tone forward-inverse roundtrip, strip-size invariance, immutable
parameters, clipping/gamut reports and invalid parameter/pixel/buffer contracts.
All 69 v5 tests pass, repository-wide ruff passes and new Python files are
formatted. NumPy was added only to the external lightweight dev environment.
No full suite, full-frame processing, notebook, model or background service ran.
Remaining MVP work includes source decode/ICC, geometry rasterization, CFA-first
sensor/ADC, independent DNG/TIFF roundtrip, configuration, reports and notebook.
All subsequent phases and Colab-only scientific validation remain pending.

### Iteration 5: MVP Bayer and temporal sensor

`sensor.py` samples the selected Bayer phase before converting clean camera RGB
into electrons, applying Poisson shot noise and independent Gaussian read noise,
then analogue gain, ADC rounding, pedestal and clipping into a uint16 memmap.
Unity clean signal denotes base-ISO full well; ISO gain changes saturation using
the same immutable hypothesis that supplies NoiseProfile. Working memory is
bounded by strip height and row width. Absolute sensor row/stage seeds make
outputs independent of strip partitioning. Two source-resolution runs must use
the same final geometry and hypothesis; differing Poisson means do not imply
identical residuals. This is not a claim of calibrated spatial sensor maps.
NumPy version is recorded for RNG reproducibility. Full sensor effects remain
pending in their ordered phase. Caller discards partial output on failure.

Validation: 15 additional lightweight cases cover all Bayer layouts on odd
geometry, a tiny analytic-to-ADC path, strip invariance, gain/clipping, invalid
buffers/pixels, zero-effect read noise and flat/dark variance including ADC
quantization. All 84 v5 tests pass. These are synthetic self-consistency tests,
not scientific calibration or a full-scale/Colab benchmark. Remaining MVP work
includes source decoding/ICC, geometry rasterization, DNG/TIFF and independent
rawpy validation, configuration, reports and the notebook.


## Implementation evidence: sixth MVP increment

`prd_v5/raster.py` applies the geometry affine transform with pixel-center
bilinear sampling into float32 memmaps, using only row-sized working arrays.
A boolean memmap marks source support; sensor margins have zero encoded signal
and false support. Required outpainting raises before output writes. This
explicit encoded-sRGB baseline neither executes nor substitutes for HAT.
The input must already have an explicit normalized sRGB colour policy.

Validation: seven additional cases cover identity, known interpolation values,
margins, unavailable generation, buffer contracts and invalid pixels. A tiny
raster-to-analytic-to-Bayer fixture checks ADC values independently against the
sRGB transfer equation and Bayer phase. All 91 v5 tests pass. No full suite,
notebook, inference or full-frame work ran. Source decoding/ICC, orchestration,
DNG/TIFF roundtrip, config, reports and notebook remain MVP work; scientific
fidelity and Colab-only validation remain unresolved.


## Implementation evidence: seventh MVP increment

`decode.py` connects actual JPEG/PNG files to normalized disk-backed sRGB
arrays. Unprofiled RGB requires explicit assumption; embedded RGB ICC profiles
use LittleCMS relative-colorimetric conversion without black-point compensation.
Invalid ICC is blocked. Reports retain source hashes, conversion policy, stored
raster orientation and endpoint counts as clipping indicators rather than truth.
Pillow buffers decoded source pixels; a configurable eight-byte-per-pixel gate
rejects over-budget inputs before loading. This estimates decoder storage, not
peak RSS. Additional strip buffers scale with strip height. ICC conversion is
8-bit; HDR/16-bit source precision and EXIF autorotation are not implemented.

Validation: seven new tiny tests cover JPEG/PNG values, ICC conversion, strip
invariance, policy/resource rejection before decoding and output contracts. The
existing analytic E2E fixture now starts from an actual PNG and runs decode,
geometry, inverse ISP and Bayer/ADC against independently calculated codes.
MVP orchestration, DNG/rawpy/TIFF roundtrip, config, reports and notebook remain
pending. No models, full-frame processing or Colab validation ran.

## Implementation evidence: eighth MVP increment

`prd_v5/dng.py` now bridges the immutable camera, geometry and analytic colour
hypothesis to the existing bounded uncompressed DNG writer without changing
legacy defaults. Tags preserve CFA phase, crop, 14-bit black/white levels in
uint16 storage, ISO/shutter/aperture and temporal NoiseProfile. A virtual D65
XYZ-to-camera matrix derives from the same camera-to-sRGB transform used in
unprocessing, and AsShotNeutral derives from its white balance. Both profile
illuminants are explicitly D65, with no dual-illuminant calibration claim.
Unknown private identity is not emitted. Structural validation checks all tags
and every stored sample in strips; independent rawpy and Adobe checks remain
explicitly not measured. The writer record format is a serialization bridge,
not completion of the rich v5 registry.

Validation: eight new cases cover four CFA layouts, colour-transform algebra,
shot-noise-off tag omission, ADC overflow rejection before output, and a tiny
analytic inverse-ISP -> CFA/electrons/ADC -> actual DNG sample check. All 106 v5
tests passed, repository-wide ruff passed, and changed Python formatting passed.
The external lightweight environment gained tifffile. No full suite, notebook,
model, full-frame processing or background process ran. Independent LibRaw/TIFF
roundtrip, config/orchestration/reporting and notebook remain MVP work before
advancing to fidelity. Full-scale and independent developer checks remain
Colab acceptance work.


## Implementation evidence: ninth MVP increment

`prd_v5/roundtrip.py` adds independent rawpy/LibRaw validation using the existing
strict sample/CFA/levels/WB/colour-matrix/dimension gate, plus orientation checking
and actual 16-bit sRGB TIFF development. TIFF samples reopen through tifffile
memmap and compare exactly in strips. Rendering explicitly selects linear
demosaicing, camera WB, sRGB transfer, no auto brightness and no rotation.
DefaultCrop remains DNG metadata; TIFF uses the LibRaw visible raster rather
than claiming application of the default crop. LibRaw buffers whole frames,
so a 64-byte/photosite admission estimate with a 512 MiB default blocks 50 MP
until a Colab caller explicitly supplies a larger budget. This estimate is not
a measured peak RSS bound. Failed callers must discard partial outputs.

Validation: five new tiny tests cover real LibRaw development for all four Bayer
patterns, JSON reports, mismatched sample rejection, memory admission and path
alias rejection. The analytic-to-ADC-to-DNG fixture also now develops and verifies
a TIFF independently. All 111 v5 tests pass; repository-wide ruff and changed
Python formatting pass. rawpy 0.27.1 was installed only in the external lightweight
environment. No full suite, model, notebook or full-frame job ran. Full-scale
Colab memory/decode checks, Adobe conformance and scientific colour accuracy
remain unmeasured. Configuration, orchestration/reports and the notebook remain
MVP work before fidelity.

## Implementation evidence: tenth MVP increment

`pipeline.run_analytic` provides a single executable source-to-DNG-to-independent
TIFF path using the existing bounded stages and one caller-supplied immutable
camera hypothesis. It requires a new output directory, admits decoder/LibRaw
budgets before allocation, closes and removes temporary memmaps on all paths,
and writes a completion manifest only after successful exports. The JSON carries
geometry, camera/ISP state, partial provenance registry, stage validation reports,
resource limits and streaming SHA-256 hashes of both deliverables. This is an
explicit bilinear analytic runner, not a claimed HAT/hybrid implementation or the
complete PRD configuration. Scientific acceptance remains not measured. Failed
runs can retain partial deliverables for diagnosis without a completion marker.

Validation: two new tests exercise an actual tiny PNG through independent TIFF,
JSON and hash verification, overwrite protection, preallocation memory admission
and injected-stage failure cleanup. All 113 v5 tests and repository-wide ruff
pass. No full suite, model, notebook, full-frame job or background service ran.
Complete configuration, packaged masks/reports and the notebook remain MVP work;
all later phases and full-scale Colab acceptance remain pending.


## Implementation evidence: eleventh MVP increment

The analytic runner now packages `source_support.npy` after successful independent
DNG/TIFF validation. Strip copying into a boolean NPY memmap retains the existing
geometry support without allocating a full-frame RAM mask. The completion manifest
records its hash, size, dtype, shape, supported count and full-sensor coordinate
semantics. False means an unsupported sensor margin; this mask makes no clipping,
learned-detail confidence or recovered-truth claim. Failed mask exports cannot
produce a completion manifest, and temporary stage arrays are still removed.

Validation: the tiny source-to-TIFF test reopens the packaged mask without pickle,
checks shape/dtype/support count and all artifact hashes. A new export-failure
test checks absence of completion and scratch arrays. All 114 v5 tests and
repository-wide ruff pass. No full suite, notebook, full-frame job, model or
background process ran. Remaining MVP work includes complete config/registry,
HTML reports, further uncertainty masks and the notebook. Later ordered phases
and full-scale Colab/scientific validation remain pending.


## Implementation evidence: twelfth MVP increment

The analytic bundle now retains `source_clipped.npy`, a boolean disk-backed
full-sensor mask identifying source-supported resampled sRGB pixels with any
channel at 1. Strip-sized comparisons and writes retain bounded working memory.
The completion manifest includes its hash, dimensions, count and coordinate and
uncertainty semantics. Unsupported sensor margins are false and must be read
alongside the support mask. This is an endpoint indicator after ICC conversion
and bilinear resampling: interpolation can dilute original endpoints, and
clipping below the endpoint is undetected. It neither recovers highlights nor
certifies unclipped radiance. Mask export failure still prevents completion.

Validation: two additional tiny PNG-to-independent-TIFF cases check exact mask
locations, single-channel and all-channel endpoints, the analytic clipping count
and strip-size invariance. All 116 v5 tests and repository-wide ruff pass; changed
Python formatting and diff checks pass. No full suite, model, notebook, full-frame
job or background service ran. Complete MVP configuration/registry, HTML reports
and the notebook remain pending before fidelity; all later ordered phases and
Colab-only scientific/full-scale validation remain unresolved.


## Implementation evidence: thirteenth MVP increment

The analytic runner now exports a self-contained offline `report.html` before
writing the completion manifest. It presents image/mask hashes and byte counts,
verbatim stage evidence, mask semantics, shared camera/ISP state, partial registry,
resource limits and pending work. It matches the existing synthraw report's system
typography and neutral tables. All metadata is escaped; no scripts, external
assets or image decoding are used. Scientific acceptance stays explicitly not
measured, separate from successful independent file roundtrips. The manifest
hashes the HTML; HTML lists image/mask artifacts without a circular self-hash.
Failed report generation leaves no completion manifest.

Validation: all 118 lightweight v5 tests pass, including metadata injection
escaping, actual tiny source-to-TIFF-to-report/hash output and report-failure
cleanup. Repository-wide ruff and changed Python formatting pass. No notebook,
model, full-frame job, full suite or background service ran. Complete MVP
configuration/registry and the notebook remain pending before fidelity. Later
phases and Colab-only scientific/full-scale acceptance remain unresolved.

## Implementation evidence: fourteenth MVP increment

`registry.analytic_registry` extends the executed bundle's field records to source
hash/header observations, explicit decoding policy, supplied analytic ISP values,
geometry requests and derived sensor transforms/crops. Each record includes units,
mode, provenance, method and validation limits. Synthetic assumptions are distinct
from observations and arithmetic derivations; confidence/posterior remain null.
Added records exclude source paths and private EXIF. Existing camera and user-only
identity records retain their semantics. The runner and HTML now carry this richer
partial registry without changing rendering or sensor defaults.

Validation: 119 lightweight v5 tests pass under the heavy-test lock, including
nondefault ISP/gain/geometry consistency, detached records, JSON serialization,
privacy defaults and actual tiny source-to-independent-TIFF manifest evidence.
Repository-wide ruff and changed Python formatting pass. No full suite, notebook,
model, full-frame processing or background service ran. Complete configuration,
remaining registry fields and the notebook remain MVP work before fidelity;
later phases and Colab-only scientific/full-scale validation remain unresolved.

## Implementation evidence: fifteenth MVP increment

`config.AnalyticRunConfig` supplies a frozen, versioned JSON configuration for
the executed analytic subset, including actual shared camera/ISP hypotheses,
50 MP geometry, colour assumption and resource budgets. Its run method passes
these settings directly to the existing bounded pipeline. Parsing rejects
unknown/duplicate fields, nonfinite numbers, invalid domains and unsupported
versions. File paths remain execution arguments. Conservative development
budgets still reject full-scale work before allocation; larger budgets must be
supplied explicitly in Colab. This subset does not claim complete PRD settings,
field-mode orchestration or learned execution.

Validation: 137 lightweight v5 tests pass under the test lock. New cases verify
JSON roundtrip, malformed configuration rejection, a real tiny PNG through
independent LibRaw TIFF with nondefault camera/ISP settings, and full-scale
memory rejection before output creation. Repository-wide ruff and changed-file
formatting checks pass. No notebook, model, full-frame job, background process
or full repository test suite ran. Complete panel/schema coverage and notebook
remain MVP work; subsequent ordered phases and scientific/Colab validation are
still pending.

## Implementation evidence: sixteenth MVP increment

The analytic runner and strict JSON config now expose the existing geometry
planner's aspect-preserving/native modes and all three native fit policies.
Native geometry retains 8640x5760 independently of requested target MP; effective
transforms/crops continue through rasterization, DNG tags and provenance. Defaults
remain unchanged. Unsupported outpainting is rejected during preflight before
output creation or any frame allocation, rather than after allocating native
arrays. Configuration validation now shares the planner's geometry domains,
including its 1000 MP upper bound; no extra full-frame validation work is needed.

Validation: 146 lightweight v5 tests pass under the shared lock. New tests cover
JSON roundtrip of every native fit policy, native full-size memory rejection even
with a tiny target request, unavailable generation rejection before allocation,
and invalid geometry settings. Existing tiny PNG-to-LibRaw-TIFF execution remains
green. Repository-wide ruff and changed Python formatting pass. No notebook,
model, full-frame processing, full repository suite or background service ran.
Complete MVP panel/registry coverage and notebook remain pending; later ordered
phases and full-scale Colab/scientific acceptance remain unresolved.

## Implementation evidence: seventeenth MVP increment

`analytic-config.schema.json` provides a portable JSON Schema 2020-12 contract
for every currently implemented analytic configuration field, including nested
camera/ISP values, privacy-preserving rejection of unknown identity fields and
geometry domains. It explicitly describes subset scope and the required second
validation step through `AnalyticRunConfig.from_json`. Matrix conditioning,
finite arithmetic, duplicate-key detection and runtime resource admission remain
executable gates, not claims made by structural schema validation. JSON Schema
accepts integral numbers such as `1.0` for integer fields; the stricter parser
requires integer tokens. Defaults and existing execution behavior are unchanged.

Validation: 163 lightweight v5 tests pass under the shared lock, including schema
meta-validation, exact dataclass field coverage, default/nondefault saved config
validation, invalid domain/privacy cases and a singular matrix demonstrating the
structural/semantic boundary. Repository-wide ruff and new Python formatting
checks pass. `jsonschema` was installed only in the external lightweight dev
environment; it is also a dependency of the existing nbformat dev requirement.
No full repository suite, notebook, model, full-frame job, commit or background
service ran. Complete MVP panel/registry coverage and notebook remain pending;
later phases and Colab-only scientific acceptance remain unresolved.

## Implementation evidence: eighteenth MVP increment

Successful analytic bundles now include `config.json`, parsed by the existing
strict `AnalyticRunConfig` contract and hashed alongside image/mask artifacts in
the manifest and offline report. It retains actual camera/ISP hypotheses, geometry,
colour assumption and resource budgets; paths remain execution arguments. Direct
runner settings are validated before allocation. Configuration export occurs
before the completion marker, so a failed export cannot advertise completion.

Validation: all 164 lightweight v5 tests pass under the shared test lock. A tiny
nondefault PNG run saves/reloads its settings and reproduces identical DNG, TIFF,
mask and configuration hashes in the same runtime. Repository-wide ruff and
changed Python formatting pass. This proves local replay, not cross-runtime
bitwise reproducibility or calibrated camera truth. Complete MVP configuration,
registry and notebook remain pending before the later ordered phases. Colab-only
full-scale/scientific validation remains unresolved. No model, notebook, full-frame
job, full repository suite, commit or background service ran.

## Implementation evidence: nineteenth MVP increment

Analytic completion manifests and offline reports now retain loaded Python,
NumPy, Pillow, LittleCMS, tifffile, rawpy and native LibRaw versions. OS family
and machine architecture provide portable runtime context without hostname,
user identity, paths or environment variables. Optional missing LittleCMS is
null. These records support diagnosis of replay differences without claiming
dependency pinning or cross-runtime reproducibility.

Validation: all 166 lightweight v5 tests pass under the shared test lock.
Tests cover actual tiny source-to-independent-TIFF runtime serialization and
HTML evidence, native library version reporting, absent optional codec support,
detached records and escaped runtime metadata. Repository-wide ruff and changed
Python formatting checks pass. No model, notebook, full-frame job, full repository
suite, commit or background service ran. Complete MVP configuration/registry and
the notebook remain pending before later ordered phases. Full-scale Colab,
cross-runtime replay and scientific acceptance remain unresolved.

## Implementation evidence: twentieth MVP increment

The analytic inverse ISP now reports a bounded forward-model consistency check
on the actual stored clean float32 camera values. Each strip is rendered through
the supplied exposure/WB, matrix, display clipping, optional smoothstep and sRGB
transfer. Reports retain maximum absolute error, RMSE, normalized encoded-sRGB
units and a declared 1e-5 maximum-error tolerance. Irreversible camera gamut
flooring is reflected as a failed diagnostic, not hidden or promoted to a claim
of calibrated colour accuracy. The existing manifest/HTML stage evidence carries
these results; bundle export remains independent of this diagnostic status.

Validation: 167 lightweight v5 tests pass under the shared lock, including
nonidentity matrix/WB/exposure/tone reconstruction, endpoints, strip-invariant
metrics, an independently calculated gamut-flooring failure and tiny actual
PNG-to-LibRaw-TIFF integration. Repository-wide ruff and changed Python formatting
pass. No model, notebook, full-frame job, full repository suite, commit or
background service ran. Complete MVP panel/registry and notebook remain pending
before later ordered phases. Full-scale Colab and scientific acceptance remain
unresolved.

## Implementation scheduling checkpoint

Parent review after iteration 26: the analytic path already executes through independent DNG/TIFF development. Prioritize the remaining direct deliverable now: create the single Colab notebook with setup, hardware inspection, every PRD setting and maximum-scale defaults, then advance fidelity, learned RAW dispatch, full sensor, full optics and complete metadata in order. Do not spend the remaining iteration budget on additional small analytic-only schema, report or provenance polish while those phases have not been attempted. Each next increment should close a substantial remaining roadmap requirement end to end, reusing existing tested synthraw implementations where appropriate. Finish with one locked full-suite run and lint, and replace accumulated increment history with a concise final implemented/deferred matrix. Unsupported research checkpoints may be deferred with evidence; available analytic/sensor/optics functionality and the notebook itself should be implemented rather than deferred.


## Executable notebook camera panel evidence

The single Colab notebook now routes eight supported catalogue fields (CFA,
ADC depth, black level, full well, ISO, shutter, aperture, seed) through validated
field-mode resolution into the actual paired pipeline. Explicit values require
USER mode; missing estimates and unsupported edits fail before image work.
Both child bundles preserve the same selected hypotheses and mode/provenance.
Tiny PNG-to-DNG-to-independent-LibRaw-TIFF integration verifies a panel ISO
selection in both bundles; invalid pair evidence fails before output creation.
No notebook or full-scale job was executed locally. Remaining ISP catalogue
mapping, later roadmap phases and Colab scientific validation are unresolved.

## Executable notebook white-balance panel evidence (iteration 30)

The notebook now resolves AsShotNeutral with camera catalogue inputs before
paired execution. USER neutral values must be positive, finite and green-normalized;
their reciprocals become ISP white-balance gains. Selection records feed both
bundles. AUTO/FABRICATE retain the explicit hypothesis; absent USER/ESTIMATE
evidence fails. No illuminant estimator is claimed. DNG ColorMatrix1/2 cannot
be equated to camera-to-sRGB inputs and remain unsupported catalogue edits.

Tiny real PNG-to-paired-DNG-to-independent-LibRaw-TIFF tests verify expected
red/blue signal scaling, unchanged green, decoded white balance and persisted
selection provenance. Notebook format/syntax and panel wiring are statically
validated without executing notebook cells. Full-scale Colab validation and the
remaining ordered fidelity, learned RAW, sensor, optics and metadata work remain
unresolved.

## Fidelity checkpoint and tiled execution evidence (iteration 31)

`prd_v5.fidelity` now accepts caller-supplied HAT/Real-ESRGAN x4 checkpoint
contracts with streamed SHA-256 verification, full repository revision, source,
licence and explicit acceptance. The Colab/CUDA loader uses tensor-only loading
and validates Spandrel architecture, RGB channels and scale. No checkpoint is
invented or downloaded. Execution reuses the existing disk-backed halo/blend
engine with 512/64 defaults, visible OOM tile reduction and failure cleanup.
Reports preserve requested/effective backend, integrity evidence and no fallback.

Lightweight tests cover actual tiny tiled output with model doubles, OOM retry,
invalid-output cleanup, integrity/acceptance and incompatible model descriptors.
Real checkpoint compatibility, full-scale GPU memory/performance and scientific
fidelity remain Colab-only unresolved validations. Notebook/paired RAW wiring,
HDR, exposure/profile selection and comparison metrics remain fidelity work;
this standalone SR API does not claim completion of that phase.
