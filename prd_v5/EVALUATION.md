# PRD v5 evaluation and build plan

## Full optics increment: bounded radial attenuation API

`optics.Vignetting` implements an immutable centered achromatic attenuation
prior on clean linear camera RGB, intended before CFA and electron-domain noise.
Transmission is `1 - corner_strength * r_squared`, with pixel-center radius
normalized by half the sensor diagonal. Zero copies pixels exactly; nonzero
coefficients require explicit experimental acceptance. HDR radiance is retained
without unit clipping. Separate float32 disk-backed output, bounded row strips,
input-domain checks, flush-on-success and JSON-safe effect evidence are supported.
This is a synthetic forward effect, not inferred lens calibration or removal of
shading already present in rendered input.

All 409 lightweight v5 tests pass under the shared resource lock. Tiny tests
independently calculate every output pixel, verifying center/corner behavior,
symmetry, HDR retention, strip invariance, persisted output, zero defaults and
invalid-buffer/acceptance/domain rejection. Repository-wide ruff passes.
Pipeline, shared config and schema integration are implemented below. Paired-run
and notebook controls are integrated below. Distortion, CA, spatial PSF, diffraction,
defocus and motion remain next optics work before final metadata. Full-scale
memory and physical validation remain unresolved Colab-only work. No notebook,
model, download, full-frame job, background service or commit was started.

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

Parent review after iteration 45: the analytic path, notebook, paired fidelity SR, crop comparisons, exposure and colour/profile adapters now exist. The remaining 15 iterations must prioritize substantive missing roadmap phases: implement analytic HDR and its uncertainty in one integrated increment; implement honest learned-RAW residual dispatch and unavailable-model reporting in one integrated increment; reuse existing synthraw sensor/optics implementations for complete physical stages and notebook controls; then complete metadata, outputs, acceptance reporting and locked full-suite validation. Avoid separate adapter, wiring and notebook iterations for each individual field. Update the executable pipeline and notebook together when adding a phase. The analytic path already executes through independent DNG/TIFF development. Prioritize the remaining direct deliverable now: create the single Colab notebook with setup, hardware inspection, every PRD setting and maximum-scale defaults, then advance fidelity, learned RAW dispatch, full sensor, full optics and complete metadata in order. Do not spend the remaining iteration budget on additional small analytic-only schema, report or provenance polish while those phases have not been attempted. Each next increment should close a substantial remaining roadmap requirement end to end, reusing existing tested synthraw implementations where appropriate. Finish with one locked full-suite run and lint, and replace accumulated increment history with a concise final implemented/deferred matrix. Unsupported research checkpoints may be deferred with evidence; available analytic/sensor/optics functionality and the notebook itself should be implemented rather than deferred.


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

## Fidelity-to-RAW integration evidence (iteration 32)

The single-run `pipeline.run_analytic` accepts an opt-in `sr_checkpoint`.
Verified tiled x4 encoded-sRGB output feeds geometry resampling, analytic inverse
ISP, CFA/electron noise, DNG export and independent LibRaw/TIFF validation.
The original geometry and camera hypotheses are preserved; only the resampling
source size and scale change. Checkpoint errors propagate without fallback or
completion marker, and temporary reconstructed arrays close and disappear.
Stage evidence retains checkpoint provenance and requested/effective backend.
`config.json` remains the analytic subset; manifests explicitly require the
separate checkpoint evidence for SR replay. Clipping masks describe reconstructed
samples when SR is enabled, not original highlight endpoints.

Tiny model-double integration verifies an independently calculated signal change
in decoded DNG samples and successful independent TIFF roundtrip. This establishes
wiring only. Real HAT/Real-ESRGAN quality, compatibility, peak memory and full-scale
execution remain unvalidated and Colab-only. Pair/config/notebook SR dispatch,
HDR and exposure/profile selection remain pending fidelity work.

Validation: 244 lightweight v5 tests pass under the shared lock, repository-wide
ruff and changed Python formatting pass, and `git diff --check` is clean. No
notebook execution, downloads, real inference, full-frame processing, commits or
background services were started.

## Paired checkpoint SR evidence (iteration 33)

`AnalyticRunConfig.run` and `paired.run_analytic_pair` now accept the optional
`sr_checkpoint` execution argument. Sequential 2K/4K children use the same frozen
checkpoint specification and immutable analytic hypotheses. Pair preflight
checks checkpoint integrity before output creation; each child independently
verifies and loads it. The pair manifest records shared checkpoint evidence,
labels SR execution accurately and requires separate checkpoint bytes for replay.
Existing analytic defaults and portable analytic configuration remain unchanged.

Tiny end-to-end model-double tests prove actual SR pixel effects in both decoded
DNGs, independent TIFF roundtrips, shared checkpoint/camera evidence and temporary
array cleanup. Invalid checkpoint preflight creates no output; failure loading
the second child retains the first bundle without a pair completion marker.
247 lightweight v5 tests pass under the shared lock; repository-wide ruff,
changed Python formatting and diff whitespace checks pass. No heavy local work,
notebook execution, downloads, commits or background services were started.

Notebook SR dispatch, HDR, exposure/profile selection and comparison metrics
remain fidelity work. Real checkpoint compatibility, scientific fidelity and
full-scale memory/performance remain unresolved Colab-only validations. Later
learned RAW, extended sensor, optics and final metadata phases remain pending.


## Notebook checkpoint dispatch evidence (iteration 34)

The single notebook now accepts a user-supplied `SR_CHECKPOINT` dictionary,
preflights its hash/licence/backend through `panel.resolve_sr_panel`, installs
optional SR dependencies only in its Colab execution cell and passes the shared
specification into sequential paired RAW execution. No weights are downloaded.
Missing checkpoints require explicit analytic-baseline acceptance; invalid
supplied checkpoints fail even when baseline acceptance is enabled. Requested,
planned and effective SR states distinguish preflight from completed execution.
Tile/overlap edits outside the wired 512/64 defaults fail explicitly.

Lightweight tests cover integrity failure, backend mismatch, absent-checkpoint
opt-in and notebook AST dispatch/completion-state ordering. No notebook, real
model, download or full-scale job ran locally. Real HAT/Real-ESRGAN compatibility,
GPU memory and scientific fidelity remain unresolved Colab-only checks. HDR,
exposure/profile selection and comparison metrics remain fidelity work before
learned RAW, extended sensor, optics and final metadata phases.

## Full-resolution paired render metrics (iteration 35)

Sequential paired execution now compares both completed developed TIFFs before
publishing pair completion. `comparison.compare_renders` uses read-only TIFF
memmaps and bounded float64 row strips to measure normalized MAE, RMSE, maximum
absolute error, RGB signed bias, PSNR and endpoint saturation fractions at full
resolution. No alignment, downsampling, resizing or exposure fit is performed.
Metrics include sensor margins in the LibRaw visible raster, without DefaultCrop.
Unequal output shapes explicitly skip metrics. Identical samples report null
PSNR with an infinity explanation, preserving strict JSON. Mappings close on
success and failure; comparison errors withhold the pair completion marker.

Validation: independently calculated sparse RGB differences verify every metric
and strip invariance; tiny actual PNG-to-DNG-to-independent-TIFF paired execution
verifies identical renders, unequal-shape skips and completion failure handling.
All 260 lightweight v5 tests pass under the shared lock; repository-wide ruff
and changed Python formatting checks pass. These are render
disagreement diagnostics, not source registration, recovered truth, perceptual
quality or a benchmark acceptance claim. Alignment, licensed reference datasets,
HDR and joint exposure/profile selection remain fidelity work. Full-scale memory,
real checkpoint inference and scientific validation remain Colab-only unresolved
checks. No notebook, model download, real inference, heavy processing, commit or
background service was started.

## Executable joint exposure posterior (iteration 36)

`exposure.resolve_exposure` reuses the existing standard-stop exposure estimator
under an explicit Sony-like camera envelope and caller-supplied scene EV100 prior.
It atomically selects ISO/shutter/aperture through v5 field resolution, returning
one immutable camera and executable selection records. Records retain the joint
alternatives, normalized relative support and EV prior, with null confidence.
The pipeline already preserves those records in manifest/HTML. ISO recomputes
sensor gain, saturation and noise; shutter/aperture remain metadata hypotheses.
ISP exposure gain is independent because absolute radiance and throughput are
unknown. Existing defaults are unchanged. This is heuristic exposure selection,
not calibrated probabilities, image metering or recovered capture settings.

Validation: eight new lightweight tests verify deterministic joint selection,
normalized support, EV response, invalid-input rejection, actual ADC pixel changes
and a tiny real PNG-to-DNG-to-independent-LibRaw-TIFF bundle with retained joint
provenance. All 268 v5 tests pass under the shared lock; repository-wide ruff and
changed Python formatting pass. No heavy local work or background services ran.
Notebook posterior controls, HDR, colour constancy, profile selection and validation
crops remain fidelity work. Colab full-scale/scientific validation and later
ordered phases remain unresolved.

## Notebook joint exposure dispatch (iteration 37)

The single notebook now exposes `EXPOSURE_PRIOR` and passes it through
`resolve_analytic_panel` into the executable joint exposure resolver. An explicit
prior requires all three ISO/shutter/aperture fields in ESTIMATE mode without
values; conflicting or incomplete choices fail before pixel work. The selected
immutable camera drives both sequential runs, with joint candidate evidence
retained in both registries. The default null prior preserves current behavior.
No image metering, calibrated probability or radiometric shutter/aperture effect
is claimed; ISO changes the existing sensor electronics.

Validation: tiny real PNG-to-paired-DNG-to-independent-LibRaw-TIFF coverage
verifies increased ADC samples against the unchanged baseline and matching joint
provenance in both bundles. Static notebook validation checks explicit prior
wiring without executing cells. All 274 lightweight v5 tests pass under the
shared lock; repository-wide ruff, changed Python formatting and diff whitespace
checks pass. No background processes or heavy jobs were started. HDR, colour
constancy/profile selection and validation crops remain fidelity work; later
ordered phases and full-scale Colab/scientific validation remain unresolved.

## Bounded colour-constancy hypothesis (iteration 38)

`colour.resolve_gray_world` adds opt-in executable colour-constancy evidence for
the fidelity phase. It scans normalized encoded-sRGB arrays/memmaps in bounded
row strips, inverts the sRGB transfer, excludes endpoint pixels across all
channels and selects G-normalized gains through the existing ISP field resolver.
The immutable ISP and selection records can drive the analytic runner, retaining
ESTIMATED provenance, null confidence, channel statistics and explicit scene and
channel-proxy assumptions. Missing/degenerate evidence fails without unity fallback.
Existing pipeline defaults are unchanged. This is an analytic gray-world heuristic,
not a learned model, calibrated illuminant or recovered capture white balance.
Rendered WB/tone/colour cannot be disentangled. The inverse ISP divides by gains;
it does not neutralize the rendered input or establish colour accuracy.

Validation covers independently calculated gains, endpoint exclusions, strip
invariance, strict JSON/provenance and actual disk-backed inverse-ISP pixel
changes with passing forward consistency. Notebook dispatch and coherent shared
paired-source colour evidence remain pending, along with HDR/profile selection,
later ordered phases and full-scale/scientific Colab validation. No notebook,
model download, heavy processing, commit or background service was started.

All 285 lightweight v5 tests pass under the shared test lock. Repository-wide
ruff, changed Python formatting and diff whitespace checks pass.

## Shared colour panel evidence (iteration 39)

`panel.resolve_analytic_panel` now accepts an explicit normalized encoded-sRGB
`colour_source` array/memmap and bounded `colour_strip_rows`. Evidence requires
AsShotNeutral ESTIMATE mode without a value; missing entries and conflicting
modes/values fail. The gray-world candidate goes through normal ISP resolution,
retaining its statistics, ESTIMATED provenance, null confidence and limitations.
One resolved immutable ISP and its records can drive both paired sources without
independently sampling or estimating different white-balance hypotheses. Existing
calls and defaults are unchanged. The caller selects and decodes the shared proxy;
notebook decoding/dispatch remains pending. No capture WB accuracy is claimed.

Validation: a tiny real PNG-to-paired-DNG/TIFF test checks expected CFA channel
scaling against an unselected baseline, LibRaw WB metadata, both manifests and
independent roundtrip completion. Four conflict cases verify rejection. All 290
lightweight v5 tests pass under the shared test lock; repository-wide ruff,
changed Python formatting and diff whitespace checks pass. No heavy execution,
background process or commit was started. HDR/profile selection, later ordered
phases and full-scale/scientific Colab validation remain unresolved.

## Notebook shared colour source (iteration 40)

The notebook now exposes optional `COLOUR_SOURCE_PATH` and dispatches it through
`panel.resolve_analytic_panel_from_source` with the existing decoder budget,
strip size and explicit unprofiled-sRGB policy. A null path preserves defaults.
An explicit proxy requires AsShotNeutral ESTIMATE mode without a value. The
helper decodes into temporary disk-backed float32 storage, resolves one immutable
ISP for both paired runs, retains source hash and colour/decode evidence in the
WB posterior, and closes/removes scratch on success or failure. Decoder admission
precedes scratch allocation. No capture WB accuracy or learned estimator is claimed.

Validation: tiny RGB PNG tests independently check selected gains, evidence,
success/failure cleanup, memory admission and null-path compatibility. Static
notebook/schema tests check panel wiring without notebook execution. All 294
lightweight v5 tests pass under the shared test lock; repository-wide ruff passes.
HDR/profile selection, later ordered phases and full-scale/scientific Colab
validation remain unresolved. No heavy work, background process or commit started.

## Native-resolution validation crop metrics (iteration 41)

`comparison.compare_renders` accepts an optional explicit half-open
`(top, left, bottom, right)` visible-raster region. Both uint16 TIFFs are sliced
at identical native pixel coordinates before bounded strip metrics, retaining
original raster shapes and selected coordinates in strict JSON evidence. No
resampling, registration, exposure fitting or DefaultCrop application occurs.
Explicit regions can compare common coordinates on unequal rasters; this does
not imply scene correspondence. Empty, noninteger, negative and out-of-raster
regions fail, with both mappings closed. Whole-raster defaults remain unchanged.

Validation: independently calculated sparse differences verify crop-only MAE,
RMSE, channel bias, saturation and strip invariance while excluding an external
changed pixel. Unequal-raster common regions and failed-bounds cleanup are tested.
All 303 lightweight v5 tests pass under the shared lock; repository-wide ruff
and changed Python formatting pass. Notebook/report crop selection, HDR/profile
selection and later ordered phases remain pending. Full-scale and scientific
Colab validation remain unresolved. No notebook, model, heavy processing,
background service or commit was started.

## Paired native-resolution crop evidence (iteration 42)

`run_analytic_pair(..., comparison_region=(top, left, bottom, right))` now
publishes optional `crop_comparison` alongside unchanged whole-raster metrics.
Coordinates, sample counts and scientific limitations survive JSON serialization.
Syntax is checked before source I/O; bounds are checked against both developed
TIFFs, and failure withholds pair completion while preserving child bundles.
Tiny PNG-to-DNG-to-LibRaw-TIFF integration verifies crop counts and equivalence
to independent direct comparison. Tests cover malformed and out-of-bounds regions.
Notebook crop selection remains pending; no full-scale or scientific validation
ran locally. Later ordered phases remain pending.

## Colab native-resolution crop selection (iteration 43)

The single v5 notebook now exposes optional `COMPARISON_REGION` and forwards it
to the paired runner. Default `None` preserves whole-raster comparisons; an
explicit half-open `(top, left, bottom, right)` adds crop evidence to the printed
result and saved pair manifest. Panel comments and running documentation identify
LibRaw visible-raster coordinates and explicitly disclaim scene registration.

Validation: 35 targeted notebook, paired-run and comparison tests pass under the
shared lock, including tiny real PNG-to-DNG-to-LibRaw-TIFF crop integration.
Static notebook validation checks the untouched full-scale defaults, schema,
unexecuted cells and crop argument dispatch. Repository-wide ruff, changed Python
formatting and diff whitespace checks pass. No notebook execution, heavy work,
background process or commit started. HDR/profile selection, later ordered phases
and full-scale/scientific Colab validation remain unresolved.
# Explicit D65 profile selection evidence (iteration 44)

`profile.resolve_d65_profile` converts an explicit user XYZ-to-camera D65 matrix
into the analytic camera-to-linear-sRGB convention and resolves it as USER_SET
evidence. The returned immutable ISP and selection records use the existing
runner contracts. Singular, nonfinite and ill-conditioned matrices fail before
execution. WB, exposure and tone remain unchanged. This selects a supplied
hypothesis, not an inferred or verified camera calibration. Auxiliary DNG
transforms, chromatic adaptation, dual-illuminant interpolation and notebook
catalogue dispatch remain unsupported; DNG reports retain virtual-profile wording.

Validation: 62 targeted lightweight profile, colour, DNG and field tests pass
under the shared lock. Independent matrix arithmetic verifies actual tiny
inverse-ISP pixel changes, forward consistency, selection provenance and matching
derived DNG matrices under both D65 illuminants. Repository-wide ruff passes.
No full-scale job, notebook, model, background service or commit was started.
HDR, benchmark adapters, later ordered phases and Colab scientific validation
remain unresolved.

# Catalogue D65 profile dispatch evidence (iteration 45)

The notebook's existing `ColorMatrix1/2` row now accepts one nested 3x3
XYZ-to-camera D65 matrix in USER mode through `panel.resolve_analytic_panel` and
its source-decoding wrapper. Conversion reuses the tested profile resolver;
USER_SET evidence retains the original matrix and single-D65 limitations in
both paired manifests. AsShotNeutral remains independently selectable. Both
exported ColorMatrix tags use this matrix and D65 illuminants. Absent
AUTO/FABRICATE values preserve the supplied ISP; unresolved USER/ESTIMATE,
non-USER explicit values and invalid matrices fail before execution.

Validation: 56 targeted lightweight profile, panel and notebook tests pass,
including tiny paired PNG-to-DNG/TIFF execution, independent serialized matrix
checks, provenance retention and simultaneous WB selection. Repository-wide
ruff and changed Python formatting pass. No models, full-scale processing,
notebook execution, background processes or commits were started. Automatic
profile estimation, dual-illuminant interpolation, HDR and later ordered phases
remain pending; full-scale and scientific Colab acceptance remain unresolved.

## Integrated analytic HDR hypothesis (iteration 46)

The single notebook, replay schema and executable paired pipeline now support
an opt-in normalized inverse-Reinhard hypothesis via
`AnalyticISP.hdr_max_radiance`. Default 1 disables expansion and preserves prior
behavior; explicit M > 1 inverts y = r(1 + 1/M)/(1 + r), mapping normalized
linear display values to [0, M]. It runs in bounded row strips after inverse
smoothstep, before camera colour/WB inversion and CFA/electron sensor noise.
The forward self-consistency diagnostic applies the matching compression.
Reports retain experimental status, expanded-pixel counts, clipping uncertainty,
no learned execution and unchanged sensor saturation without empty headroom.
The notebook exposes M=16 with separate experimental acceptance false; requested
HDR remains true. Effective execution is recorded only after pair completion.
This is a supplied tone-curve hypothesis, not recovered highlight detail,
calibrated HDR, or an available learned checkpoint. Endpoint masks retain their
existing narrower semantics. Full-scale/visual/scientific checks remain Colab-only.

Validation includes independent inverse/forward arithmetic, endpoints, strip
invariance, invalid domains, replay configuration, static notebook dispatch and
tiny real paired DNG/LibRaw/TIFF outputs with independently calculated ADC values.
No notebook, download, real inference, heavy processing or background service ran.
Learned RAW, full sensor/optics, final metadata and benchmark acceptance remain
pending in roadmap order.

## Learned RAW residual dispatch (iteration 47)

Implemented executable experimental camera-aware residual dispatch through the
single-run pipeline, shared paired configuration and single Colab notebook.
Immutable checkpoint/domain contracts verify bytes, normalization, exact camera
hypotheses and explicit licence/experimental acceptance. Bounded deterministic
tiles produce clean camera-space residuals and confidence before CFA/electron
noise. Confidence gating, weighting, nonnegative flooring and zero-effect counts
are recorded. ANALYTIC rejects an accidental adapter, HYBRID explicitly skips
missing adapters and LEARNED_RGB2RAW fails before source I/O without one. Strict
mode still uses the specified analytic-plus-residual architecture. Saved config
contains the analytic baseline; replay additionally needs the caller-loaded
adapter. Pair/child manifests retain requested/effective execution evidence.

No compatible RAW-Flow/SpiralDiff/RAW-Diffusion loader or camera-adapted weights
are supplied or certified. The generic contract cannot verify that a callable
uses the declared checkpoint. Model-specific normalization, camera adaptation,
halo/overlap inference and licensed genuine-RAW benchmark validation remain
research integrations. Notebook defaults preserve requested RAW 1024/128; an
explicit generic adapter requires 1024/0 and experimental acceptance, preventing
an unsupported overlap request from silently executing. Default missing-adapter
HYBRID runs visibly retain analytic execution. Full sensor, optics and metadata
are the next ordered phases; full-scale/model/scientific checks remain Colab-only.

Validation uses tiny model doubles only: numerical confidence gating, absolute
tile coordinates, checkpoint/domain/normalization rejection, malformed outputs,
strict pre-I/O failure, paired PNG-to-DNG-to-independent-LibRaw/TIFF with measured
ADC changes and manifest evidence, notebook syntax/wiring and failure cleanup.
No notebook, research model, download, heavy job or background service ran.

All 349 lightweight v5 tests passed under the shared test lock. Repository-wide
ruff, changed Python formatting and diff whitespace checks passed. No commits
or background processes were created.

## CFA-domain sensor body effects (iteration 48)

`SensorEffects` now supplies immutable, JSON-replayable synthetic PRNU, DSNU,
row/column FPN, temperature-scaled dark current and hot/dead defect priors.
The executable single and sequential paired pipelines retain these parameters
in config.json, namespaced registry records and sensor-stage evidence. Body maps
use independent component/absolute-row streams from the shared camera seed;
column maps require only one row of memory. CFA selection precedes all effects.
PRNU changes the photoelectron expectation before Poisson sampling. Hot defects
add exposure-dependent electron expectation; dead defects override photo/hot/dark
expectations but retain subsequent read noise and offset electronics. Dark current
is a Poisson expectation when shot noise is enabled, with a declared doubling
interval relative to 20 C. These are uncalibrated models, not Sony measurements.

The notebook exposes `SENSOR_VALUES`, honors each corresponding ENABLE switch,
requires `ACCEPT_SENSOR_EXPERIMENT` for nonzero effects and records effective
states only after pair completion. Zero amplitudes retain exact previous defaults
and report zero-effect, even when a feature was requested. The existing PRD field
catalogue remains intact; body coefficients are explicit structured hypotheses,
not automatically estimated from catalogue field modes. DNG NoiseProfile still
describes only baseline temporal noise, excluding body maps/dark current; this
limitation is explicit in stage evidence. Sensor nonlinearity and dual conversion
gain remain pending before full optics and final metadata, as do defect-mask
artifacts and physical calibration. Full-scale memory and scientific checks remain
Colab-only and unresolved.

Validation: 365 lightweight v5 tests pass under the shared lock, including exact
PRNU-before-Poisson arithmetic, dark temperature/exposure arithmetic, body spatial
structure and strip/seed invariance, hot/dead precedence, zero-default equivalence,
config replay and actual tiny paired DNG-to-independent-LibRaw/TIFF validation.
Notebook validation is static only. No model, notebook, download, full-scale job,
commit or background service was started.


## Full sensor increment: bounded nonlinear charge response

`SensorEffects.nonlinearity_compression` now executes a monotone rational
charge-to-signal compression after shot sampling and before read electronics.
The coefficient is an explicit uncalibrated synthetic prior in [0, 1]. Zero
preserves existing behavior; nonzero execution remains experimental. The shared
immutable config, JSON schema, notebook structured sensor panel, registry and
stage manifest retain the value. Reports state the formula, order and baseline
NoiseProfile limitation. No Sony calibration or scientific acceptance is claimed.

Validation: 372 lightweight v5 tests pass under the shared resource lock,
including independent deterministic arithmetic, seeded shot/read ordering,
strip invariance, domain rejection, config replay and tiny paired DNG/LibRaw/TIFF
roundtrip with nonzero compression. Repository-wide `python -m ruff check .`
passes. No notebook, model, full-scale processing or background service ran.
Dual conversion gain, remaining optics/metadata work and real-response/full-scale
Colab validation remain unresolved. No commits were made.

## Full sensor increment: realized defect occupancy evidence

Sensor-stage manifests now record total photosites, realized hot/dead selections,
hot/dead overlaps and hot selections surviving the dead override. Counts use the
existing row-bounded seeded maps without allocating full-frame masks or changing
ADC behavior. They describe synthetic selections, not detected defects or proof
of changed codes: zero hot-current rates, rounding and saturation can leave selected
sites unchanged. Dead sites still retain read/offset electronics. Counts do not
replace the still-pending defect-mask artifacts or physical calibration.

Validation: all 375 lightweight v5 tests pass under the shared resource lock.
Independent seeded-map reconstruction covers zero, complete and partial occupancy,
dead precedence and strip invariance. Tiny paired DNG/independent-LibRaw/TIFF
execution retains counts in both manifests, including hot selections with zero
hot-current contribution. Repository-wide ruff and diff whitespace checks pass.
No models, notebooks, downloads, full-scale jobs, background services or commits
were started. Dual conversion gain, defect-mask export, optics/final metadata and
full-scale/scientific Colab validation remain unresolved.

## Full sensor increment: bounded defect-mask output API

`BayerSensor.process(..., defect_mask=...)` optionally writes a caller-owned,
sensor-sized uint8 memmap using the exact defect selections applied to ADC
simulation. Bit 0 records hot selection, bit 1 dead selection, and value 3 retains
overlap despite dead-charge precedence. Every row is overwritten and the buffer
is flushed on success. Shape, dtype, writability and buffer separation are checked
before ADC writes. Working memory remains row-bounded and requesting the mask
consumes no additional random draws. Stage evidence distinguishes mask writing
from occupancy reporting. Flags describe synthetic selections, not detected
camera defects or guaranteed changes to ADC codes.

Validation: all 379 lightweight v5 tests pass under the shared resource lock.
Tests independently reconstruct flags, check partial/zero/full occupancy, overlaps,
strip invariance, persisted bytes, unchanged ADC output and pre-write rejection of
invalid/aliased buffers. Repository-wide ruff and diff whitespace checks pass.
The pipeline does not yet request or package this optional buffer; run-bundle
artifact export and notebook wiring remain pending. Dual conversion gain,
optics/final metadata and full-scale/scientific Colab validation also remain
unresolved. No notebook, model, download, full-scale job, background service or
commit was started.

## Full sensor increment: packaged defect-selection artifacts

The single-run pipeline and sequential paired/Colab runner now export
`sensor_defects.npy` from the exact sensor API flags after the independent
DNG/TIFF image gate. Both the working mask and packaged uint8 NPY are disk-backed;
copying uses bounded row strips. Temporary mappings close on success/failure.
Manifest hashes, shape, full-sensor coordinates, bit meanings, occupancy counts
and scientific limits survive JSON and offline HTML reporting. Zero priors emit
all-zero masks without changing ADC behavior or seeds. The additional disk budget
is one byte per photosite for each working and packaged mask. No new panel switch
is required: masks follow the existing unconditional mask-output behavior.

Validation: tiny actual PNG-to-DNG-to-independent-LibRaw/TIFF runs cover zero,
complete and partial defect fractions, independently reconstructed seeded flags,
overlap counts, persisted hashes, JSON/HTML evidence, strip-invariant decoded
RAW/TIFF pixels and identical mask bytes. DNG strip layout changes container
bytes across strip heights, so pixel equality is the relevant invariant. Full-scale memory/I/O and scientific calibration remain
Colab-only. Dual conversion gain, optics and final metadata remain unresolved.
No notebook, model, download, heavy processing, background service or commit ran.

All 382 lightweight v5 tests pass under the shared resource lock.
Repository-wide ruff, changed Python formatting and diff whitespace checks pass.

## Full sensor increment: explicit ISO-switched electronics profile

`electronics.DualConversionProfile` now resolves an explicit immutable two-regime
full-well/read-noise profile into the existing shared camera hypothesis and USER
selection records. High regime applies at ISO >= switch_iso. Explicit experiment
acceptance and a descriptive label are required. Both regimes and branch selection
survive selection evidence in run JSON/HTML. The selected full well determines
base conversion gain, saturation and temporal NoiseProfile with the existing
ISO/100 analogue gain law. Pixels and DNG noise metadata use that same camera;
no independent metadata-only gain switch is introduced. Equal selected electronics
report zero-effect. Default camera/config behavior remains unchanged.

This is a simplified user-supplied effective electronics hypothesis, not measured
Sony dual conversion gain, simultaneous dual readout, or scientific validation.
Config replay retains selected electronics; changing ISO requires explicit profile
reselection. Notebook/catalogue dispatch remains the next integration step before
full optics and final metadata. Full-scale and photon-transfer checks remain
Colab-only. Tests use tiny arrays and real PNG-to-DNG-to-independent-LibRaw/TIFF
execution, checking ISO boundary arithmetic, seeded noise samples, derived DNG
NoiseProfile, provenance, replay, zero-effect and invalid profile rejection.
No model, notebook, download, full-scale job, background service or commit ran.

Validation: all 392 lightweight v5 tests pass under the shared resource lock.
Repository-wide `python -m ruff check .`, changed Python formatting and diff
whitespace checks pass.

## Full sensor integration: executable notebook electronics field

The existing Dual conversion gain catalogue field now accepts a USER dictionary
with all `DualConversionProfile` fields plus `accept_experimental=True`. Null AUTO
preserves existing defaults. Selection follows resolved ISO, including joint
exposure selection, and rejects conflicting explicit Full-well capacity input.
Both electronics records replace their base records without losing ISO evidence.
Invalid modes, missing profiles and unaccepted experiments fail before pixel work.
The notebook passes these records into the existing immutable paired-run config;
completed-pair state distinguishes absent profiles and zero-effect selections.
Run JSON/HTML retain experimental profile and branch evidence. No measured Sony
profile, automatic calibration or simultaneous dual readout is claimed.

Validation: 399 lightweight v5 tests pass under the shared resource lock, including
panel-to-real-DNG/independent-LibRaw/TIFF execution, ISO branch selection, preserved
provenance, input immutability, conflicting edits and static notebook validation.
Repository-wide ruff and changed Python formatting checks pass. Full optics and
final metadata remain next ordered phases. Full-scale and physical calibration
validation remain Colab-only. No notebook, model, download, full-scale job,
background service or commit was started.


## Full optics integration: replayable single-run attenuation

`run_analytic` and `AnalyticRunConfig` now execute `Vignetting` after the
analytic/learned clean RGB stage and before CFA sampling and electron noise.
A separate float32 disk-backed working raster adds 12 bytes per photosite of
scratch disk; processing remains row-bounded and mappings close on failure.
Default zero strength preserves sensor input values and existing random streams.
Nonzero strength requires boolean `accept_experimental_optics=True` before output
creation. Config JSON and its strict schema retain both settings for replay.
Stage evidence and a synthetic registry record appear in JSON and offline HTML,
including changed pixels, minimum transmission, zero-effect/experimental status
and explicit physical limits. This forward prior can compound shading already
in rendered input; it does not infer a lens or undo source shading.

Validation: tiny constant PNG -> DNG -> independent LibRaw/TIFF runs independently
bound attenuation arithmetic in ADC space to one code, verify observable corner
darkening, zero/default status, persisted replay and JSON/HTML evidence. Config
schema coverage includes nonzero settings and invalid fields. All 414 lightweight
v5 tests pass under the shared lock; repository-wide ruff, changed Python format
and diff whitespace checks pass. Paired-run/notebook controls, other optical
effects, full-scale memory/I/O and scientific calibration remain pending. No
notebook, inference, download, full-scale job, service or commit ran.

## Shared paired and Colab vignetting controls (iteration 9)

The single v5 notebook now passes `VIGNETTING_VALUES`, the enable switch and
explicit experimental acceptance into one immutable shared configuration.
Zero defaults preserve existing pixels. Disabled execution retains requested
parameters while reporting zero effect. After successful paired completion,
`effective.json` records the actual 4K stage evidence; both child stage records
are linked in the pair manifest. The catalogue field remains descriptive, with
explicit coefficient controls documented in README rather than inferred calibration.

Tiny real paired DNG/LibRaw/TIFF tests verify shared nonzero attenuation, lower
RAW signal, corner fall-off, zero baseline and identical developed source pairs.
Notebook JSON/schema and AST checks verify defaults, dispatch and completion gating.
All 416 v5 tests pass under the shared lock; repository-wide ruff passes.
Distortion, CA, PSF, diffraction, defocus, motion and final metadata remain pending.
Full-scale memory, GPU/model and physical validation remain unresolved Colab work.
No notebook execution, model download, heavy processing or commit was performed.

## Full optics increment: bounded radial and tangential distortion API (iteration 10)

`optics.Distortion` now executes an immutable synthetic Brown-Conrady backward
sampling map on clean linear RGB before CFA. Four explicit bounded coefficients
(k1, k2, p1, p2) default to zero; nonzero use requires experimental acceptance.
Pixel-center coordinates use half-diagonal normalization. Bilinear interpolation
with edge extension gathers only row strips from an optionally disk-backed source
into a separate float32 memmap. Entire source-domain validation includes unsampled
pixels. HDR values survive without unit clipping; success flushes output and
returns JSON-safe parameter, changed-pixel and out-of-footprint evidence.

This is an explicit synthetic output-to-input map, not inversion of a calibrated
forward projection or lens estimation. It has no antialiasing or flux/Jacobian
correction and can compound distortion already present in rendered input. Pipeline,
shared configuration/schema and notebook integration remain pending, followed by
CA, PSF, diffraction, defocus, motion and final metadata. Physical calibration and
full-scale memory/I/O remain unresolved Colab-only validation.

Validation: 32 new tiny tests independently calculate scalar coordinates and
four-neighbor interpolation, checking radial/tangential effects, edge extension,
HDR, center retention, persisted output, strip invariance, zero identity,
degenerate dimensions and rejection before writes. All 448 lightweight v5 tests
pass under the shared resource lock. Repository-wide ruff, changed-file formatting
and whitespace checks pass. No notebook, model, download, full-frame job,
background service or commit was started.
