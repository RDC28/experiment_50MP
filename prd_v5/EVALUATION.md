# PRD v5 evaluation and build plan

Evaluated 2026-10-29 before implementation. `PRD_v5.md` is the supplied document copied byte for byte. This evaluation does not rewrite it. The deliverable is one new Colab notebook under `prd_v5/notebooks/`, backed by composable, tested Python. No notebook, model download, full-frame inference, or training is permitted on the 8 GB development Mac. Full-suite tests must use `/Users/rohit/Projects/firstmate/data/exp50mp-heavy-lock.sh`.

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
