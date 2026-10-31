# PRD v5 evaluation and build plan

Evaluated 2026-10-29 before implementation. `PRD_v5.md` is the supplied document copied byte for byte. This evaluation does not rewrite it. The deliverable is one new Colab notebook under `prd_v5/notebooks/`, backed by composable, tested Python. No notebook, model download, full-frame inference, or training is permitted on the 8 GB development Mac. Full-suite tests must use `/Users/rohit/Projects/firstmate/data/exp50mp-heavy-lock.sh`.

## Scope and interpretation

Build the supplied 50 MP experiment. The earlier request mentioned 100/200 MP outputs and diffusion/transformer training from scratch; this document instead requests 50 MP, existing pretrained models and analytic physical simulation. That is a scope divergence, not permission to implement the earlier request.

Maximum settings means full 50 MP, HAT preferred, hybrid inverse ISP, HDR and learned residual requested, full sensor and optics controls exposed and requested, all deliverables enabled, no fast-development preset. It does not mean maximizing noise, ISO, distortion or invented detail, nor allocating the largest possible GPU tile. Retain the PRD's 512/64 SR tiles, 1024/128 requested RAW tiles, mixed precision and memmap accumulation as quality/resource defaults. Generic residual execution reads bounded halo context and writes each core once; notebook RAW overlap controls the per-side context. Predictors retain responsibility for sufficient receptive field and physical-edge padding. Preserve Sony-like 14-bit ADC in a 16-bit storage container. Preserve aspect ratio by default. Identity-sensitive GPS/serials stay user-only, unobservable metadata fabrication remains false. Expose every scalar, profile, coefficient, uncertainty/prior, field mode and experimental switch through the panel or an editable structured configuration section.

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
| ELD | [Vandermode/ELD](https://github.com/Vandermode/ELD), MIT | Physical-noise research reference. The actual repository licence endpoint reports MIT; no ELD checkpoint is required by this build. A paper reference is not a licence grant. |
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

## Implementation review after gnhf

The initial Codex `gpt-6.1-sol` loop made 48 successful commits, then hit its usage limit three times. Following firstmate steering after reset, the same objective/model ran for 12 additional iterations, all successful. No push occurred. Full run records are outside Git in the task's private tmp directory. The PRD remains byte-identical to its supplied copy. None of the 60 loop commits has an agent attribution trailer.

Independent review then connected the remaining optical stages, added an executable full-quality preset and bounded Lanczos resampling, corrected original-source clipping masks, completed output/registry coverage, widened the v5 ADC/ISO domain, and made the notebook bootstrap its checkout and SR dependencies. A tiny end-to-end optical test caught and prevented an output-buffer wiring error. Tests of the full bundle reopen actual generated 12/14/16-bit hypotheses through LibRaw; tests of real model execution remain Colab-only.

| Phase | Implemented executable behavior | Remaining limits |
| --- | --- | --- |
| MVP | JPEG/PNG header/hash/ICC policy, memory-gated decode, aspect/native geometry, seeded immutable camera, analytic inverse ISP, CFA before physical electron noise, ADC, real DNG and independent LibRaw uint16 TIFF, registry/replay/JSON/HTML | Input is 8-bit rendered RGB; stored raster orientation is used. Exact original capture recovery is impossible. |
| Fidelity | Verified user-supplied HAT/Real-ESRGAN x4 checkpoints through spandrel, tiled disk accumulation with visible OOM retry, Lanczos baseline/post-resize, experimental analytic inverse-tone HDR, joint exposure heuristic, gray-world WB and explicit D65 profile matrix selection, sequential paired sources and full-resolution whole/crop differences | HAT checkpoint is not auto-downloaded or invented. No claimed learned-SR gain without held-out truth. LPIPS (official package, tiled), calibrated DeltaE and sampled MTF are implemented and wired into the pair report and notebook, but stay not measured without caller-supplied aligned references or edge evidence. An Adobe `dng_validate` runner (`prd_v5/dng_validate.py`) and a best-effort Linux build script are built but unverified on Colab: the gate reports passed or failed only when the real tool ran, otherwise not measured. A HDRUNet architecture loader for user-supplied checkpoints (`prd_v5/hdrunet.py`) is built; public-checkpoint auto-download for HAT, HDRUNet and RAW-Flow remains unbuilt (no trustworthy SHA-256), see the sections below. Generic residual halo inference is implemented; actual model seam validation remains unmeasured. |
| Learned RAW | Camera-domain/checkpoint-hash/normalization/licence contract, confidence-gated residual application, analytic/hybrid/strict-model-required dispatch and explicit skip evidence | No bundled RAW-Diffusion/SpiralDiff/RAW-Flow checkpoint loader, model-specific camera adaptation or training. A HDRUNet loader (`prd_v5/hdrunet.py`) gives an experimental learned inverse-tone-mapping path for user-supplied checkpoints only; no learned ITM weights are auto-downloaded. Advertised upstream weights do not prove Sony compatibility. This is adapter infrastructure, not a claim those models execute. |
| Full sensor | CFA-domain Poisson shot noise, read noise, PRNU/DSNU, row/column FPN, thermal dark-current expectation, hot/dead maps and realized defect masks, monotone charge compression, supplied ISO-switched effective full-well/read-noise profile, shared absolute-coordinate seeds | Synthetic priors, not measured Sony calibration. Dark electrons share the Poisson stage; shot-noise-off disables that randomness. The affine DNG NoiseProfile describes baseline temporal noise, not nonlinear/body-map variance. Small flat/dark measurements explicitly test that limited model. |
| Full optics | Pre-CFA radial/tangential Brown-Conrady sampling, vignetting, lateral/axial CA, center-to-edge Gaussian PSF, aperture/wavelength/pixel-pitch diffraction approximation, defocus, global line-motion blur; bounded strips/maps and tested halos | Synthetic Gaussian/line approximations, not Airy rings or measured lens MTF. Axial CA has no inferred depth field. Global motion does not reconstruct local object motion. Nearest/reflect boundaries are disclosed. |
| Final metadata/output | Every PRD table field registered with required provenance schema; unknown identity/history stays null. Source/checkpoint hashes, actual optical/sensor settings, DNG tags and explicit methods, canonical DNG/TIFF/preview, clean/noisy electron-equivalent mosaics, masks, JSON/HTML, pair HTML and per-artifact hashes | Undefined physical fields remain UNSPECIFIED. Some registry concepts have no standard DNG tag. Full Adobe conformance, scientific real-RAW/HDR/colour benchmarks and full-size edit sweeps remain unmeasured. |

`presets.full_quality_config()` and the notebook enable the implemented direct sensor/optics switches with modest nonzero priors (the supplied dual-regime profile remains optional), full 50 MP, HDR, Lanczos where a verified HAT checkpoint is absent, and every output. Enabling the full feature set is not a request to maximize destructive noise/blur strengths. Conservative library defaults remain available. GPS, serials, timestamp and other unknown identity/history values are not fabricated. Requested HAT/HYBRID/learned settings remain visible even when missing weights cause the explicitly allowed analytic fallback. Experimental acceptance flags are preselected by this maximum-settings interpretation; they are not licence acceptance for downloaded models.

The notebook contains every parameter-table row as an editable structured request and all required switches. Only implemented estimators/overrides execute: unsupported edits fail clearly, and unknown fields retain explicit unresolved records. Raw tile/overlap controls execute on supplied residual adapters. Global guidance, spectral sensitivity and unobservable capture fields cannot become working estimators by appearing in the panel. Every boolean has a requested/effective state; the final report normalizes states to enabled, disabled, skipped because dependency unavailable, or enabled but no-op. A supplied incompatible checkpoint fails rather than triggering another backend.

Virtual geometry uses a declared 35.9 mm sensor-width hypothesis and derives virtual height/pitch from raster aspect; this is not a recovered or calibrated native Sony sensor. Noise and optical priors can compound artifacts already baked into a rendered source. The HDR branch can still saturate synthetic photosites; saturation is not evidence of recovered highlight truth. The generated-region mask indicates potentially model-generated content/unsupported margins, not a learned-detail truth oracle. Original-source endpoint uncertainty is recorded independently of SR-generated endpoints.

### Validation evidence and remaining Colab work

Local work is limited to small fixtures, unit tests, packaging checks and lint. No notebook, actual model, model download, training, 50 MP inference or reconstruction was run here. The full suite is invoked only through the shared heavy-job lock. Final validation on 2026-10-30: **2,419 tests passed, 18 skipped** in 242.17 seconds through the shared heavy-job lock. Skips cover intentionally disabled full notebook/model execution and absent optional Torch, ExifTool and Haar-cascade facilities. Repository-wide `ruff check .` passes; formatting checks pass for all v5 Python/tests/notebook cells. The built wheel was imported outside the checkout and passed packaged-resource, full-preset replay, low-ISO and revision-identity checks. Notebook JSON/schema and literal defaults validate without execution. The PRD passes a byte comparison with the supplied private copy. All proof/logs and wheel artifacts are outside Git.

Automated pre/post-noise RAW-domain disagreement is now produced from the exported full-resolution mosaics. Windowed native-pixel frequency-band disagreement is also produced for render pairs and requested crops. Registration, ground-truth interpretation and MTF benchmark reporting remain deferred.

Scientific acceptance stays conditional. The generated reports distinguish valid DNG/CFA/registry/output evidence and baseline synthetic photon-transfer measurements from unmeasured seam residuals, real RAW/HDR/colour fidelity, SR-vs-Lanczos truth improvement, and full EV/WB/highlight/shadow/demosaic editability. The notebook produces inspectable artifacts for a Colab experiment; this branch is not empirical evidence that the 50 MP capture meets every research gate. Those validations require real hardware execution, compatible licensed checkpoints and genuine aligned reference data.

## Concrete phase-by-phase build plan

1. MVP: run the committed notebook on Colab, first verifying a tiny independently decoded DNG, then the full 50 MP source pair. Record actual GPU/VRAM/RAM/disk, stage times and output dimensions; retain configuration, hashes and all reports. Do not treat this local unit-test evidence as full-size execution evidence.
2. Fidelity: obtain and verify an official compatible HAT checkpoint and explicit model terms, pin its revision/hash, benchmark tile size/precision on the actual GPU, and compare with Lanczos on genuine RAW-derived held-out pairs. Validate source registration before extending full-resolution reports to RAW pre/post-noise differences, spectra, MTF, LPIPS, calibrated colour and HDR truth.
3. Learned RAW: select an actual camera-compatible RAW-Diffusion/RAW-Flow/SpiralDiff checkpoint and loader, verify its normalization and camera domain, implement and test receptive-field/halo blending, then compare confidence-gated residuals against the analytic baseline. Keep unavailable backends visibly skipped; never add sensor noise inside the learned semantic layer.
4. Full sensor: calibrate real flat/dark frame pairs and ISO regimes when available. Extend baseline photon-transfer evidence to body-map PSD, PRNU correlation, dark-current scaling, defects and nonlinear/dual-gain behavior. Reconcile affine DNG NoiseProfile limitations explicitly rather than claiming nonlinear noise is represented exactly.
5. Full optics: validate distortion/CA/PSF/diffraction/defocus/motion approximations with known synthetic targets and calibrated lens data. Check halos, boundaries and full-frame memory on Colab; assess whether effects compound artifacts already baked into each source. Preserve synthetic provenance without calibration.
6. Final metadata: run independent Adobe DNG conformance and full-size exposure -3/-2/0/+2/+3 EV, WB, highlight/shadow and demosaic sweeps; verify crop/orientation and every populated standard tag against the registry and manifest. Record each acceptance gate as passed, failed or unmeasured. Keep unknowable identity/location fields empty and scientific acceptance conditional until required evidence exists.

## Completion research, 2026-10-30

Rechecked the actual author READMEs through the GitHub API before the completion build. No weights were downloaded on the development Mac.

- HAT's [official README](https://github.com/XPixelGroup/HAT) publishes the [model folder](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0) and names `HAT_SRx4_ImageNet-pretrain.pth`. Repository licence: Apache-2.0. This is rendered RGB SR, not a Sony camera RAW model. A folder URL is not an individual artifact pin; no SHA-256 has yet been verified.
- [RAW-Diffusion](https://github.com/SonyResearch/RAW-Diffusion) documents training and sampling for FiveK Nikon/Canon and NOD Nikon/Sony under Apache-2.0. The README does not supply a ready checkpoint download. Its camera domains must not be relabelled alpha1 II.
- [SpiralDiff](https://github.com/Chuancy-TJU/SpiralDiff) explicitly links [Chuancy/SpiralDiff](https://huggingface.co/Chuancy/SpiralDiff) as its pretrained model repository. Its licence file is MIT despite an Apache-2.0 README badge. The public Hugging Face model metadata endpoint returned HTTP 401 without credentials in this research session. Artifact names and hashes remain unverified, so the advertised link alone cannot justify an automatic download pin.
- [RAW-Flow](https://github.com/liuzhen03/RAW-Flow) links [author checkpoints](https://drive.google.com/drive/folders/1pUWOdyogxZSumnRCv9HcffCR4sl4Ue60), naming `nikon_d700.pth`, `canon_eos_5d.pth`, and `pascalraw.pth`, approximately 350 MB each, under MIT. An individual file URL and trustworthy SHA-256 still need metadata verification without downloading weights locally.
- [HDRUNet](https://github.com/chxy95/HDRUNet) is an author-published learned single-image HDR reconstruction option. The README points to its own `pretrained_model` directory and inference config, trained for NTIRE2021 single-frame HDR. Inspect its actual artifact metadata, licence, normalization and architecture before integrating; it is not calibrated Sony highlight truth.

A missing digest stays unknown. A Git blob SHA-1, a Drive MD5, or a locally invented digest must never be recorded as checkpoint SHA-256. The completion build below must distinguish runnable models from access-limited or unverified artifacts.

### HAT individual Drive artifacts and remaining checksum blocker

Metadata-only author-source inspection on 2026-10-30 resolved the
[official model folder](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0)
to the following individual files. The folder HTML identifies each filename and
file ID; both individual preview pages also return the matching filename.
Only HTML metadata and GitHub source text were read, never checkpoint bodies.

| Variant | Author file preview | Individual download candidate | Folder modification date |
| --- | --- | --- | --- |
| HAT ImageNet-pretrained x4 | [HAT_SRx4_ImageNet-pretrain.pth](https://drive.google.com/file/d/1cxls85ZE7kalhNy47eBJI_L_Lwf9hxRI/view) | [Drive file download](https://drive.google.com/uc?export=download&id=1cxls85ZE7kalhNy47eBJI_L_Lwf9hxRI) | 20 May 2022 |
| HAT-L ImageNet-pretrained x4 | [HAT-L_SRx4_ImageNet-pretrain.pth](https://drive.google.com/file/d/1uefIctjoNE3Tg6GTzelesTTshVogQdUf/view) | [Drive file download](https://drive.google.com/uc?export=download&id=1uefIctjoNE3Tg6GTzelesTTshVogQdUf) | 2 March 2023 |

The download links are candidates constructed from verified Drive IDs, not
tested byte downloads or immutable content pins. Drive file IDs and modification
dates do not establish byte identity. Neither preview page exposed a SHA-256
label or a 64-hex digest. A metadata request to
`https://www.googleapis.com/drive/v3/files/FILE_ID?fields=id,name,size,sha256Checksum,md5Checksum,version`
for each file returned HTTP 403 `PERMISSION_DENIED`, explicitly rejecting
unregistered callers and requiring an API key or other API consumer identity.
This is a metadata authentication blocker, not evidence that the files are
private or that Drive has no checksum. An authorized metadata response containing
`sha256Checksum` could resolve integrity without a local weight download.
No byte size or SHA-256 is asserted here.

The author repository at revision `1638a9a822581657811867bf670717f8371fc3e5`
has no GitHub releases (`releases` API returned `[]`); its complete recursive
tree (`truncated: false`) contains no `.pth`, `.gitattributes` or file named for
SHA/checksum metadata. Consequently there is no Git LFS checkpoint pointer to
inspect in that tree. Repository terms remain
[Apache-2.0](https://github.com/XPixelGroup/HAT/blob/1638a9a822581657811867bf670717f8371fc3e5/LICENSE);
no separate weight licence was established by these metadata checks.

The pinned author test configurations for
[HAT](https://github.com/XPixelGroup/HAT/blob/1638a9a822581657811867bf670717f8371fc3e5/options/test/HAT_SRx4_ImageNet-pretrain.yml)
and [HAT-L](https://github.com/XPixelGroup/HAT/blob/1638a9a822581657811867bf670717f8371fc3e5/options/test/HAT-L_SRx4_ImageNet-pretrain.yml)
both specify RGB input, x4 PixelShuffle output, image range 1, window size 16,
embedding dimension 180, strict loading and checkpoint key `params_ema`.
HAT has six groups of depth six; HAT-L has twelve, so filenames cannot safely be
interchanged with a fixed six-group architecture. These are rendered-RGB SR
models evaluated on bicubic-downsampled image benchmarks, with no camera-specific
Sony RAW calibration. The existing spandrel loader validates architecture, scale
and RGB channels from a verified user checkpoint, rather than treating the
filename as an architecture guarantee.

Disposition: individual official HAT and HAT-L artifacts are now identified;
automatic Colab download remains skipped because the required independently
verified SHA-256 is unavailable through the inspected unauthenticated metadata.
Continue supporting user checkpoint paths with expected SHA-256 and explicit
licence acceptance. Do not replace this precise integrity reason with a claim
that HAT weights are absent. No models, weights or notebook were executed, and
no inference quality or full-size memory guarantee follows from this research.

### RAW-Flow individual Drive artifacts and execution limits

Metadata-only inspection on 2026-10-30 resolved the author-linked
[Drive folder](https://drive.google.com/drive/folders/1pUWOdyogxZSumnRCv9HcffCR4sl4Ue60)
to three individual files. Each preview returned HTTP 200 and its matching
filename. Only HTML and source text were read; no checkpoint body was requested.

| Camera/domain | Individual author preview | Download candidate |
| --- | --- | --- |
| FiveK Nikon D700 | [nikon_d700.pth](https://drive.google.com/file/d/1rzKzm5e3xwH7EBBOd8WmohHwIrD3067i/view) | [Download](https://drive.google.com/uc?export=download&id=1rzKzm5e3xwH7EBBOd8WmohHwIrD3067i) |
| FiveK Canon EOS 5D | [canon_eos_5d.pth](https://drive.google.com/file/d/1rRaVxcsJpGM6UEdju4CqvmcGlua-PZA1/view) | [Download](https://drive.google.com/uc?export=download&id=1rRaVxcsJpGM6UEdju4CqvmcGlua-PZA1) |
| PASCALRAW dataset domain | [pascalraw.pth](https://drive.google.com/file/d/1nAmBJHrUzGvPbC-BM5_hK5IjjpmiZswa/view) | [Download](https://drive.google.com/uc?export=download&id=1nAmBJHrUzGvPbC-BM5_hK5IjjpmiZswa) |

These candidates identify mutable Drive files, not immutable byte pins. No
SHA-256 label appeared in the folder or previews. For each ID, requesting
`https://www.googleapis.com/drive/v3/files/FILE_ID?fields=id,name,size,sha256Checksum,md5Checksum,version`
returned HTTP 403 `PERMISSION_DENIED`, requiring registered API consumer identity.
Thus an authenticated checksum metadata response or independently supplied
expected SHA-256 remains necessary. Public preview access does not establish
checksum API access. No checkpoint SHA-256 is asserted.

The author repository revision is `ddddedf1cc01830a5831f7e44d2d5991492c9e0a`.
Its [complete recursive tree](https://api.github.com/repos/liuzhen03/RAW-Flow/git/trees/ddddedf1cc01830a5831f7e44d2d5991492c9e0a?recursive=1)
returned `truncated: false`, with no checkpoint, `.gitattributes` or checksum
sidecar; there is no checkpoint LFS pointer in that tree. The author releases
API returned `[]`. The pinned [licence](https://github.com/liuzhen03/RAW-Flow/blob/ddddedf1cc01830a5831f7e44d2d5991492c9e0a/LICENSE)
is MIT, copyright 2026 The RAWFlow Authors; no separate checkpoint terms were
found. The camera labels above follow the author README, not Sony calibration.

Source inspection of the [wrapper](https://github.com/liuzhen03/RAW-Flow/blob/ddddedf1cc01830a5831f7e44d2d5991492c9e0a/src/models/rawflow_wrapper.py)
shows RGB input in [0,1], internal [-1,1] normalization and four-channel packed
RAW output at half the input height and width. It builds RGB/RAW autoencoders,
context guidance and a 20-step flow model. Loading uses `weights_only=True` but
`strict=False`; a production adapter must reject missing/unexpected keys rather
than accepting partially initialized models. The wrapper stacks all RGB patches,
retains all predicted patches and overwrites overlaps during stitching. This
does not satisfy bounded full-frame memory or seam-blending requirements.
It also supplies no calibrated confidence output. A bounded adapter needs
streamed patches, consistent global context, explicit CFA/normalization mapping
and an independently justified confidence policy before pre-noise fusion.

The pinned [Nikon evaluation configuration](https://github.com/liuzhen03/RAW-Flow/blob/ddddedf1cc01830a5831f7e44d2d5991492c9e0a/configs/eval/fivek_nikon.yaml)
uses 512-pixel patches, 16-pixel overlap, 256-pixel global context and fusion
disabled. These are upstream settings, not validated 50 MP defaults here.

Disposition: **individual public artifacts identified; automatic pinned
URL+SHA-256 download remains blocked by unavailable authenticated checksum
metadata**. This does not mean weights are absent. Verified user paths remain
supported by the generic residual contract, but that contract is not a real
RAW-Flow architecture loader. Camera mapping, bounded architecture execution
and Colab validation remain implementation work, separate from the digest
blocker. Analytic fallback must remain visible and learned accuracy unmeasured.

Validation: checked official source revision, release/tree metadata, all three
preview filenames and checksum API errors; `git diff --check` passes. No model,
notebook, checkpoint download, background service, commit or push was started.

### HDRUNet individual artifact and inference contract

Metadata/source inspection on 2026-10-30 used `gh-axi api` with `--full` and raw-text Accept headers for source files only. The [recursive author tree](https://api.github.com/repos/chxy95/HDRUNet/git/trees/e9bd4b90a27d7531a38b8c97db05feb368b96dc2?recursive=1) was complete (`truncated: false`). No checkpoint body was requested, downloaded or executed.

| Property | Verified evidence |
| --- | --- |
| Source revision | `e9bd4b90a27d7531a38b8c97db05feb368b96dc2` on the author's `main` branch |
| Individual checkpoint | [pretrained_models/pretrained_model.pth](https://github.com/chxy95/HDRUNet/blob/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/pretrained_models/pretrained_model.pth), 6,662,614 bytes. The actual directory is plural, unlike the README's `pretrained_model` wording. |
| Immutable download candidate | [revision-pinned raw file](https://raw.githubusercontent.com/chxy95/HDRUNet/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/pretrained_models/pretrained_model.pth). A HEAD request returned HTTP 200, `application/octet-stream` and matching Content-Length. This verifies availability, not file contents. |
| Integrity metadata | Tree blob ID `c3eebd40ba4341925a407b7dbac4b8051910a8bd` is Git SHA-1, not SHA-256. HEAD ETag `792c3f5d4d8e96988d5c42c6387d72c0b788becc70cca585af11cb708ced6586` is an opaque HTTP validator: the response does not identify its algorithm or attest that it hashes the checkpoint bytes. Do not promote it to SHA-256. |
| Additional artifact channels | Author releases API returned `[]`. The complete tree has no `.gitattributes`, LFS pointer-sized checkpoint, checksum sidecar or separate model licence. This is a regular Git blob by the available metadata; do not request the blob body merely to search for an LFS digest. |
| Licence/domain | [MIT licence](https://github.com/chxy95/HDRUNet/blob/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/LICENSE), copyright 2021 XyChen. Author README identifies NTIRE2021 single-frame HDR competition data and recommends that data distribution for testing. No Sony alpha1 II calibration or camera-specific domain is established; no separate checkpoint terms were found in the tree. |

The [test configuration](https://github.com/chxy95/HDRUNet/blob/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/codes/options/test/test_HDRUNet.yml) selects `HDRUNet(in_nc=3, out_nc=3, nf=64, act_type='relu')`, scale 1, and `condition: image`. The [dataset loader](https://github.com/chxy95/HDRUNet/blob/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/codes/data/LQ_condition_dataset.py) supplies two identical RGB CHW float tensors. Its [reader](https://github.com/chxy95/HDRUNet/blob/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/codes/data/util.py) divides uint8 by 255 and uint16 by 65535, with no inverse transfer applied in that input path. This must not be confused with the existing clean-camera residual normalization.

The [architecture](https://github.com/chxy95/HDRUNet/blob/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/codes/models/modules/UNet_arch.py) uses two stride-2 reductions, two PixelShuffle upsamplers, SFT conditioning and 2/8/2 residual trunks, ending in a learned image-mask contribution plus a predicted output. That mask is not calibrated confidence. Spatial alignment implies padding tiles to multiples of four and preserving the stride phase across tile origins; this is an implementation requirement inferred from the source, not a tested upstream tiling guarantee. The [loader](https://github.com/chxy95/HDRUNet/blob/e9bd4b90a27d7531a38b8c97db05feb368b96dc2/codes/models/base_model.py) expects a direct state dictionary and removes optional `module.` prefixes. A future Colab loader should use restricted `weights_only=True` loading, strict key/shape validation and bounded halo inference rather than executing the upstream training harness.

Disposition: **public individual artifact found; automatic pinned URL+SHA-256 download remains unavailable because a trustworthy byte SHA-256 was not established by the inspected metadata**. This is not a claim that weights are absent or that a real loader is impossible. A user-supplied checkpoint and independently supplied expected SHA-256 can unblock integrity verification; a real architecture loader, explicit HDR radiometric scaling contract, confidence gate and Colab validation remain to be built. Keep analytic HDR fallback visible and highlight truth unmeasured. Do not infer calibrated radiance or recovered clipped highlights from this model's output.

### Generic residual halo completion

The residual contract now supplies clipped context windows of at most `(tile + 2*halo)` per spatial dimension, with absolute context origins, immutable baseline inputs and core-only confidence-gated writes before CFA/noise. Adapter defaults are 1024-pixel cores and 128-pixel per-side halos; notebook `RAW_TILE`/`RAW_OVERLAP` replace those settings. This is generic execution infrastructure, not a RAW-Diffusion/RAW-Flow/SpiralDiff loader. Global-attention models, stride-phase constraints and insufficient halos still need model-specific handling. Physical-edge padding remains predictor-owned. No seam threshold is marked passed without model evidence.

Small spatial-filter fixtures compare tiled output against an independent whole-image neighborhood computation, including one-pixel cores, partial edge tiles, oversized context, source margins and confidence weighting. The generated-DNG paired fixture also exercises nonzero halo settings through ADC and LibRaw and records them in both child manifests. No learned model was executed.

Validation for this incremental completion: 55 focused residual, notebook-schema, DNG and geometry fixture tests passed; v5 lint and changed-Python formatting checks passed. The tiny DNG tests exposed a legacy writer lookup for `outpaint_bands`; v5 now aliases its equivalent `generated_bands` property without changing serialized geometry. No full suite, notebook execution, checkpoint download or model execution was performed.

### Automatic paired RAW comparison completion

The sequential pair runner now compares both exported Bayer arrays automatically, with bounded row strips and no resizing, alignment or exposure fitting. Reports include sample count, signed bias, MAE, RMSE and maximum absolute error in electron-equivalent units, both overall and for each absolute 2x2 CFA phase (including separate green sites). Requested native-pixel crops retain absolute CFA phase. Unequal shapes visibly remain not measured; malformed or nonfinite arrays fail before a pair completion manifest is published.

Clean arrays represent expected photoelectrons before stochastic/body effects. Noisy arrays represent ADC-referred electrons after pedestal removal, including clipping and quantisation. Their metric difference cannot isolate temporal noise or prove source fidelity. Same geometry, CFA and camera hypotheses do not establish scene registration. The pair HTML and JSON retain these limits and do not mark scientific acceptance passed.

Validation uses tiny generated-DNG pairs and independently computed array differences, odd-origin crops, strip-size invariance, unequal rasters and invalid samples. No model, notebook or full-size reconstruction was executed locally.


### Native-pixel frequency-band comparison completion

Every equal-shape render comparison now includes encoded-sRGB luma-proxy spectra for the full visible raster and optional native-pixel crop. Nonoverlapping patches bound FFTs to 256x256 pixels. Each patch is demeaned and Hann-windowed; window-energy normalization and pixel-weighted aggregation report both image powers and complex-spectrum difference power in five radial bands (0, 1/16, 1/8, 1/4, 1/2, 1 cycles per pixel). The final band includes diagonal frequencies above the axial Nyquist frequency. Partial edge patches are included when both dimensions have at least four pixels; smaller edge slivers are counted as omitted and an entirely too-small crop stays not measured. This is a local, window-dependent disagreement measurement, not MTF, a calibrated radiance spectrum, ground-truth improvement or evidence of source-supported detail. Demeaning intentionally excludes constant brightness differences, which remain visible in the ordinary render metrics.

Known sinusoidal fixtures verify frequency separation and independently expected signal power; additional fixtures verify zero difference for identical spectra, DC removal, crop coverage, partial-edge accounting, bounded FFT allocation and JSON-safe unavailable results. Tiny generated-DNG paired tests verify the existing automatic report path. No model, checkpoint, notebook or full-size reconstruction was run locally.

### Exported metadata coherence completion

The final bundle now rereads every emitted standard tag from the actual DNG directory and compares it with the manifest, `dng.*` registry entries, matching catalogue fields and explicit aliases/groups (ISO, shutter, aperture, black/white levels, matrices and illuminants). This includes AsShotNeutral and preserves the writer's absolute 1e-6 rational rounding allowance; integer and string tags compare exactly. Intentionally omitted optional tags must remain absent. Missing tags, missing registry entries and disagreements produce a failed gate with named mismatches. No pixel raster is loaded for this check.

A passed result covers the emitted standard tags and their mapped registry fields only. The result explicitly lists unmapped fields: full well, physical sensor geometry, provenance, body effects and other concepts do not acquire a standard tag or an independent physical validation through this check. Scientific acceptance remains conditional. A complete simulator-versus-manifest dependency audit is still separate work.

Tiny generated-DNG fixtures exercise nonidentity WB, rational matrices, real tag corruption, registry corruption, missing entries and optional-tag absence. Bundle integration fixtures validate the JSON and HTML gate output. No model, notebook, checkpoint or full-size reconstruction was executed.


### HDR recovered-truth claim audit completion

Every completed bundle audits the original-source endpoint and support masks
in bounded row strips and records their counts in the JSON/HTML HDR gate. The
pipeline writes `hdr_claims.npy` from two sources: pixels whose analytic HDR
expansion exceeds the source clip level, and, when the learned HDRUNet residual
ran, accepted and changed pixels whose corrected radiance, in the same
source-relative linear units, exceeds the clip level on any channel.
`delivery.audit_hdr` passes it to
`validate_hdr_claims`: any claim inside a source-clipped or unsupported pixel
fails the gate (expansion maps clipped endpoints to the radiance cap, so HDR
runs on sources with clipped pixels report `failed`), otherwise it reports
`passed`. The gate is `not measured` only when neither analytic HDR nor the
learned residual ran. Only unit fixtures and tiny end-to-end fixtures ran; the
learned claim path was exercised with a stub constant-residual adapter, not a
real HDRUNet checkpoint. The reusable `validate_hdr_claims`
API accepts an explicit aligned boolean claim mask: an intersection
with a source endpoint or unsupported geometry fails the gate, counting the
forbidden union once. Invalid mask dimensions, types and alignment fail clearly.
The mask describes original source endpoints, including black endpoints, rather
than newly saturated synthetic photosites or post-SR endpoints.

A claim-policy pass is separate from the HDR gate status. Without calibrated
aligned HDR reference data, numerical HDR fidelity and the overall HDR gate
remain **not measured**, even when there are no forbidden claims. Non-endpoint
claims do not establish truth. This audit does not inspect free-form prose or
validate a learned confidence estimate, and it is not an HDR benchmark metric.

Validation: 25 focused claim-policy, tiny generated-DNG bundle and report tests
passed; v5 lint and changed-Python formatting checks passed. Fixtures cover
clipped/unsupported overlaps, strip-size invariance, read-only memmaps, malformed
masks, missing claims and unsupported truth assertions. No notebook, model,
checkpoint download or full-size reconstruction was executed.

### Generated-DNG exposure sweep API

`develop_exposure_sweep` now independently opens a generated DNG and sequentially
exports -3/-2/0/+2/+3 EV linear-light uint16 TIFFs. Camera WB, linear demosaicing,
fixed highlight clipping, no automatic brightness and a caller-supplied baseline
EV keep the edit contract explicit. Memory admission uses the RAW raster before
unpacking, one render is retained at a time, an existing destination is never
overwritten, and a failed sweep removes its newly created directory. Reports
record output paths, multipliers, mean code and zero/saturation sample counts.

Inspection of the installed rawpy Params documentation establishes that
`exp_shift` supports only 0.25 to 8 (-2 to +3 EV). The sweep therefore uses
`bright=2**(baseline_EV+edit_EV)` consistently with gamma=(1,1), rather than passing
an unsupported -3 EV shift. This is developer brightness scaling of unchanged
RAW samples, not a new physical exposure or proof of recovered highlights.
Linear output differs from the ordinary sRGB-transfer development. Highlight
recovery and physical dynamic range remain not measured.

Twelve focused roundtrip tests pass, including actual tiny DNG development,
independent linear scaling checks, source immutability, memory admission,
existing-directory protection, invalid settings and partial-write cleanup.
Automatic bundle/notebook invocation and WB/highlight/shadow/demosaic sweeps
remain deferred. Full 50 MP execution and visual edit acceptance require Colab;
no model, checkpoint download or notebook execution occurred locally.

### Automatic exposure sweep integration

The maximum preset and notebook now request sequential -3/-2/0/+2/+3 EV
exports for both generated DNGs. `AnalyticRunConfig.exposure_sweep` is a strict
boolean with a conservative false library default; notebook
`ENABLE_EXPOSURE_SWEEP` defaults true. Execution uses the actual ISP-derived
BaselineExposure and existing development memory budget. Disabled runs record
that state explicitly. A failed sweep prevents a completion manifest.

Child manifests, pair summaries and JSON/HTML validation retain the measured
developer response; nested edit TIFFs have per-artifact SHA-256 and byte counts.
Allow an additional 3 GB disk for both 50 MP sweeps. The RAW editability gate
stays not measured: WB/highlight/shadow/demosaic sweeps, visual acceptance and
full-size Colab execution remain pending, and brightness edits cannot establish
physical latitude or recovered highlight truth. Tiny real DNG pipeline fixtures
verify replay settings, baseline scaling, exported hashes, pair propagation,
disabled state and failure completion policy. No model or notebook was executed.

Validation for this integration: 110 focused pipeline, configuration/schema,
notebook-static, delivery, pair and roundtrip fixtures passed; v5 lint, changed
Python formatting and diff whitespace checks passed. No full suite, checkpoint
download, notebook execution, commit, push or background service was started.

### Generated-DNG WB sweep API

`develop_wb_sweep` now exports sequential cool/neutral/warm linear-light uint16
TIFFs from unchanged generated DNGs, using explicit user WB derived from camera
multipliers. Strength defaults to 1.25: cool reduces red and increases blue;
warm reverses those gains. Green remains the reference. A missing second-green
multiplier uses the first green. LibRaw normalizes gains, so the fixture validates
channel ratios rather than absolute code scaling. Disabling LibRaw pixel scaling
also disables the intended WB response and is deliberately avoided.

Memory admission occurs from the header before unpacking, a fresh directory is
required, and failed exports remove only that directory. Reports retain exact
gains, RGB means, clipping counts and paths. These edits establish developer
response, not calibrated Kelvin/illuminant accuracy or physical latitude.
Notebook/default/report integration remains deferred, as do highlight/shadow
and demosaic sweeps. No model, weights or notebook was executed locally.

Validation: 19 focused roundtrip fixtures pass, including actual DNG colour-ratio
response, source immutability, output precision, overwrite protection, memory
limits, invalid controls and failure cleanup. V5 lint, changed Python formatting
and diff whitespace checks pass.

### Automatic WB sweep integration

The maximum preset and notebook now request sequential cool/neutral/warm edits
for both generated DNGs. Strict `wb_sweep` and bounded `wb_sweep_strength`
configuration serialize for replay; notebook `ENABLE_WB_SWEEP=True` and
`WB_SWEEP_STRENGTH=1.25` expose them. The actual ISP BaselineExposure and existing
LibRaw memory admission apply. Disabled runs record that state explicitly, and
a sweep failure prevents a completion manifest.

Child manifests, pair summaries and JSON/HTML RAW editability evidence retain
the measured WB response; nested TIFFs receive byte counts and SHA-256 hashes.
Allow an additional 1.8 GB disk for the pair. Relative gains are not calibrated
Kelvin, illuminant truth or physical latitude; RAW editability stays not measured.
Highlight/shadow/demosaic sweeps and full-size visual Colab acceptance remain
pending. No model, weights or notebook was executed locally.

Validation: 122 focused pipeline, configuration/schema, notebook-static, delivery,
pair and roundtrip fixtures passed. After formatting the README and notebook,
all four notebook-static fixtures passed again. V5 lint, formatting and diff
whitespace checks pass. No background service, commit or push was started.

### Generated-DNG demosaic sweep API

`develop_demosaic_sweep` sequentially exports LINEAR/VNG/AHD linear-light uint16
TIFFs from unchanged generated DNG samples. Camera WB, baseline brightness,
highlight clipping, output space and orientation are held fixed. Algorithm
availability is queried from the installed rawpy/LibRaw build and unavailable
methods receive explicit skip reasons. An entirely unavailable sweep stays not
measured. Runtime errors are not disguised as unavailable dependencies.

Header-only memory admission precedes unpacking and output creation; existing
outputs are protected, and failed sweeps remove only their new directory.
Reports retain versions, methods, means and clipping counts. Independent tiny
DNG renders verify exact algorithm output and a textured fixture confirms that
LINEAR and AHD differ. These are developer response measurements, not detail
accuracy or visual editability acceptance. Automatic maximum preset/notebook
and bundle-report integration remains deferred, along with highlight/shadow
sweeps and full-size Colab visual acceptance. No model, weights or notebook was
executed locally.

### Automatic demosaic sweep integration

The maximum preset and notebook now request sequential LINEAR/VNG/AHD edits
for both generated DNGs. Strict boolean `demosaic_sweep` configuration serializes
for replay; notebook `ENABLE_DEMOSAIC_SWEEP=True` exposes the control. Each sweep
uses the actual ISP BaselineExposure and existing LibRaw memory admission.
Disabled runs record that state; unavailable methods retain explicit reasons,
and runtime failures prevent a completion manifest.

Child manifests, paired summaries and JSON/HTML RAW editability evidence retain
algorithm response and library versions; nested TIFFs receive byte counts and
SHA-256 hashes. Allow another 1.8 GB disk for the pair. Developer response does
not establish detail accuracy or physical latitude; RAW editability remains not
measured. Highlight/shadow sweeps and full-size Colab visual acceptance remain
pending. No model, weights or notebook was executed locally.

Validation: 105 focused pipeline, configuration/schema, notebook-static, paired,
roundtrip and delivery fixtures passed. V5 lint, formatting and diff whitespace
checks pass. Integration fixtures cover real DNG edits, replay, nested hashes,
paired propagation, disabled/unavailable states and failure completion policy.
LibRaw's version tuple is normalized to a JSON list in sweep evidence so saved
and in-memory reports agree. No background service, commit or push was started.

### Generated-DNG highlight/shadow sweep API

`develop_tone_sweep` now exports sequential low/high numbered LibRaw highlight
reconstruction and low/high shadow-lift TIFFs from unchanged generated DNGs.
The [official rawpy highlight contract](https://letmaik.github.io/rawpy/api/enums.html#rawpy.HighlightMode)
identifies levels 3 and 9 as favoring whites and colors, not calibrated recovery
strengths. The [Params API](https://letmaik.github.io/rawpy/api/rawpy.Params.html)
provides highlight controls but no shadow-lift parameter. Shadow edits therefore
explicitly apply a post-development rational channel curve
`y=g*x/(1+(g-1)*x)`, `g=2**EV`, to normalized linear-light RGB in bounded row
strips. Default low/high lifts are 0.5/2 EV; callers can supply an increasing pair
up to 4 EV. Black/white endpoints are preserved. This curve changes channel
ratios and cannot undo development quantization or clipping.

Camera WB, linear demosaic, baseline brightness and output transfer are fixed.
Only one render is retained at a time; header-only memory admission precedes
unpacking, existing outputs are protected and failed sweeps remove partial
outputs. Reports record settings, response statistics and library versions.
Highlight accuracy, shadow-radiance accuracy and physical dynamic range stay
not measured. No claim of recovered clipped-source truth is made. Automatic
maximum-preset/notebook/report integration and full-size visual acceptance remain
pending. No weights, models or notebook were downloaded or executed locally.

Validation: all 38 roundtrip fixtures pass, including independent real LibRaw
highlight renders, an equivalent independently evaluated shadow curve, increasing
lift response, endpoint preservation, source immutability, memory admission,
overwrite protection, invalid controls and partial-write cleanup. V5 lint,
changed-Python formatting and diff whitespace checks pass.


### Automatic highlight/shadow sweep integration

The maximum preset and notebook now request four sequential highlight/shadow
edits for each generated DNG. Strict `tone_sweep` and validated increasing
`shadow_lifts_ev` controls serialize for replay; notebook `ENABLE_TONE_SWEEP=True`
and `SHADOW_LIFTS_EV=(0.5, 2.0)` expose them. Execution uses the actual ISP
BaselineExposure and existing LibRaw memory admission. Disabled runs record that
state, and runtime failures prevent a completion manifest.

Child manifests, paired summaries and JSON/HTML RAW editability evidence retain
response statistics, exact settings and library versions. Each nested TIFF has
byte counts and SHA-256. Allow another 2.4 GB disk for the pair. These measured
developer responses establish neither recovered clipped-source truth nor
physical latitude. Highlight/shadow accuracy and RAW editability remain not
measured pending aligned references and full-size Colab visual acceptance.
No model, weights, notebook or full-size reconstruction was executed locally.

Validation: 153 focused pipeline, configuration/schema, notebook-static, paired,
roundtrip and delivery fixtures passed. V5 lint, all v5 Python formatting,
notebook-cell formatting and diff whitespace checks pass. Fixtures verify real
DNG exports, configurable lifts, baseline scaling, nested hashes, report
propagation, disabled behavior and failure completion policy. No background
service, commit or push was started.

### Calibrated CIEDE2000 reference API

Added `colour_metrics.calibrated_delta_e` for supplied aligned trusted-development
uint16 sRGB TIFF references. The implementation uses the existing scikit-image
dependency: its [official colour API](https://scikit-image.org/docs/stable/api/skimage.color.html)
defines sRGB-to-Lab conversion and CIEDE2000 with explicit parametric factors.
This API fixes D65, 2-degree observer and kL=kC=kH=1. Calibration requires caller
records identifying the camera/profile pair, trusted development and alignment,
plus the exact `sRGB_D65_2deg` encoding declaration. No calibration, alignment,
exposure normalization, ICC conversion or HDR validity is inferred from pixels.

Bounded row strips yield native-pixel mean/RMS/maximum DeltaE, including optional
crops. Missing reference/calibration and unequal raster shapes remain not
measured; invalid contracts fail clearly. A supplied record is an attestation,
not independent validation of calibration. No acceptance threshold or recovered
RAW/HDR truth is claimed. Automatic notebook/bundle integration remains pending.

Validation: 12 small fixtures pass, including actual TIFF black/white and red
colour differences, Sharma's published CIEDE2000 Lab pair, identity, crops,
bounded conversion, malformed contracts, dimensions and types. No model, weights,
notebook or full-size reconstruction was executed. V5 lint/format and whitespace
checks pass.


### Paired calibrated colour-reference reporting

`run_analytic_pair` now accepts per-source `colour_references` records containing
an aligned uint16 sRGB TIFF path and the explicit calibration contract. Each
2K/4K developed output receives independent whole-raster and optional native
crop CIEDE2000 evidence in the pair JSON/HTML. A supplied reference also records
its path, byte count and SHA-256 using bounded streaming. Missing references or
calibration and unequal raster shapes retain the API's not-measured state;
invalid or unreadable supplied files fail without publishing pair completion.
Unknown source labels and malformed controls fail before output creation.

Calibration remains caller-attested. A measured colour difference establishes
neither an acceptance pass nor recovered RAW/HDR truth. References are not
resampled or silently shared between sources. Child bundle Colour gates,
notebook controls and maximum-settings wiring remain deferred to a later unit
of work. Tiny real DNG/TIFF fixtures verify metric propagation, independent
numeric agreement, crop coverage, reference hashes, missing truth, malformed
controls and completion failure behavior. No model, weights, notebook or
full-size reconstruction was executed locally.

Validation: 43 focused paired, colour-metric and delivery fixtures passed;
v5 lint, all v5 Python formatting and diff whitespace checks passed. No full
suite, background service, commit or push was started.

### Notebook calibrated colour-reference controls

The maximum-settings notebook now exposes `COLOUR_REFERENCES` alongside the
source and native-pixel crop settings and passes it directly to
`run_analytic_pair`. A commented per-source record shows the exact TIFF path and
four calibration evidence fields. The default empty mapping preserves explicit
not-measured results without fabricating reference data. Supplied records use
the existing bounded CIEDE2000 measurement, reference hashing and pair JSON/HTML
reporting; `COMPARISON_REGION` also applies to reference crops. References must
already match each developed output's native geometry and trusted uint16 sRGB
encoding. No alignment, colour calibration or source sharing is inferred.

The README Colab steps now explain how to supply these records. Child bundle
Colour gates remain separate pending work; measurements appear in the pair
report and do not establish an acceptance threshold or recovered RAW/HDR truth.

Validation: 38 focused notebook-static, colour-metric and paired fixtures passed,
including real tiny DNG/TIFF reference measurement and missing-truth behavior.
Static AST checks verify the empty default and direct reference/crop wiring
without executing notebook cells. V5 lint, Python/notebook formatting and diff
whitespace checks passed. No model, weights, notebook, full-size reconstruction,
background service, commit or push was started.

### Child bundle calibrated colour evidence

Per-source notebook colour references now also reach each child bundle's Colour
validation gate. Single-source callers can pass `colour_reference={"path": ...,
"calibration": ...}` to `AnalyticRunConfig.run` or `run_analytic`, using the same
aligned uint16 sRGB calibration contract as the pair runner. Bounded whole-raster
CIEDE2000 evidence, calibration records and streamed reference hashes appear in
`validation.json`, the child manifest and HTML colour evidence. No reference,
missing calibration or unequal geometry stays not measured. The pair report
continues to retain optional crop measurements independently.

A measured difference is not a threshold pass, WB angular validation or recovered
RAW/HDR truth; overall scientific acceptance stays conditional. Malformed
reference controls fail before source I/O, and measurement failure prevents child
and pair completion manifests. External reference paths remain caller inputs,
not copied or executable replay configuration. No model, weights, notebook or
full-size reconstruction was executed locally.

Validation: 71 focused paired, report, delivery, pipeline and colour fixtures
passed, including actual tiny DNG development, calibrated reference propagation,
missing truth, HTML numeric evidence and failure completion policy. V5 lint,
all v5 Python formatting and diff whitespace checks pass. No full suite,
background service, commit or push was started.

### Sampled-edge MTF API

`mtf.edge_mtf` now measures MTF50/MTF10 from a caller-supplied uniformly
sampled linear-light edge-spread profile with explicit edge-extraction and
linearization evidence. It computes the first-difference line spread, normalized
real FFT magnitude and first downward threshold crossings by linear interpolation.
The [NumPy real FFT contract](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
and [frequency-spacing contract](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html)
were checked before implementation. Profiles are limited to 65,536 samples;
zero padding refines interpolation without adding spatial information. Frequencies
are restricted to both profile and native-pixel Nyquist limits. No crossing is
reported as null with an explicit reason, never an invented bandwidth. Missing
profiles/evidence and flat edges remain not measured.

This is a finite-profile sampled response without derivative correction, smoothing
or windowing, not ISO slanted-edge SFR, calibrated lens MTF, automatic edge
selection or recovered-detail truth. Callers must include both plateaus and
supply linear-light samples normal to an isolated straight edge. First differences
introduce sampling-dependent attenuation; oversampling reduces that bias but does
not establish source detail. Sharpening and noise can produce nonmonotone responses;
the first downward crossing is explicitly the convention. Automatic TIFF edge
extraction and notebook/pair reporting remain deferred.

Validation: 24 small fixtures pass, including analytic Gaussian edge bandwidth,
sharp-edge absent crossings, polarity/gain/offset invariance, immutable inputs,
JSON-safe missing measurements and invalid/bounded inputs. V5 lint and Python
formatting checks pass. No model, weights, notebook, full-size reconstruction,
background service, commit or push was started.

### Bounded developed-TIFF edge extraction

`tiff_edge_mtf` now extracts native-pixel profiles from caller-selected
axis-aligned edge rectangles in developed uint16 sRGB TIFFs. It applies the
[W3C inverse sRGB transfer](https://www.w3.org/TR/css-color-4/#color-conversion-code)
before Rec.709 luminance and averaging parallel to the edge. A read-only memmap,
one-row working buffers and a 1,048,576-pixel region cap bound memory and work;
normal profiles retain the existing 16-65,536-sample limit. Handles close on
success and failure. No edge selection, orientation inference, alignment,
resampling or slanted-edge correction is performed. Missing region/evidence
and flat edges stay not measured; malformed regions and unsupported types fail.

Encoding and isolated straight-edge geometry remain caller attestations, not
inferences from TIFF metadata or an optical calibration. Results retain region,
axis and extraction evidence. Sampled response establishes neither correct
scene detail nor recovered RAW truth. Notebook and pair-report integration
remain pending. No weights, models, notebook or full-size reconstruction ran.

Validation: 33 focused MTF fixtures pass, including real encoded TIFF extraction
in both axes, independent decoded-profile agreement, missing evidence, flat
crops, bounds, encoding and region-size limits. V5 lint, Python formatting and
whitespace checks pass. No background process, commit or push was started.


### Paired sampled-MTF reporting

The pair runner now accepts an explicit `mtf_edge` record containing a native
region, normal axis and caller edge evidence. Both developed TIFFs are measured
independently through bounded linear-light edge extraction; pair JSON/HTML
retains MTF50/MTF10, response curves and evidence. Missing selections or evidence
stay not measured. Malformed controls fail before source I/O and extraction
failure prevents pair completion. Shared coordinates do not establish alignment
or edge suitability. No optical calibration, recovered detail or scientific
acceptance is inferred. Notebook controls and child validation-gate integration
remain pending. Controls remain external caller inputs, not replay configuration.

Validation: all 63 focused paired and MTF fixtures passed; v5 lint, Python
formatting and diff whitespace checks pass. Fixtures compare reports with independent
TIFF extraction, verify actual tiny DNG development, JSON/HTML evidence, explicit
missing measurements, invalid controls and failure completion policy. No model,
weights, notebook, full-size inference, background service, commit or push was
started.

### LPIPS metric API

`lpips_metric.lpips_distance` measures mean LPIPS (v0.1, alex/vgg/squeeze) over
non-overlapping patches (64 to 1024 px, at most 4096 patches) of aligned uint16
sRGB TIFFs under the same attested-alignment contract as `calibrated_delta_e`.
It loads the official [lpips](https://pypi.org/project/lpips/) package lazily
via `load_lpips` (never imported on the Mac). Metadata-only checks on 2026-10-30:
PyPI lpips 0.1.4, wheel sha256 `fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e`
(53,763 bytes), source `3846331df6c69688aec3d300a5eeef6c529435bc8460bd58201c3d62e56188fa`;
repo licence BSD-2-Clause; linear heads ship in the wheel (GitHub `lpips/weights/v0.1`:
alex.pth 6,009 B, vgg.pth 7,289 B, squeeze.pth 10,811 B). The AlexNet ImageNet backbone is
fetched by torchvision from `download.pytorch.org/models/alexnet-owt-7be5be79.pth`
(HTTP 200, 244,408,911 bytes; filename carries torchvision's hash prefix, full
SHA256 not published by metadata, so no digest is recorded). Missing reference or
contract stays not measured; unequal rasters give no implicit registration. Perceptual
distance to a reference is not recovered-detail truth. Fixtures use a stand-in
distance and fake `lpips`/`torch` modules only to test tiling, contract and loader
wiring; no LPIPS network ran. The pair report, notebook controls and the `lpips`
pip install in the Colab setup cell are wired (see below).

### Adobe dng_validate on Colab: precise skip

Metadata and source-layout inspection on 2026-10-30. The official
[Adobe DNG page](https://www.adobe.com/support/downloads/dng/dng_sdk.html)
redirects to a helpx page whose SDK link is script-rendered, so no link could be
extracted from its static HTML. `https://download.adobe.com/pub/adobe/dng/dng_sdk_1_7_1.zip`
returned HTTP 200, `application/zip`, 80,367,206 bytes, Last-Modified 2024-01-02
(this is the conventional Adobe path, not a link verified on the page). The SDK
archive is source code, not weights; its SHA-256 computed on this Mac is
`5366d9abd0e623858573bfaf0291b5e352d704f51b322753a22f82af3ae6e0c2`. Adobe
publishes no digest, so this value is a local observation of one fetch and not
an upstream attestation. Its `DNG_ReadMe.txt` documents build projects only for
macOS (Xcode 12.4 to 14.2) and Windows (Visual Studio 2022 with clang). The
archive has no Linux makefile or top-level CMake project for `dng_validate`
(only third-party libjxl/libjpeg builds), and bundles XMP SDK with boost, expat and
zlib. A Colab Linux build would be a hand-written, untested build of about 80 MB
of C++ and cannot be exercised on this Mac. Adobe SDK terms also apply.

Disposition: **runner built, Linux build unverified**. `prd_v5/dng_validate.py`
runs `dng_validate -v` on the generated DNG and records the exit code, error
lines and tool hash in the bundle's "DNG conformance" gate: passed only on a
clean run, failed on a nonzero exit or error line, otherwise not measured. The
tool comes from an explicit path (`DNG_VALIDATE_PATH` in the notebook), the
`SYNTHRAW_DNG_VALIDATE` environment variable or `PATH`. The notebook setup cell
runs `prd_v5/tools/build_dng_validate.sh`, which downloads the SDK from the
conventional URL, refuses it unless the SHA-256 above matches (a local
observation, not an Adobe attestation), and compiles the DNG SDK sources
directly with g++. It locates the SDK layout (`<top>/dng_sdk/source` beside
`<top>/xmp/toolkit`) and also compiles the bundled XMPCore toolkit (XMPCore,
XMPCommon, shared sources and the zuid MD5) against system expat through an
include shim, linking system libjpeg and zlib with JPEG XL support off. That
script was written without a Linux build project to follow and has not been
run, so compilation or linking may still fail. A failed build is recorded as a visible skip
naming the build error, and the run continues. Unit tests mock the subprocess,
so only the runner's logic is verified, not the tool or the build.

### HDRUNet loader (built, user checkpoint only)

`prd_v5/hdrunet.py` reimplements the pinned upstream architecture (revision `e9bd4b90a27d7531a38b8c97db05feb368b96dc2`, `HDRUNet(in_nc=3, out_nc=3, nf=64, act_type='relu')`) with upstream state-dict key names. Counting its convolution weights and biases by hand gives 1,651,494 parameters (6,605,976 bytes as float32), consistent with the 6,662,614 byte checkpoint once pickle/zip overhead for the key names is allowed for; this is a plausibility check, not proof that the keys match. `load_hdrunet_adapter` requires a caller SHA-256, verifies bytes before unpickling, loads with `weights_only=True`, strips `module.`, and uses `strict=True` so any missing, unexpected or mis-shaped key fails. Inference is padded to stride four with reflect padding and runs inside the existing bounded halo tile loop (tile and halo multiples of four). Upstream output is unbounded gamma-2.24 NTIRE2021 HDR on the input's scale (clipped only at zero), so values above 1 assert radiance above the source clip level. Output and input are both linearized with that gamma and their difference becomes a linear sRGB residual with 1.0 at the source clip level. `apply_residual` maps it into clean camera units through the run's `AnalyticISP` (divide by the HDR headroom `hdr_max_radiance` M when analytic HDR ran, identity at M = 1, then the inverse camera matrix and white-balance/exposure gains), so it stacks over the analytic baseline; this is an experimental mapping, not a Sony radiometric calibration. Confidence is `1 - max_channel_linear_change / max_delta` (default 0.25), not the network's own mask. Accepted, changed pixels whose corrected value, mapped back to source-relative linear sRGB, exceeds the clip level on any channel join `hdr_claims.npy`. Units and claims were tested with a stub adapter at M = 1 and M = 16. Fixture tests use a fake torch module; no checkpoint was downloaded and no model was run on this Mac. The architecture is unverified against real weights until a Colab run with a supplied checkpoint and digest. Auto-download stays blocked because no trustworthy SHA-256 was established, and highlight truth stays not measured.
