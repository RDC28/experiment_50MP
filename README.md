# Synthetic RAW 50MP Lab

A Google Colab research experiment that turns approximately 4K AI-generated images into 50-megapixel,
high-bit-depth image masters, and measures what that process can and cannot recover. The full specification
is [PRD.md](PRD.md).

Outputs are **synthetic camera-grade masters**: 16-bit TIFF and linear-light OpenEXR files reconstructed from
a rendered source. They are never camera RAW captures, and every report keeps preserved, interpolated,
inferred, simulated and encoded information apart.

**Current target: [PRD_v3.md](PRD_v3.md)** ("Final PRD: Synthetic Sony-like 50 MP RAW Reconstruction Colab"),
the target that follows cycle 1. It specifies one notebook, `synthetic_camera_raw_50mp_colab.ipynb`, that
builds a synthetic DNG and a 50 MP 16-bit TIFF through an explicit radiance, camera and sensor chain.
[PRD.md](PRD.md) remains the specification of the cycle 1 notebooks described below, which keep working unchanged.

## Synthetic camera RAW notebook (PRD_v3)

`notebooks/synthetic_camera_raw_50mp_colab.ipynb` is the PRD_v3 deliverable. It is backed by the `synthraw`
package in `src/synthraw/` (installed as the top-level package `synthraw` next to `synthetic_raw_lab`), which
uses the PRD class names, and by the tests in `tests/synthraw/`.

**Running it in Colab**

1. Open the notebook in Colab and pick a GPU runtime (Runtime > Change runtime type). The setup cell installs
   the package from GitHub at `REPO_REF`. `REPO_REF` defaults to the development branch
   `fm/exp50mp-v3-synthetic-raw`, which is deleted once it is merged. The cell tries `REPO_REF` first; if that
   branch no longer exists it warns and installs `main`, and `INSTALLED_REF` records what was installed. A
   release should pin `REPO_REF` to the full 40-character SHA of the merged commit on `main`: a pinned SHA is
   installed exactly or the cell fails, it never falls back. Any other value (a tag, a short SHA, another branch)
   is rejected.
2. Set the configuration panel. Every control is an `ExperimentConfig` field. The ones you will usually
   change are: `INPUT_MODE` (`UPLOAD`, `PATH` or `DRIVE`, with `INPUT_PATH`), `OUTPUT_GEOMETRY`
   (`SONY_A1II_NATIVE` 8640 x 5760 or `CINEMA_50MP_VIRTUAL` about 50.1 MP at the source aspect ratio),
   `SR_BACKEND`, `HDR_ENABLED`, the sensor noise and optics switches, and the scene, noise and body seeds.
   `FAST_DEV_RUN` shrinks the planned sensor for a quick check. Learned SR weights are downloaded only inside
   Colab. SwinIR and Real-ESRGAN are permissively licensed and pinned to a repository revision and a sha256
   (`src/synthraw/model_manifest.py`), so they download without any acceptance flag. Licence-gated backends
   and the experimental stubs stay disabled unless you set `ACCEPT_SR_MODEL_LICENCE = True` in the SR cell.
   Backends without a pinned checkpoint fall back and record why. Both 2K and 4K sources run the pinned x4
   checkpoint and then an exact resize, so they use the same backend. `SR_BACKEND = "LANCZOS"` needs no weights.
3. Runtime > Run all. A tiny RAW gate runs first: 64 x 48 DNG fixtures must reopen through LibRaw before any
   reconstruction starts. The chain then runs: input audit, target geometry, SR, HDR/scene-linear,
   camera geometry and exposure posterior, inverse ISP, optics, electrons and noise, CFA, ADC, metadata, DNG,
   LibRaw development, the TIFF/EXR masters, validation, the 2K/4K comparison and packaging. Each section
   runs one stage of a `synthraw.pipeline.PipelineRun`, the same code `run_pipeline` uses.
4. Everything goes into one `/content/synthraw_runs/run_YYYYMMDD_HHMMSS/` folder. The notebook ends by
   building `<run>.zip` without the full-resolution `intermediates/` and `confidence/` arrays
   (`BUNDLE_EXCLUDE_LARGE_ARRAYS = True`). It offers the zip as a browser download unless it is larger than
   `BROWSER_DOWNLOAD_MAX_MB`. It copies the whole run folder, intermediates included, to Drive before writing
   `run_report.json`, mounting Drive first (the usual Colab prompt) when it is not mounted yet. If the mount
   is declined or fails, or the copy fails (for example a full Drive), the run still completes, and the cell
   output, `run_report.json` and the zip warn that the full-resolution intermediates exist only on the Colab
   runtime. After the zip is built, the final report, bundle manifest and log, and the zip itself, are copied
   to Drive too; if that last copy fails the cell prints a warning.

**Outputs** (under the run folder)

| Folder | Contents |
|---|---|
| `raw/` | `synthetic_50mp.dng`: a CFA DNG with `Make = "Synthetic"` and a NoiseProfile matching the simulated ADC |
| `masters/` | `developed_50mp_16bit.tif` (LibRaw development, Adobe RGB), `scene_linear_50mp.exr`, `preview_8bit.png` |
| `metadata/` | `metadata.json` and `provenance.json` (every field tagged D/I/F/M with its dependencies), `dng_metadata.json`, `sony_arw_emulation.json`, `sony_tag_registry.json` |
| `intermediates/`, `confidence/` | stage arrays `01_` to `11_`, source-supported and generated-area masks |
| `validation/` | `validation_results.json`, `validation_report.html`, nine figures, `comparison_2k_vs_4k.html` and `.json`, `libraw_roundtrip.json`, `run_report.json` (`SUCCESS` or `FAILED`) |
| `logs/` | environment, model manifest, checkpoints, bundle manifest |

`validation_report.html` is self-contained. It answers the PRD's four questions, lists the hard gates and
their results, shows measured values next to the configured targets (round-trip SSIM and median deltaE00,
seams, the flat-field noise fit against the NoiseProfile, colour matrices, demosaic variants, -4..+4 EV and
white balance renders, determinism and seeds), and reports the generated and source-supported area
percentages. The hard gates include NoiseProfile agreement with the flat-field fit, the demosaic variants and
the -4..+4 EV and white balance stress tests. If any hard gate fails, both the report and `run_report.json`
say FAILED. QA targets that are missed (round trip, deltaE00, seams, the noise-fit R^2) are listed with the
value measured, and thresholds are never adjusted to make a run pass.

The 2K/4K section runs a second source through the same settings and seeds, so source resolution is the only
thing that changes. It compares the locked settings (a hard gate), the difference map, edge and texture
agreement on named crops, the exposure posterior, white balance, HDR masks and DNG statistics, and sets
`ESTIMATOR_UNSTABLE` when an estimate moves beyond its tolerance. Set `COMPARISON_SECOND_INPUT` to the
other-resolution export of the same composition. If you leave it empty, the notebook makes a half-resolution
copy of the input and the report labels it as derived. The second run doubles compute. It is written to
`<run>_comparison/` and is not bundled.

**Exposure.** The leading ISO hypothesis of the exposure posterior sets the analogue gain, so the DNG ISO and
the simulated noise agree: a higher ISO collects fewer photoelectrons for the same DN and is noisier. The
exposure scale maps the top of the reconstructed HDR headroom (`2^HDR_HIGHLIGHT_EV`) to full well divided by
that gain (the ADC ceiling), so the reconstructed highlights stay below `WhiteLevel` in the DNG and the TIFF
instead of surviving only in the EXR.
The DNG therefore develops darker by default; raise exposure in the RAW developer to bring the midtones up.

**Scientific boundary.** The output is a synthetic sensor reconstruction of a hypothetical exposure. It is
never a genuine camera capture. Every metadata field carries a provenance class: D (deterministic),
I (inferred), F (fabricated, never given a confidence) or M (measured). The DNG `Make` is `Synthetic`, never
`SONY`. No Sony MakerNote blocks or realistic Sony serials are written, and no capture date or GPS is
inferred. The Sony ARW emulation is a JSON description only. The developed TIFF looks flatter and less
saturated than the source. This is inherent, not a colour bug: the inverse ISP removes the source's display
tone curve to recover scene-linear values, and a linear RAW development does not put it back. The
round-trip metrics measure the difference after global alignment and tone normalisation, and also report it
without them.

**Verified so far.** Locally, on tiny fixtures only (no notebook execution, model inference or 50 MP
arrays), the `tests/synthraw` suite and ruff pass. The suite covers every stage and a tiny end-to-end
`run_pipeline` run on 96 x 40 and 48 x 20 PNGs for both geometries with the Lanczos backend. That run
produces the PRD output tree, a DNG that LibRaw reopens bit-identically, `validation_report.html` with every
hard gate passing, and `comparison_2k_vs_4k.html`. The notebook "Run all" test
(`tests/synthraw/test_notebook_run_all.py`) runs by default: it executes every cell after the pip install on
a 48 x 20 PNG with Colab mocked. **Still to verify in Colab:** a full-size 50 MP run, learned SR (SwinIR, HAT,
Real-ESRGAN) with downloaded weights, GPU memory fallbacks, and the validation and comparison reports
produced at full resolution.

[docs/prd_v3_checklist.md](docs/prd_v3_checklist.md) maps every item of the PRD delivery checklist and every
toggle row to its module, tests and state: tested locally, pending Colab, gated stub or not implemented.

## Status

| PRD phase (section 30) | State |
|---|---|
| Phase 0: feasibility and environment | Implemented |
| Phase 1: classical baselines | Implemented |
| Phase 2: pretrained super-resolution | Implemented |
| Phase 3: human and crowd (notebook 01) | Implemented: Phase 2 SR presets (conservative and balanced), YuNet or manual face observations, per-tier identity metrics, local failure recovery, comparisons and master exports; Colab verification pending |
| Phase 3: night scenery (notebook 02) | Implemented: Phase 2 SR presets (conservative and balanced), luminance-region masks, Layer 5B night processing with the PRD 12.6 failure rule, before/after night metrics, comparisons and master exports; Colab verification pending |
| Phase 4: universal notebook (notebook 03) | Implemented: scene statistics and optional face detection classify each input, human and night scenes go to the notebook 01 and 02 pipelines and other scenes to a common pipeline (gated SR ladder, combined masks, universal tone and colour, face protection); every run ends with one verified final image; Colab verification pending |
| Phase 5: synthetic sensor research | Implemented |
| Phase 6: evaluation and optimisation (notebook 06) | Implemented: run collection and comparison of notebooks 00 to 05, identical-crop review sheets, the PRD 31 matrix and E09 full-pipeline ablation, the E10 controlled reference test on freely licensed photographs, performance measurements and the final experiment report (PRD 33, 34); Colab verification pending |

Phase 0 exit criterion: a valid 50 MP master created and exported from a 4K input by classical
interpolation. Phase 1 exit criterion: baselines that are reproducible and measurable. Phase 2 exit criterion:
full-resolution SR runs successfully and produces no major seams. The report of every standard or research run
states whether its run met them. Phase 5 exit criterion: synthetic data is correctly labelled and any DNG output
passes validation; every Phase 5 report states whether its run met it.

## Repository layout

The layout follows PRD section 4.4. Modules are added when the phase that needs them is built.

```text
PRD.md                         specification
README.md
requirements.txt               exact versions the tests were run with
pyproject.toml                 package metadata (minimum versions) and tool settings
configs/baseline_default.yaml  Phase 0/1 parameters, every key documented in src/config.py
configs/super_resolution_default.yaml  Phase 2 profile: the baselines plus pretrained SR
configs/night_default.yaml     Phase 3 night profile (PRD 19.1), Layer 5B keys documented in src/night_processing.py
configs/sensor_research.yaml   Phase 5 parameters, every key documented in src/sensor_config.py
configs/human_default.yaml     human profile, documented in src/human_config.py
configs/universal_default.yaml Phase 4 universal profile (PRD 20.1), documented in src/universal_config.py
configs/evaluation_default.yaml Phase 6 evaluation profile, documented in src/evaluation_config.py
docs/models.md                 SR model evaluation, weights, checksums, licences, commercial-use terms
MODELS.md                      notebook 01 detector artifacts and pointer to SR weight terms
notebooks/00_Feasibility_Baselines.ipynb  Phases 0 to 2
notebooks/01_Human_Crowd_50MP.ipynb  Phase 3 human and crowd
notebooks/02_Night_Scenery_50MP.ipynb  Phase 3 Nighttime Scenery
notebooks/03_Universal_50MP.ipynb  Phase 4 Universal Reconstruction
notebooks/05_Synthetic_Sensor_Research.ipynb  Phase 5
notebooks/06_Evaluation.ipynb  Phase 6 Evaluation and Optimisation
notebooks/synthetic_camera_raw_50mp_colab.ipynb  PRD_v3 synthetic camera RAW notebook (see above)
evidence/colab_runs/           report.md and manifest.json of the captain's Colab runs (no images), collected by notebook 06
src/                           the shared package, imported as synthetic_raw_lab
  config.py                    parameters with defaults, accepted values, effects and warnings (PRD 6, 22.2)
  resolution.py                exact 50 MP target dimensions (PRD 3.5, 25.3)
  ingestion.py                 read-only input store, decoding and validation (PRD 3.3, 3.4, 7.1)
  color_management.py          ICC matrix/TRC interpretation, linear Rec.709 working space (PRD 7.1, 7.2)
  baselines.py                 Lanczos and bicubic interpolation in linear light (PRD 9)
  sr_models.py                 SR weights registry, checksum-verified download, devices, inference wrapper (PRD 10.1)
  super_resolution.py          tiled SR to the exact target, blending, seam checks, memory fallback,
                               hallucination controls (PRD 10.2 to 10.5)
  exporters.py                 16-bit TIFF, OpenEXR and review images with read-back validation (PRD 17, 23.6)
  quality_metrics.py           source-consistency metrics and image statistics (PRD 23.2, 27.4)
  subjects.py                  main subject: faces (OpenCV YuNet, else the Haar cascade) or a saliency centre of interest
  comparison.py                identical evaluation crops across methods, main subject first (PRD 9.2, 26)
  final_output.py              the run's final 50 MP image: selection rule, final files, Colab download and Drive copy
  environment.py               runtime inspection, memory benchmark, storage estimate (PRD 21, 28.1)
  report_builder.py            Markdown report and figures (PRD 27)
  pipeline.py                  the Phase 0 to 2 run with per-stage logging (PRD 5.2, 22.3)
  night_processing.py          Layer 5B night parameters, region segmentation, processing, failure rule and
                               night metrics (PRD 8.3, 12, 12.6, 19, 23.5)
  night_pipeline.py            the Phase 3 night run, figures and report (PRD 19.3, 27)
  sensor_simulation.py         synthetic sensor model, CFA sampling, noise, quantisation, demosaic (PRD 14)
  synthetic_dng.py             research DNG and mosaic TIFF writers with structural and LibRaw validation (PRD 14.6)
  bit_depth_study.py           bit-depth representations and the E01 information test (PRD 14.5, 24)
  sensor_config.py             Phase 5 parameters (PRD 14.4, 22.2)
  sensor_research.py           the Phase 5 run: simulation, exports, E08 round trip, bit-depth study
  sensor_report.py             Phase 5 Markdown report
  human_pipeline.py            human upscaler interface over the Phase 2 SR backend (Lanczos stand-in kept as an explicit option)
  human_config.py              validated human profile and parameter documentation
  human_reconstruction.py      in-memory human reconstruction and identity recovery
  human_comparisons.py         mandatory face/crowd comparisons and mask overlay
  human_run.py                 file-to-report human orchestration and validated masters
  face_preservation.py         manual boxes, provisional tiers, identity metrics and conservative local failure recovery
  region_masks.py              feathered classical region estimates, manual overrides and mask overlays
  crowd_processing.py          bounded simulated microcontrast, protective region blending, and classical crowd consistency warnings
  test_images.py               deterministic synthetic test image
  scene_analysis.py            scene statistics, face observations and content classification (PRD 20.2)
  universal_config.py          the universal profile and the human and night presets it derives (PRD 20.1)
  universal_pipeline.py        the Phase 4 run: routing, SR ladder gates, common pipeline, evaluation, final image
  universal_report.py          Phase 4 Markdown report
  evaluation_config.py         Phase 6 profile: ablation, reference test, aggregation and outcome thresholds
  ablation.py                  the PRD 31 matrix and E09 ablation engine, full-reference metrics, review and blind sheets
  reference_test.py            E10: licence-checked Commons references, degradation models, scoring against ground truth
  run_collection.py            reads run folders and zips of notebooks 00 to 06, comparison tables, identical-crop sheets
  evaluation_report.py         Markdown reports of the ablation and reference-test runs
  final_report.py              the final experiment report: PRD 34 answers, PRD 33 outcome, best Colab configuration
src/synthraw/                  the PRD_v3 package, imported as synthraw (PRD_v3 class names)
tests/                         pytest suite; cycle 1 notebook execution and real-network tests are opt-in
inputs/{human,night,universal}/  read-only copies of ingested sources (not in Git)
outputs/{human,night,universal}/  full-resolution masters per run (not in Git)
weights/                       downloaded model weights (not in Git)
reports/                       per-run report, manifest and figures (not in Git)
```

The reconstruction notebooks end the same way: one final 50 MP image, saved again under `final/` in the run's
masters folder by `final_output.publish` and delivered by `final_output.deliver` (a browser download of the
final TIFF in Colab, or of the EXR when no TIFF was exported, and a copy to `FINAL_DRIVE_DIR` when Drive is
mounted). Notebook 00 chooses it among its outputs with `final_output.select`; notebooks 01 and 02 make one
master each, which is their final image; notebook 03 publishes the master of the route it chose the same way.
Notebook 05 (synthetic sensor research) does not produce a final image.

## Running in Google Colab

1. Open `notebooks/00_Feasibility_Baselines.ipynb` in Colab (File > Open notebook > GitHub, or upload it).
2. The repository is public, so the setup cell clones it without credentials. A Colab secret named
   `GITHUB_TOKEN` (a read-only token) is optional, only for cloning a private fork; alternatively copy the
   repository to Google Drive and set `PROJECT_DIR`. A token is sent with each Git request only and is never
   stored in the clone.
3. To keep the masters, enable `MOUNT_DRIVE` and point `OUTPUT_ROOT` at a Drive folder.
4. Choose the input (`upload`, a `path`, or the `synthetic` test image) and a preset, then run all cells.
   The run ends with its final 50 MP image (see below): with `DOWNLOAD_FINAL` the browser downloads the final
   16-bit TIFF, about 235 MB for a standard run, and with `MOUNT_DRIVE` the final TIFF and EXR are also copied to
   `FINAL_DRIVE_DIR`. The report-and-figures zip is a separate, small download.
5. Super-resolution (`ENABLE_SUPER_RESOLUTION`, on by default) needs a GPU runtime (Runtime > Change runtime
   type) to finish in minutes. The first run downloads about 135 MB of SwinIR weights into `WEIGHTS_DIR`; point
   it at Drive to keep them.

Without `PROJECT_DIR`, the setup cell keeps a clone in `/content/experiment_50MP`, checks out `REPO_REF` there
on every run and reinstalls the package from it with `pip install`, so re-running the cell picks up new
commits. The package declares minimum versions only, so Colab keeps its preinstalled NumPy, SciPy, Pillow and
OpenCV; each run records the versions and the Git commit it actually used.

## Notebook 01

Open `notebooks/01_Human_Crowd_50MP.ipynb` and run all cells. The same Colab setup applies.
The default `standard` preset builds the full 50 MP target from an uploaded image; `research` uses the same
target and currently supplies the same detailed comparisons and metrics, without extra ablations. `quick`
enlarges the source to 1.5x its pixel count (about 12 MP for a 4K source), at most the target, so it upscales
any source smaller than the target on both axes while running faster. `INPUT_MODE="synthetic"` uses a 384 x 256 crowd fixture
with manual ground-truth boxes across all three face tiers. YuNet detection runs by default
(`FACE_DETECTOR="yunet"`); inspect its observations and add `MANUAL_FACE_BOXES` in source pixels as needed.
Manual boxes win overlaps. Use `FACE_DETECTOR="none"` for manual-only observations and select
`PRIMARY_SUBJECT_INDEX` if needed. YuNet and SR weights are cached in `WEIGHTS_DIR` (default `<project>/weights`);
point it at Drive to keep them. A failed run's manifest names the failing stage and gives retry advice for its cause.

Detector versions, bundled XML hash, confidence semantics and commercial-use terms are documented in
[MODELS.md](MODELS.md) and recorded at runtime in the manifest. Detection is never identity proof.

The package API `synthetic_raw_lab.human_run.run_human_crowd` writes TIFF16 and half EXR masters,
a review JPEG, all PRD 18.3 comparisons, a mask overlay, per-tier identity metrics, crowd review flags,
and stage timing/memory in `report.md` and `manifest.json`. The notebook archives reports separately.
Notebook 01 defaults to `SR_BACKEND="swinir"`; `realesrgan` and explicit
`lanczos_stand_in` interpolation are also available. Learned backends install the
`[sr]` extra and expose device, precision, tile size, overlap and weights-directory controls.
Conservative uses the Phase 2 fidelity preset and feeds final reconstruction; balanced uses
the perceptual preset for comparison only. Identity failures fall back locally to source-derived
Lanczos. Learned outputs are Inference (synthesized detail); the stand-in is Interpolation.
Missing SR dependencies or unavailable weights fail explicitly with guidance to select the stand-in.
See [docs/models.md](docs/models.md) for SR weights, checksums, licences and commercial-use caveats.
Identity consistency is not identity proof; visual anatomy and unavailable identity checks require human review.
SR blending, skin smoothing, face-excluding sharpening, tonal contrast, colour balance and seeded grain are exposed.
`skin_texture_smoothing` defaults to 0 (exact no-op), accepts [0, 1], and applies a
5-pixel bilateral filter (linear colour sigma 0.05, spatial sigma 2 output pixels),
weighted by estimated skin including pre-partition facial samples. Each channel change
is bounded to 0.02 times the blend weight. Complete observed face boxes cap the weight
by `regions.primary_face` or `regions.background_faces` (both default 0); overlapping
boxes use the stricter cap. High strengths erase natural texture. This independent
Simulation stage precedes sharpening and identity recovery; it recovers no texture.
Grain is off by default, bounded achromatic random texture with face suppression; its Simulation
stage reports linear-luma high-frequency energy before and after injection, never recovered detail. Human processing retains
multiple target-sized arrays; use `quick` for a smaller first check and inspect measured peak memory.

## Running locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .     # includes PyTorch and spandrel for super-resolution

python -m pytest                          # unit tests, light enough for a laptop
SYNTHETIC_RAW_LAB_HEAVY_TESTS=1 python -m pytest   # also executes the notebooks and, with cached weights, the SR networks
ruff check . && ruff format --check .
```

For a full local run, open the notebook in Jupyter with `INPUT_MODE` set to `synthetic` or `path`, or call the
package directly:

```python
from synthetic_raw_lab import config, pipeline, test_images

source = test_images.write_test_image("/tmp/synthetic_4k.png", 3840, 2160)
pipeline.run(source, config.load_config("configs/baseline_default.yaml"), ".")  # full 50 MP, both baselines
```

A full 50 MP run holds one float32 50 MP buffer (about 600 MB) at a time; peak resident memory is reported
in every run. Presets (PRD 21.6): `quick` builds a 2 MP output, `standard` the full target, and `research`
adds a repeat of every baseline (bit-identical check) and the memory benchmark.

Super-resolution runs with `config.load_config("configs/super_resolution_default.yaml")`. It uses CUDA, then
Apple MPS, then the CPU. The heavy tests also need the weights already cached in `weights/` or in
`$SYNTHETIC_RAW_LAB_WEIGHTS` for the SR cases; they never download. Colab is the verification environment for
full-resolution SR. On a shared machine, run a full 50 MP
SR job on its own: it holds about 1 to 2 GB of device memory plus the 600 MB output.

## What a run does

1. **Ingest and validate** (Preservation). Copies the source to `inputs/<category>/` and makes the copy
   read-only, then decodes the copy with OpenCV, tifffile or OpenEXR without channel swaps or tone changes.
   Records dimensions, aspect ratio, bit depth (and 8-bit data hiding in 16-bit containers), channels, alpha,
   ICC profile, EXIF, embedded text keys, clipping, histograms, a JPEG quality estimate, an 8x8 blockiness
   ratio, corruption and duplicates.
2. **Interpret colour.** Uses the embedded ICC profile (matrix/TRC RGB or grey; LittleCMS and
   table-based sRGB profiles resolve to exact sRGB), assumes sRGB for untagged files and linear Rec.709 for
   EXR, and refuses other profiles unless `source_color_space` overrides them. The working space is
   linear-light Rec.709/sRGB primaries, D65, float32, labelled display-referred, never scene-referred.
3. **Plan the target.** Picks integer dimensions within +/-0.1% of 50,000,000 pixels with the smallest
   aspect-ratio error (even sides by default), or exactly 50,000,000 pixels in `exact` mode when a divisor
   pair keeps the aspect ratio. A 3840 x 2160 source becomes 9426 x 5302 (49,976,652 pixels).
4. **Interpolate** (Interpolation). Lanczos-3 and Keys bicubic (a = -0.5) in linear light with Pillow's
   float32 resampler, one channel and one baseline at a time. Kernel overshoot is kept in float.
5. **Super-resolve** (Inference; optional, Phase 2). For each mode, the network runs at its native x4 on
   overlapping tiles aligned to its attention-window grid, one strip of tiles at a time (never the whole x4
   image), and the overlaps are crossfaded with the quarter next to each tile edge left out. Each strip is
   resampled with Lanczos-3 in linear light to its rows of the exact target, with every sample held within the
   range of the network pixels under the kernel's main lobe (anti-ringing: Lanczos otherwise rings below zero on
   the crisp edges a network draws, which the TIFF clips to black as dark halos). A low-frequency correction,
   applied as a gain so it cannot push a pixel below zero, then pulls tone and colour back to the source (Mode A
   conservative: fidelity network; Mode B balanced: perceptual network; Mode C aggressive, experimental:
   perceptual network uncorrected). Out-of-memory tiles are split, then moved to the CPU. Seam gates and a
   |SR - Lanczos| heatmap show where detail was synthesised. Choices and licences:
   [docs/models.md](docs/models.md).
6. **Export** (Packaging). OpenEXR (half by default, linear Rec.709, chromaticities and a scene-referred
   caveat in the header) and 16-bit sRGB TIFF (embedded sRGB ICC profile, lossless Deflate, provenance JSON).
   Each file is re-read and checked: dimensions, sample type, colour metadata, provenance and a pixel hash.
7. **Measure** (Measurement). Reduces each TIFF master to source size by area averaging and reports PSNR,
   SSIM, MS-SSIM, symmetric Canny edge displacement, CIEDE2000 and the L* distribution shift. LPIPS needs
   pretrained weights and is not computed yet.
8. **Compare** (Measurement). Finds the main subject: faces with OpenCV's YuNet detector (its 0.2 MB model is
   downloaded once into `WEIGHTS_DIR` and checksum-verified; see [docs/models.md](docs/models.md)), else the
   Haar cascade of OpenCV 4.x, otherwise a spectral-residual saliency centre of interest.
   Saves identical crops from every baseline and SR mode next to the source and an amplified difference: the
   primary face first (its eyes, then nose and mouth when the face is larger than a crop), else the centre of
   interest, then the most detailed regions spread apart. Manual `crops` replace them. Also previews and delta E
   maps.
9. **Final 50 MP image** (Packaging). One output becomes the run's final image (PRD 27.3, 32.3).
   `final_output: auto` takes `sr_conservative` when the category is `human` or the primary face is at least 96
   source pixels (PRD 10.2, 11.2 tier 1), `sr_balanced` for other scenes, and passes over any output whose
   exports failed validation or that failed a seam gate, down to Lanczos and bicubic; `sr_aggressive` is never
   chosen automatically. A named `final_output` picks that output. Its TIFF and EXR are saved again, verified
   byte for byte, as `final/final_<name>` in the run's masters folder (hard links where the file system allows).
10. **Report.** Writes `reports/<run>/report.md` plus `manifest.json` with the provenance record of PRD 6.2.
    The conclusions rank every output on each metric (marking the SR modes whose correction makes source
    consistency partly true by construction) and state which output is final and why.

## Phase 3: nighttime scenery

`notebooks/02_Night_Scenery_50MP.ipynb` runs `night_pipeline.run_night` with the profile
`configs/night_default.yaml` (load it with `night_processing.load_night_config`). It segments the PRD 8.3
luminance regions (manual overrides allowed), builds the PRD 19.3 comparisons one at a time (bicubic, Lanczos
and the conservative and balanced Phase 2 SR modes), applies the region-aware Layer 5B night processing to the
configured SR mode (Lanczos without SR) with the PRD 12.6 failure rule falling back block by block, exports the
master to `outputs/night/<run>/`, and reports the PRD 23.5 night metrics before and after. The highlight
shoulder rolls clipped lights off but never recovers clipped radiance. The module docstring of
`src/night_pipeline.py` lists the stages.

## Phase 4: universal notebook

`notebooks/03_Universal_50MP.ipynb` runs `universal_pipeline.run_universal` with the profile
`configs/universal_default.yaml` (load it with `universal_config.load_universal_config`). Default policy: the
most conservative settings that produce measurable improvement.

1. **Analyse and classify** (`src/scene_analysis.py`). Scene statistics (luma distribution, dark share, PRD 8.3
   luminance regions, colourfulness) and faces (manual boxes, then YuNet or Haar) give one of four categories,
   with every cue, threshold and margin recorded: `human` (largest face at least 96 source px, or a crowd, by
   day or night), `night` (low median luma and a large dark share), `night_with_faces` (a night scene whose
   faces do not dominate it), or `general`. A decision within 15% of
   its threshold is low confidence and uses the conservative general settings; `analysis.manual_override`
   confirms or corrects it.
2. **Route.** `human` scenes run the notebook 01 pipeline and `night` scenes the notebook 02 pipeline with the
   first SR rung as their input; their own reports are linked. If that SR input fails its seam gate, the scene
   is re-processed without SR. `general` and `night_with_faces` scenes run the common pipeline: the Lanczos
   baseline, then the SR ladder (`super_resolution.modes`, conservative only by default), where a rung is kept
   only if it passes the seam gate and the measurable-improvement gates (`improvement`: more high-frequency
   energy than Lanczos without losing source consistency or adding clipping); then the combined region and face
   masks and the Layer 5B engine with the universal policy (no lift or colour change, a low highlight shoulder,
   debanding, low region-aware sharpening, the PRD 12.6 failure rule), or the night policy outside the face
   masks. Faces get the same tone and colour without sharpening or denoising, and per-face consistency metrics.
3. **Evaluate.** Every route's final 16-bit TIFF is measured against the Lanczos baseline the same way.
4. **Final image.** The route's TIFF and EXR masters are saved again under `outputs/universal/<run>/final/` by
   the shared `final_output.publish` (hard links where possible, verified by SHA-256) and delivered by
   `final_output.deliver`: copied to Drive when it is mounted and downloaded in Colab.

## Phase 6: evaluation and optimisation

`notebooks/06_Evaluation.ipynb` closes the first cycle. Run it in Colab on a GPU runtime; each part can be
switched off in section 2.

1. **Run collection** (`src/run_collection.py`). Reads every `manifest.json` under the run folders, folders and
   report zips it is given: the captain's notebook 00 runs in `evidence/colab_runs` (included by default), the
   notebook 01, 02 and 03 zips (list them in `RUN_SOURCES`, for example on Drive, or set `UPLOAD_RUN_ZIPS`), and
   its own ablation and reference-test runs. Runs are grouped by source SHA-256 and compared: source consistency
   of every output, seam gates, night metrics, identity by face tier, runtime, SR time and memory. A notebook 03
   run's delegated notebook 01 or 02 run is linked to it. To add runs to the repository's evidence, copy only
   their `report.md` and `manifest.json` into `evidence/colab_runs/<run id>/`.
2. **Visual review sheet.** Identical crops from every run of one source, at 100% from the full-resolution
   masters when they are found (in place or by name under `MASTERS_ROOTS`), otherwise from the review previews,
   which the row label says.
3. **Ablation** (`src/ablation.py`, E02, E07, E09, PRD 31). One input, processed as a 1536-pixel crop at the
   scale a full 50 MP run of the whole input uses, through 17 configurations that switch one stage at a time:
   upscale (bicubic, Lanczos, conservative, balanced, aggressive SR), the scene's specialised processing (the
   notebook 01 region detail, or the notebook 02 or 03 Layer 5B profile), tone only, face handling (notebook 03
   face masks, plus notebook 01 identity recovery on human scenes), grain, bloom and the Phase 5 sensor round
   trip. Every configuration records runtime, peak memory, detail energy, source consistency and clipping; the
   stage contributions compare the full pipeline with it minus each stage. It also runs the PRD 28.3 SR settings
   sweep (precision, tile size, overlap) and the E01 bit-depth study, and writes a labelled review sheet and a
   blind sheet with its key kept separately (PRD 26.3).
4. **Controlled reference test** (`src/reference_test.py`, E10). Two camera photographs from Wikimedia Commons,
   a Holi festival portrait and a forest path at night (both CC BY-SA 4.0), are fetched at run time; the
   licence, author and page are read from the Commons API and checked, and each file is verified against a
   pinned SHA-1. A centre crop of each is degraded to 1/1.7134 of its size (the 6336 x 2688 to 50 MP scale) by
   three models (ideal area sampling; a camera forward model with optics, the Phase 5 sensor and JPEG; an
   AI-render proxy with smoothing, sharpening and JPEG 99) and reconstructed back. Accuracy against the original
   includes the above-source-limit error: about 1.0 is what adding no detail scores, below 1 is real detail
   recovered, above 1 is invented detail that is wrong more than right.
5. **Final report** (`src/final_report.py`). Answers the 18 questions of PRD 34 one by one, each marked
   answered, partial, not measured (naming the run that would measure it) or reasoned, recommends a PRD 33
   outcome from the reference test with the provisional thresholds in `configs/evaluation_default.yaml`, picks
   the cheapest configuration within 0.1 dB of the most accurate one, and states what this upscaling approach
   can and cannot achieve. Blind-review scores entered in the notebook feed questions 13 and outcome A.
   Re-running section 13 with more runs collected updates every answer.

## Phase 5: synthetic sensor research

`notebooks/05_Synthetic_Sensor_Research.ipynb` starts from a processed linear master (a Phase 0/1 Lanczos EXR,
or a source image it first turns into one) and runs `sensor_research.run`:

1. **Simulate** (Simulation). The assumed camera colour transformation (`color_matrix`, the inverse of an
   illustrative crosstalk, and `white_balance_multipliers`), synthetic exposure scaling, CFA sampling (RGGB,
   BGGR, GRBG or GBRG), optional Poisson shot and Gaussian read noise of a stated hypothetical sensor, and
   quantisation to a 10 to 16-bit ADC with black and white levels (PRD 14.2, 14.4).
2. **Export** (Packaging). The mosaic as a labelled 16-bit TIFF and, with `export_synthetic_dng`, a DNG 1.4
   file (preview in IFD 0, CFA image in a SubIFD, ColorMatrix1 for D65, AsShotNeutral). The DNG is accepted
   only after every tag is re-read and matched, LibRaw (rawpy) reads back the same pattern, levels, white
   balance, colour matrix and a bit-identical mosaic, and ExifTool `-validate` passes when ExifTool is
   installed (PRD 14.6).
3. **E08 round trip** (Interpolation, Measurement). Bilinear and Malvar-He-Cutler demosaicing, plus LibRaw's
   AHD from the DNG, develop the mosaic back to linear Rec.709; full-frame PSNR and the Phase 1 metrics on
   full-resolution tiles measure the loss against the master.
4. **Bit depth** (Measurement). 8-bit and 16-bit sRGB integers, half and float linear: quantisation error,
   headroom, shadow levels after an identical push, drift through repeated transforms, real file sizes, the
   mosaic's ADC quantisation and, for a source image, E01: whether expanding its bit depth adds information.

Every mosaic, DNG, sidecar, manifest and report carries the PRD 14.3 label: the data is simulated from
processed RGB, never camera RAW or sensor measurement. Adobe's `dng_validate` is not run; see the report's
conclusions.

Human region estimates (Measurement) use encoded-sRGB colour distances, not semantic segmentation.
Hair is limited to the head band and must differ from a median sample outside/above it by more than
0.10 RGB Euclidean distance. Clothing is limited to the body prior below the face and must be within
0.15 of a central patch immediately below the chin. Feathering stays inside accepted support.
These provisional thresholds can miss similar-coloured hair/background or patterned clothing;
inspect the overlay and supply manual masks when needed.

Mask weights reserve manual overrides first, then use faces (primary before other faces) > hands >
skin > hair > clothing > bokeh > background. Automatic weights consume remaining capacity, so the
per-pixel sum is at most one. Conflicting manual overrides whose sum exceeds one are rejected rather
than silently altered. Bokeh uses a smooth low-local-variance score only outside person regions,
capped at 95% of available background capacity; background receives the remainder. It estimates
smoothness, not optical defocus. Skin colour/texture measurements use separate pre-partition face-chroma samples so face protection
does not remove their evidence; these samples are not enhancement weights.

Notebook 01 classical controls: `general_sr_strength` defaults to 1 and accepts
0 to 1, blending the Lanczos baseline with each selected upscaler output. The
explicit Lanczos stand-in gives identical candidates at every strength; higher
values retain more synthesized detail and possible artifacts with learned backends.
`edge_sharpening` defaults to 0 (bit-identical no-op) and accepts 0 to 1. It applies
a simulated unsharp mask with sigma 1 output pixel and maximum per-channel
linear change 0.02 times strength, excluding complete observed face boxes before
identity recovery. High strengths can amplify halos and artifacts. Both controls
have independent StageRecords. Learned SR synthesizes detail; sharpening is Simulation.

The human profile's PRD 11.6 recovery gate includes
`identity.max_landmark_displacement_normalized` (default `0.05`, to be calibrated
experimentally). When five-point landmarks are available, the maximum point
displacement divided by the source face's shortest side must not exceed this
threshold. Exceeding it triggers reduced enhancement, remeasurement and, if
needed, original-derived Lanczos fallback. Missing landmarks remain unavailable
and require human review; they do not establish identity acceptance.
