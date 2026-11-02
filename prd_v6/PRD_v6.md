# PRD v6: Single-Run, Accuracy-Gated Synthetic 50 MP RAW (Sony A1 II)

Status: draft for captain review, 2026-11-02. Supersedes PRD v5 as the next
experiment. `prd_v5/PRD_v5.md` is unchanged and remains the record of the v5
scope; this document says what v6 keeps, changes and drops.

## 1. Why v6

The captain set two targets for the next revision:

1. **One run.** The notebook is a single Colab "Run all" with no manual steps
   between cells. It runs at maximum quality and automatically downloads the
   results when it finishes.
2. **95% accurate RAW.** The synthetic process should produce a RAW image that
   is 95% accurate against a real camera RAW.

PRD v5 (PRs #14 and #15) delivered a broad, well-tested analytic pipeline, but
it was built to make a *plausible* RAW, not an *accurate* one. Nothing in v5
measures the delivered mosaic against a real camera mosaic. The evidence so far
points the other way:

- The cycle-1 final report found the best SR only +0.31 dB over Lanczos
  against ground truth, and the full pipeline (balanced SR, grain, sensor round
  trip) about -5.5 dB worse than plain Lanczos. Those numbers come from the
  cycle-1 final report; reports are not stored in Git (`reports/README.md`).
- PR #13 measured one real Sony ILCE-1M2 (A1 II) ARW
  (`docs/reference-arw-comparison.md`) and showed that a virtual sRGB camera,
  missing camera white balance and empty HDR headroom make the synthetic RAW
  measurably unlike the real one.
- v5 notebook defaults still use an identity camera matrix and unit white
  balance, an assumed full well of 52,000 e- (the A1 II measures about 33,000),
  about 52 px of synthetic lens distortion in the corners, and a stack of
  fabricated sensor defects. Each of these moves the mosaic away from what the
  A1 II recorded.

v6 therefore changes the product question from "does the output look and parse
like a camera RAW?" to "how close is the output to the RAW the camera actually
recorded, measured in linear RAW space, and is that at least 95%?". Every stage
must earn its place against that measurement.

## 2. Goals and non-goals

### Goals

- G1. A single Colab notebook, `prd_v6/notebooks/synthetic_50mp_v6.ipynb`, that
  completes from "Run all" with no input between cells, using the maximum
  quality preset, and downloads its results automatically (section 9).
- G2. A precise, honest definition of "95% accurate RAW" against a real A1 II
  ARW, measured in linear RAW space, used as the **primary pass/fail gate**
  (section 6).
- G3. A pipeline whose stages are chosen by measured accuracy, not by
  completeness of the camera model. Stages that lower accuracy are removed
  from the max preset, except measured shot and read noise, which is a
  property of the deliverable (section 5.1).
- G4. When the target is missed, the run still completes, still downloads
  everything, says MISSED plainly and names the largest error contributor with
  numbers (section 7). When any step raises an exception, the single driver
  cell's `try/finally` still packages and downloads a summary archive that
  explains the failure (section 9.2). This guarantee covers raised exceptions
  only; for process loss see section 9.2.
- G5. Committed evidence of at least one full Colab run (section 11, M5).

### Non-goals

- Recovering the exact original capture of an arbitrary JPEG or PNG. Without a
  reference RAW of the same scene, accuracy cannot be measured, and v6 makes no
  accuracy claim for such inputs (section 8.2).
- Cameras other than the Sony A1 II. The reference, matrix, levels and noise
  are A1 II measurements.
- Training a new RGB-to-RAW model in this cycle. Section 7.4 lists it as the
  escalation path if the analytic route cannot reach the target.
- Visual "camera look" realism as a goal in itself. Effects that make the
  output look more like a camera but less like the recorded scene are out of
  the max preset (open question Q9).
- Running notebooks, models or full-frame Python on the development Mac. All
  heavy work is Colab-only; local work is unit tests and lint.

## 3. Users and use

The captain opens the notebook in Colab, selects the stated GPU tier
(section 9.1), presses Runtime > Run all, and comes back to a set of
downloaded archives plus a one-page verdict: PASS, MISSED or INVALID against
the 95% target, with the evidence behind it. Optionally the captain supplies
their own 2K and 4K source images by URL in the config cell, and the same frozen
pipeline produces deliverable DNGs for them.

## 4. Terms

- **Reference ARW.** The real Sony ILCE-1M2 capture measured in PR #13:
  raw 8672x5784, visible 8640x5760, RGGB, 14-bit, black 512, LibRaw maximum
  16383, practical clip 15360, ISO 100, 1/250 s, f/8, 28 mm. Not committed to
  Git (106 MB, licence unknown, see Q4).
- **Benchmark render.** A rendered RGB image made from the reference ARW by a
  fixed, declared recipe (section 6.1). It plays the role of "the JPEG/PNG the
  user has". The pipeline sees only the render and camera-model constants.
- **Delivered DNG.** The pipeline's main output: a 14-bit RGGB DNG with the
  visible area 8640x5760.
- **Linear RAW space.** Black-subtracted photosite values in DN, before
  demosaicing, white balance, colour matrix or tone curve.
- **Camera-model constants.** Facts true of every A1 II frame: CFA pattern and
  phase, black and white levels, clip point, colour matrix, conversion gain,
  read noise, full well. The pipeline may use these.
- **Frame-specific facts.** Facts true only of the reference frame: its as-shot
  white balance, exposure, pixel values. The pipeline must not use these, except
  in clearly labelled oracle diagnostics (section 7.2).

## 5. Pipeline

### 5.1 Stage decisions

Each v5 stage is classified by its expected effect on accuracy against the
reference. "Expected" means reasoned from the measurements we have; section 11
M3 replaces every expectation with a measured ablation.

| v5 stage | Expected effect on accuracy | v6 decision |
| --- | --- | --- |
| Input decode, ICC handling, geometry planning to 8640x5760 | Required | Keep |
| Exact inverse sRGB transfer (piecewise W3C) to linear light | Helps; any approximation is pure error | Keep, exact form only |
| Camera matrix: identity (virtual sRGB camera) | Hurts; R and B planes systematically wrong | **Replace** with the measured A1 II matrix (`src/synthraw/camera_profile.py`) |
| White balance [1,1,1] | Hurts; wrong channel ratios in RAW | **Replace** with an illuminant estimate (section 5.3) |
| Fidelity SR, HAT-L (x2 or x4), PSNR-trained | Small help; +0.31 dB best case in cycle 1 | Keep, scale and self-ensemble chosen by ablation |
| GAN or perceptual SR (Real-ESRGAN class) | Hurts; invents texture that is not in the scene | **Drop** |
| Image-domain grain | Hurts; adds error that is not sensor noise | **Drop** |
| Back-projection consistency (output, downsampled by the render model, must equal the input) | Helps; removes SR colour and level drift | **Add** |
| Synthetic optics: distortion, CA, PSF, defocus, motion, diffraction, vignetting | Hurts; the real lens already put these into the source, so adding them again double-applies them; distortion alone moves corners about 52 px | **Drop** |
| Analytic HDR expansion of unclipped tones (v5 `HDR_MAX_RADIANCE` 16) | Hurts outside clipped regions; changes tones the render already fixed | **Drop** outside clipped regions; keep clipped-region extension only if ablation shows no loss |
| HDRUNet | Unverified on this camera; changes unclipped tones | **Drop** from max preset |
| Learned RGB-to-RAW for other cameras (RAW-Flow Nikon/Canon/PASCALRAW, RAW-Diffusion, SpiralDiff) | Hurts; wrong spectral response, no A1 II checkpoint exists | **Drop** |
| Mosaic to RGGB at the reference CFA phase | Required | Keep |
| Saturation at the A1 II clip (15360) | Helps; matches real highlight behaviour | Keep, at the measured clip, not 16383 |
| Shot and read noise | Costs accuracy by design: the delivered DNG's independent noise adds directly to the P1 error (about 2-3% NRMSE near the median, section 7.1); needed for a believable RAW | Keep, at **measured** A1 II values: gain about 2.08 e-/DN (G), read noise 4 e- (upper bound), full well about 33,000 e-; exempt from the rule below |
| PRNU, DSNU, FPN, dark current, hot and dead pixels, nonlinearity | Hurts; fabricated patterns are not in the reference (its measured hot-pixel rate is 6e-6 and dead 2e-8) | **Drop** |
| ADC quantization to 14 bit, black 512 | Required | Keep |
| DNG writer, LibRaw TIFF, `dng_validate` | Required | Keep |
| Edit sweeps (exposure, WB, demosaic, tone) | No effect on the gate; useful evidence of editability | Keep, outputs excluded from default downloads |
| Colour references, LPIPS, MTF, render-space metrics | Reported only; not RAW accuracy | Keep as secondary report |

Rule for the frozen max preset: a stage stays enabled only if the ablation on
the benchmark shows it changes the primary accuracy score A (section 6.3) by
-0.1 percentage points or better on every plane. A stage that helps one plane
and hurts another by more than that is dropped or made plane-neutral.

Shot and read noise are exempt from this rule. They are a property of the
deliverable, not an accuracy stage: the delivered DNG is a noisy camera RAW
(Q6 default) and the gate scores that file, so noise stays on even though it
lowers A. Its cost is reported every run as ladder rung L1 (section 7.2).

### 5.2 v6 order

```
source RGB (8-bit sRGB render)
  -> decode, exact inverse sRGB -> linear RGB
  -> fidelity SR (HAT-L, fp32, tiled, self-ensemble if it passes ablation)
     to 8640x5760
  -> back-projection consistency against the source
  -> undo illuminant (white-balance gains from the illuminant estimate)
  -> sRGB(D65) -> XYZ -> A1 II camera RGB (measured matrix)
  -> exposure placement (camera-model prior; see 6.2 for how the gate treats it)
  -> clean mosaic, RGGB at reference phase            [clean DNG written here]
  -> electrons, shot + read noise at measured values, clip at full well
  -> ADC 14-bit, black 512, clip 15360                [delivered DNG]
  -> DNG, LibRaw TIFF, dng_validate, reports
```

### 5.3 Illuminant

A white-balanced render does not contain the scene illuminant. The pipeline has
to assume one, and the reference shows how much this matters: the A1 II as-shot
neutral is [0.4883, 1, 0.5636] and its daylight neutral is
[0.3744, 1, 0.7402]. Assuming daylight on this frame puts the R plane about
23% low and the B plane about 31% high before any other error.

v6 estimates the illuminant from the render using a camera-model prior
(for example, the A1 II daylight-to-tungsten locus plus scene statistics) and
records the estimate and its uncertainty. It must not read the reference
frame's as-shot neutral. Whether the captain allows a user-declared illuminant
in the gated run is open question Q2.

## 6. The 95% accuracy target

### 6.1 Benchmark construction

The benchmark is built inside the notebook from the reference ARW, and the
pipeline is not told the recipe beyond "8-bit sRGB render of an A1 II frame".

Render R1 (gated):

1. Decode the ARW with rawpy (version pinned in `requirements.txt`, currently
   0.27.1): camera white balance, sRGB output primaries, linear gamma (1, 1),
   no auto-bright, highlight mode clip, AHD demosaic, 16-bit output, no flip.
2. Area-downsample in linear light by exactly 2.25 to 3840x2560 (the 4K
   benchmark) and by exactly 4.5 to 1920x1280 (the 2K benchmark). Both ratios
   divide the 8640x5760 visible area exactly, so the render is pixel-registered
   to the reference by construction and no alignment search is needed.
3. Apply the exact sRGB transfer function, quantize to 8 bit, save as PNG with
   an sRGB ICC tag.

Render R2 (stress, reported only): R1 plus a standard filmic tone curve and
JPEG quality 90. It shows how fast accuracy falls when the render is less
invertible, which is the realistic case for user photos.

The camera matrix LibRaw used for R1 is recorded in the benchmark manifest. If
it differs from the measured matrix in `camera_profile.py`, the run reports both
and the gap is attributed in the error budget; the pipeline keeps using the
measured matrix because it is a camera-model constant.

### 6.2 Comparison space

- Reference x: the ARW visible area 8640x5760, black-subtracted, per photosite.
- Candidate y: the delivered DNG visible area, decoded with the same rawpy,
  black-subtracted using the DNG's own black level.
- The DNG visible area must be 8640x5760 at the reference CFA phase. A size or
  phase mismatch makes the run INVALID, not a low score.
- Exposure: a single global scalar k = sum(x*y) / sum(y*y), fitted over the
  mask and shared by all three planes. Absolute exposure (scene radiance times
  shutter, aperture and ISO) cannot be recovered from a render whose exposure
  was normalized, so one scalar is allowed and reported in EV. Because it is a
  single scalar, white balance, colour matrix and plane-ratio errors still
  count in full. Whether the captain allows even this scalar is Q1; the
  unfitted (k = 1) score is always reported.
- Mask: the visible area minus a 16 px border (8608x5728 photosites), minus
  sensor-clipped reference photosites, defined as raw value (before black
  subtraction) >= 0.98 x 15360 = 15052.8 DN; no truth is available there. The
  source-clipped pixels of the 8-bit render are kept in the mask and reported
  separately, because recovering them is part of the job. If sensor-clipped
  photosites exceed 1% of the visible area excluding the border, the reference
  frame is unsuitable and the run is INVALID. The border is not counted
  towards that 1%.

### 6.3 Primary gate

Both conditions must hold on the 4K benchmark, R1 render, delivered DNG, for
each colour plane p in {R, G, B} (the two green sites pooled into G):

- **P1, energy accuracy.**
  A_p = 1 - sqrt( sum((k*y - x)^2) / sum(x^2) ) >= 0.95.
  Equivalently, the normalized RMS error in linear RAW is at most 5% of the
  reference signal energy in that plane.
- **P2, photosite agreement.** At least 95% of masked photosites in the plane
  satisfy |k*y - x| <= max(0.05 * x, 3 * sigma), where
  sigma = sqrt(sigma_ref(x)^2 + k^2 * sigma_syn(y)^2), sigma_ref uses the
  measured A1 II gain and read noise, and sigma_syn uses the pipeline's own
  noise parameters (zero for a noise-free DNG).

The headline "accuracy" printed by the notebook is min over planes of A_p,
shown as a percentage, with the P2 fractions next to it. PASS needs P1 and P2
on all three planes.

Why both: P1 catches large errors in bright, high-energy areas and in colour.
P2 catches errors spread across many small photosites (shadows, fine texture)
that P1 lets through because they carry little energy. The 3-sigma term exists
because two independent noisy captures of the same scene cannot agree better
than their noise; with it, a perfect pipeline loses only about 0.3% of
photosites to noise, so the 95% bar is spent on real scene error.

### 6.4 Why this definition is honest and not trivially met

Definitions that were considered and rejected:

| Candidate | Why rejected |
| --- | --- |
| 1 - MAE / full scale | An all-black image scores about 88% on this frame (G median is 1690 DN of a 14,848 DN range). Trivially met. |
| R^2 >= 0.95 | Weaker than P1: for typical scenes NRMSE 5% corresponds to R^2 of about 0.995. 95% R^2 tolerates errors near 20%. |
| PSNR, SSIM, LPIPS or Delta E in developed sRGB | Measure the render, not the RAW. A tone curve hides shadow and highlight errors, and they say nothing about channel ratios in camera space. |
| Per-channel gain fit before scoring | Silently forgives white-balance and matrix errors, which are the largest expected errors. |
| Scoring on heavily downsampled images | PR #13 showed downsampled comparisons hid high-frequency differences. v6 scores every photosite. |
| Clean mosaic as the gated output | The delivered DNG includes noise; the user gets that file. The clean score is reported as a diagnostic. |

Sanity checks, run on every benchmark and shipped in the report.

Hard checks. Each must give the expected result or the run is INVALID (the
metric is broken, so no verdict is possible):

- Reference against itself: A = 1.000, P2 = 100%.
- Reference against an independent noise realization of itself (from the
  measured noise model): P1 at least 0.97 and P2 at least 99%. This is the
  noise floor; it proves the target is not unreachable because of noise.

Sensitivity checks. Each perturbation is applied to the noise-floor
realization and scored against the reference. Compared with the unperturbed
noise-floor score, it must lower min A_p by at least 1 percentage point and
lower the P2 fraction (on every plane, or on R for the R gain). These are
monotonic checks, not gate failures, because how far a perturbation falls is
scene dependent. A check that misses its margin is flagged as a
metric-sensitivity warning in the verdict and report; it does not make the run
INVALID.

- Constant mean predictor (expected A about 0.2 to 0.5, scene dependent).
- Shift by 1 photosite of the same colour (2 raw photosites).
- R plane gain +10%.
- Gaussian blur of each CFA plane with sigma 2 raw photosites (1 same-colour
  site).

### 6.5 Reported, not gated

- The same metrics on the 2K benchmark (gated only if Q3 says so).
- R2 stress render results.
- Clean mosaic (noise-free) score.
- Unfitted score (k = 1) and k in EV.
- Diagnostic per-plane white-balance-fitted score, labelled as diagnostic.
- L1 variant of P1.
- Band split: error energy below and above the source Nyquist frequency, so
  detail loss is separated from colour and level error.
- Tone zones: shadows, midtones and highlights by reference value, and the
  source-clipped region on its own.
- Error maps per plane (PNG, full resolution in the data bundle, downscaled in
  the summary), and error histograms.
- Secondary noise gate (pass/fail, not part of the 95% target): the delivered
  DNG's G-plane conversion gain, measured by photon transfer
  (`src/synthraw/realism.py`), is within 15% of the reference's. Read noise is
  reported only, because single-frame estimates are scene-contaminated upper
  bounds.

### 6.6 Run status

- **PASS**: integrity gates pass, hard sanity checks pass, primary gate
  passes.
- **MISSED**: integrity and hard sanity checks pass, primary gate fails. The
  report states the gap per plane and the largest contributor from the error
  budget (section 7.2).
- **INVALID**: integrity or a hard sanity check fails, the reference is
  unsuitable, one of driver steps 1-4 raised (section 9.2), or a required
  stage silently did nothing. No accuracy claim is made.

A failure in driver step 5 (production) or step 6 (sweeps and secondary
reports) does not change the PASS or MISSED verdict already computed by the
benchmark; the verdict and the banner mark it "completed with errors" and
name the failed step.

Thresholds are fixed in code and in this PRD. The notebook never relaxes a
threshold, never retries with different settings to reach a pass, and never
reports a diagnostic score as the headline. In every status, results are
packaged and downloaded, unless the runtime process itself is lost
(section 9.2).

### 6.7 Integrity gates kept from v5

DNG parses with LibRaw and passes `dng_validate`; CFA pattern and phase match;
metadata is coherent (levels, matrices, crop); no tile seams; provenance and
hashes are complete; every enabled stage changes its input (no silent no-op).

## 7. Feasibility

### 7.1 Is 95% reachable?

Honest answer: plausible on the G plane at 4K, unlikely on R and B unless the
illuminant is known, and unlikely at 2K. v5 as shipped is expected to fail by
construction; this has not been measured yet and M1 will measure it.

Error sources, with current estimates to be replaced by measurement in M1-M3:

| Source | Estimated cost in NRMSE | Notes |
| --- | --- | --- |
| Detail lost to downsampling, 4K (x2.25) | about 3-6% | SR recovers a fraction; cycle 1 suggests a small one |
| Detail lost, 2K (x4.5) | about 6-10% | Alone likely exceeds the 5% budget |
| Illuminant error | up to 23% (R) and 31% (B) with a daylight assumption on this frame | Passing needs roughly 1 degree angular error, about 4% per channel |
| 8-bit quantization and sRGB clipping | small in midtones, large in deep shadows and clipped highlights | Shadows are where P2 bites |
| Demosaic and colour-matrix mismatch between R1 and the inverse | unknown until M1 | Recorded in the benchmark manifest |
| Noise (two independent realizations) | about 2-3% near the median, less in highlights | Bounded by the noise-floor check |

So the target is a real stretch: on G at 4K the budget is tight but possible;
on R and B the illuminant alone can exceed it.

### 7.2 Error budget ladder

Every benchmark run also scores a ladder of oracle variants. Oracles use
frame-specific facts and are labelled as diagnostics; they never feed the
delivered DNG or the headline.

| Rung | What changes | Cost it isolates |
| --- | --- | --- |
| L0 | Delivered DNG | Headline |
| L1 | Clean mosaic (no noise) | Noise |
| L2 | L1 with the oracle illuminant (reference as-shot neutral) | Illuminant estimation |
| L3 | Full-resolution R1 render (no downsampling) through the same inverse | Resolution and SR |
| L4 | L3 with the oracle illuminant | Remaining inverse ISP error (matrix, demosaic, quantization) |

The verdict text names the rung step with the largest drop in A per plane, for
example "MISSED: A = 91.2% (R). Largest contributor: illuminant estimation
(-5.8 points on R). Next: resolution (-2.1 points)."

### 7.3 What changes from v5

Covered in section 5.1. In short: replace the identity matrix and unit white
balance with the measured matrix and an illuminant estimate; use measured noise
and levels; add back-projection; drop GAN SR, grain, synthetic optics,
fabricated sensor effects, unclipped HDR expansion, HDRUNet and other-camera
learned models.

### 7.4 If the target is missed

The run reports MISSED with the ladder, and the notebook output is still the
best measured pipeline. The decision on what to do next belongs to the captain
(Q10). The escalation options, in order of expected value:

1. If L2 vs L1 dominates: accept a user-declared illuminant (Q2) or invest in a
   better illuminant estimator.
2. If L3 vs L1 dominates: accept the 4K-only claim, or evaluate stronger
   fidelity SR.
3. If L4 is itself below 95%: the analytic inverse is the limit; train or
   fine-tune an A1 II-specific RGB-to-RAW model on paired A1 II frames. This
   needs many more reference frames and is a separate PRD.

## 8. Inputs

### 8.1 Benchmark inputs

- Reference ARW, fetched by URL and checked against a pinned SHA-256
  (hosting is Q4).
- HAT-L checkpoints (`HAT_SRx4_ImageNet-pretrain.pth`, and SRx2 if the ablation
  selects it), Apache-2.0, fetched from a stable mirror (a GitHub release asset
  on this repo, with the licence notice) and checked against a SHA-256 pinned
  on Colab in M0. The official Google Drive folder is the source of truth but
  is not used for runtime downloads because Drive quotas fail intermittently.

### 8.2 Production inputs (optional)

`SOURCE_2K_URL` and `SOURCE_4K_URL` in the config cell. When set, the frozen
pipeline also runs on them and produces deliverable DNGs. There is no reference
RAW for these, so their outputs carry no accuracy claim; the report says the
pipeline measured X% on the benchmark and that this number does not transfer
automatically to other scenes. When unset, the run is benchmark-only (Q5).

## 9. Single-run notebook

### 9.1 Runtime tier

- Required: Colab Pro or Pro+ with an **A100 40 GB GPU and High-RAM** (about
  83-89 GB host RAM observed in v5 sessions).
- Accepted with a warning: L4 24 GB (same results, slower SR).
- Rejected at preflight: T4 and CPU-only. The run stops before any heavy work
  and says why.
- Free disk: at least 80 GB at preflight (v5 needed 15-30 GB for two runs;
  v6 adds the ladder, two benchmarks, the reference and archive staging).
- Expected wall time on A100: 1.5 to 3 hours for the benchmark at both
  resolutions with the ladder, plus about 30 minutes per production
  resolution. These are estimates; M5 replaces them with measured stage times.
  SR outputs are cached and shared by ladder rungs that use the same input.

### 9.2 Cell contract

The notebook has two code cells. Colab's Run all stops at the first cell that
raises, and a `try/finally` cannot span cells, so all work that can fail runs
inside one driver cell.

1. **Settings cell.** Literal assignments only, so it cannot fail, with
   defaults that need no edits: `PRESET = "max"`, `REPO_REF` pinned to a
   release commit (not `main`), `SOURCE_*_URL = None`, `DOWNLOAD_ALL = False`,
   `DRIVE_MIRROR = False`.
2. **Driver cell.** One `try/finally`. The `try` runs these steps in order:
   1. **Drive mount**, only if `DRIVE_MIRROR` is true. It is the first step, so
      the one authorization prompt happens at the start and never later.
   2. **Setup.** Clone at `REPO_REF`, install pinned packages that do not
      require a runtime restart (a restart would break Run all; the notebook
      checks this and fails with a clear message if a package would need one),
      build `dng_validate`.
   3. **Preflight.** GPU, VRAM, RAM, disk, package versions; download the
      reference ARW and checkpoints and verify every SHA-256. Any failure stops
      here, before heavy work.
   4. **Benchmark.** Build R1 and R2 at 4K and 2K, run the frozen pipeline and
      the ladder, score, sanity checks.
   5. **Production.** Only if source URLs are set.
   6. **Sweeps and secondary reports.**

   An exception in steps 1-4 ends the `try` and makes the run INVALID. Steps 5
   and 6 are each guarded on their own: an exception there is recorded with
   its traceback, the run is marked "completed with errors" next to the
   benchmark's PASS or MISSED verdict (section 6.6), and the next step still
   runs, so a step 5 failure does not skip step 6.

   The `finally` block is **package and download**, so it runs after success
   and after an exception in any step above. It records any exception and
   traceback in the run state, writes the verdict and manifest, builds
   archives from whatever the run folder holds, triggers downloads
   (section 10) and ends with a printed verdict banner. Its code is defined in
   the notebook itself and uses only the standard library and
   `google.colab.files`, so it works even when the clone or install failed.

   This guarantee covers raised exceptions only. A `finally` block does not
   run when the runtime process itself is lost (an out-of-memory kernel crash,
   a runtime disconnect or a recycle). For those cases the mitigations are the
   summary archive written early and updated during the run (section 13), the
   Drive mirror when enabled, and `download_results()` (section 10.3).

Never used anywhere: `files.upload`, `input()`, widgets that wait for a click,
`/content` paths that assume a manual upload, or silent fallbacks (for example,
Lanczos instead of HAT-L) in the max preset. If HAT-L cannot be loaded the run
is INVALID for the max preset (Q7).

### 9.3 Maximum quality settings

"Max options" in v6 means every stage that helps or holds accuracy is on, at
its highest-quality setting:

- HAT-L, fp32, tile 512 with 128 px overlap, scale (x2 or x4) and x8
  self-ensemble chosen by the M3 ablation.
- Back-projection consistency, iterations set by ablation.
- Full-resolution float32 processing; no preview-scale shortcuts in scored
  outputs.
- Measured A1 II camera model; 14-bit ADC.
- Full ladder, both benchmark resolutions, both renders, all sweeps.
- Shot and read noise at measured A1 II values in the delivered DNG (the one
  stage kept although it lowers A, section 5.1).

Stages dropped in section 5.1 are not "options" in the max preset because they
lower accuracy. Q9 default: they are removed from v6, with no opt-in look mode.

## 10. Outputs and automatic download

### 10.1 Run folder

`/content/exp50mp_v6/<run_id>/` with `run_id = <UTC timestamp>_<repo short
sha>`. Contents per variant: `delivered.dng`, `clean.dng`, `developed_16bit.tif`
(LibRaw), `preview.jpg`, masks, `metrics.json`, error maps, and run-level
`verdict.json`, `report.html`, `manifest.json` (every file with size and
SHA-256), `camera_model.json`, `config.json`, `environment.json`.

### 10.2 Download archives

Archives use ZIP_STORED for DNG and TIFF (already dense; compression only costs
time). The notebook calls `google.colab.files.download` once per archive, in
this order, and prints each archive's size and SHA-256 next to the call:

| Order | Archive | Contents | Approx size |
| --- | --- | --- | --- |
| 1 | `<run_id>_01_summary.zip` | Verdict, report, all JSON, manifest, downscaled error maps, previews | under 200 MB |
| 2 | `<run_id>_02_benchmark_4k.zip` | Full-size delivered and clean DNG, 16-bit TIFF, full-size error maps | about 0.6 GB |
| 3 | `<run_id>_03_benchmark_2k.zip` | Same for 2K | about 0.6 GB |
| 4 | `<run_id>_04_production_4k.zip` | Full-size DNG, clean DNG, TIFF, preview, masks (only if sources set) | about 0.6 GB |
| 5 | `<run_id>_05_production_2k.zip` | Same for 2K (only if sources set) | about 0.6 GB |

Excluded by default: sweep renders, NumPy arrays, ladder mosaics and the
reference ARW. `DOWNLOAD_ALL = True` adds a sixth archive with everything
(tens of GB; slow through a browser).

The summary archive goes first because it is small and holds the verdict; if
anything after it fails, the captain still has the answer.

### 10.3 When the browser blocks or misses a download

- Chrome asks "This site is trying to download multiple files" on the second
  download. If it is blocked, only archive 1 arrives. The notebook prints this
  warning before the downloads start and again in the banner.
- `files.download` only works while the Colab tab is open and connected when
  the run reaches the driver cell's `finally` block. A closed tab means no
  download, but the run folder and archives stay on the runtime.
- Recovery, printed in the banner:
  1. Allow multiple downloads for `colab.research.google.com`, then run
     `download_results()` in a new cell; it re-sends the existing archives
     without recomputing.
  2. Download the archives from the Files pane.
  3. If `DRIVE_MIRROR` is true, archives are also copied to
     `MyDrive/exp50mp_v6/<run_id>/` before the browser downloads start.
- Files on the runtime last only until Colab recycles it (idle timeout or
  session limit). The banner says so.

## 11. Milestones

- **M0, artefacts.** Pin HAT-L SHA-256 on Colab; publish the checkpoint mirror;
  host the reference ARW at a stable URL with a pinned SHA-256 (needs Q4).
- **M1, harness first.** Benchmark builder, metric implementation, sanity and
  sensitivity checks, ladder, unit tests on small synthetic fixtures. Score
  v5-as-shipped on Colab to get the baseline number.
- **M2, accuracy pipeline.** Measured matrix, illuminant estimate, exact
  inverse, back-projection, measured noise and levels, stage removals.
- **M3, ablation and freeze.** Per-stage ablation on Colab; freeze the max
  preset with the section 5.1 rule; record the ablation table.
- **M4, single-run notebook.** Cell contract, preflight, single driver cell
  with try/finally packaging, downloads, recovery function.
- **M5, evidence.** A full Run-all on the stated tier, measured stage times and
  memory, verdict and summary archive committed under `evidence/colab_runs/`
  (no DNGs or ARWs in Git).

## 12. Acceptance criteria for this PRD's build

1. From a fresh A100 High-RAM runtime, Runtime > Run all completes with no
   input between cells and downloads at least the summary archive.
2. The run reports PASS, MISSED or INVALID against section 6 exactly as
   defined, with all sanity checks shown.
3. The headline number is computed in linear RAW space on the delivered DNG.
4. Every stage in the max preset except shot and read noise has an ablation
   row showing it meets the section 5.1 rule; the noise cost is reported as
   ladder rung L1.
5. A forced exception in any driver step (for example, a failed clone or a
   wrong checkpoint hash) still produces and downloads a summary archive that
   explains the failure, with no cell left unrun by Run all. A forced
   exception in step 5 or 6 keeps the benchmark's PASS or MISSED verdict,
   marked "completed with errors". The guarantee covers raised exceptions
   only; process loss relies on the mitigations in section 9.2.
6. Unit tests and lint pass locally; full-suite tests run through the shared
   heavy-lock script.

## 13. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Illuminant not identifiable from a white-balanced render | R and B planes likely miss | Ladder isolates it; Q2 |
| One reference frame: tuning and gating on the same frame overfits | 95% may not hold on other scenes | Report the score as in-sample until held-out frames exist; Q4 asks for at least 5 frames with a held-out split |
| Browser blocks multiple downloads or the tab is closed | Captain gets only the summary or nothing | Summary first, recovery function, Files pane, optional Drive mirror |
| HAT-L or reference download fails (quota, link rot) | Run cannot start | Repo-hosted mirror, SHA-256 pins, preflight fails fast with a report |
| A100 not available or High-RAM not granted | Slower or rejected run | L4 accepted with warning; preflight states the actual tier |
| Runtime longer than estimates, Colab session recycled | Lost outputs | Measure in M5; summary archive written early and updated; Drive mirror |
| Disk exhaustion while archiving | Packaging fails | 80 GB preflight check, ZIP_STORED, exclusions by default |
| ARW licence unknown | Cannot host publicly | Q4 default: captain-owned private URL with pinned SHA-256, no redistribution |
| rawpy or LibRaw version drift changes R1 or decode | Score shifts between runs | Pin versions; record LibRaw version and matrix in the manifest |
| Pip install needing a runtime restart | Breaks Run all | Pin versions known to install cleanly on Colab; preflight check |

## 14. Open questions for the captain

1. **Exposure.** Is one global exposure scalar allowed in the gate (section
   6.2), or must the score be fully unfitted? Default in this PRD: allowed,
   unfitted score always reported.
2. **Illuminant.** May the gated run use a user-declared illuminant (for
   example, "daylight" or a colour temperature), or must it be estimated from
   the image? Default: estimated.
3. **2K gate.** Should 2K also have to reach 95%, or is 2K reported only?
   Default: reported only, because 2K is unlikely to pass (section 7.1).
4. **Reference frames.** Can the reference ARW be hosted at a stable URL, and
   under what licence? Can we get at least 5 more A1 II frames of varied scenes
   and light so that tuning and the gate use different frames? Default: a
   captain-owned private URL with a pinned SHA-256 and no redistribution; until
   more frames exist, the gate uses the single frame and the score is labelled
   in-sample (section 13).
5. **Production inputs.** Are there specific 2K/4K source images to deliver,
   or is benchmark-only the default run? Default: benchmark-only.
6. **Delivered noise.** Should the primary delivered DNG include sensor noise
   (as now), or should the noise-free mosaic be primary and the noisy one
   secondary? Default: the noisy DNG is primary and gated, exempt from the
   section 5.1 rule, with its cost reported as rung L1; the clean mosaic is
   reported only.
7. **Missing model.** If HAT-L cannot be loaded, should the max run fail
   (default) or fall back to Lanczos and say so?
8. **Downloads.** Is the default bundle set right, or should every artefact
   download (tens of GB)? Should the Drive mirror be on by default? Default:
   the five bundles in section 10.2, Drive mirror off.
9. **Look mode.** Keep the dropped v5 stages (optics, grain, fabricated sensor
   effects) as an opt-in, ungated "camera look" mode, or remove them from v6?
   Default: remove them from v6, no look mode.
10. **If MISSED.** Ship the best measured pipeline with a MISSED verdict, or
    fund an A1 II-specific RGB-to-RAW model (section 7.4, step 3)? Default:
    ship the best measured pipeline with a MISSED verdict and open a separate
    PRD for an A1 II RGB-to-RAW model.
