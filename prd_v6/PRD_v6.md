# PRD v6: Single-Run, Accuracy-Gated Synthetic 50 MP RAW (Sony A1 II)

Status: for captain review, 2026-11-02. The captain accepted the open-question
defaults ("your answers look good") and then added a requirement for three
16-bit outputs, 2K, 4K and 50 MP, each held to the 95% target (section 5.4).
The notebook this PRD specifies is built in the same change
(`prd_v6/notebooks/synthetic_50mp_v6.ipynb`, package `prd_v6/`); its first
full Colab run is milestone M5. Supersedes PRD v5 as the next experiment.
`prd_v5/PRD_v5.md` is unchanged and remains the record of the v5 scope; this
document says what v6 keeps, changes and drops.

## 1. Why v6

The captain set two targets for the next revision:

1. **One run.** The notebook is a single Colab "Run all" with no manual steps
   between cells. It runs at maximum quality and automatically downloads the
   results when it finishes.
2. **95% accurate RAW.** The synthetic process should produce a RAW image that
   is 95% accurate against a real camera RAW.
3. **Three 16-bit outputs.** The run delivers a 2K, a 4K and a 50 MP output,
   each with 16-bit data that Photoshop opens, and each aims at the same 95%
   benchmark (added by the captain after the first draft).

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
- G1a. Three delivered outputs per run, 2K (1920x1280), 4K (3840x2560) and
  50 MP (8640x5760), each a 16-bit DNG that Photoshop opens through Adobe
  Camera Raw plus a developed 16-bit TIFF that Photoshop opens directly
  (section 5.4).
- G2. A precise, honest definition of "95% accurate RAW" against a real A1 II
  ARW, measured in linear RAW space, used as the **primary pass/fail gate**
  (section 6) and applied to each of the three outputs against the real RAW at
  that output's resolution.
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
the 95% target, with the evidence behind it and one score per output. The
archives hold the 2K, 4K and 50 MP outputs, which the captain opens in
Photoshop to judge them by eye next to the numbers. Optionally the captain
supplies their own 2K and 4K source images by URL in the config cell, and the
same frozen pipeline produces the three outputs for each of them.

## 4. Terms

- **Reference ARW.** The real Sony ILCE-1M2 capture measured in PR #13:
  raw 8672x5784, visible 8640x5760, RGGB, 14-bit, black 512, LibRaw maximum
  16383, practical clip 15360, ISO 100, 1/250 s, f/8, 28 mm. Not committed to
  Git (106 MB, licence unknown, see Q4).
- **Benchmark render.** A rendered RGB image made from the reference ARW by a
  fixed, declared recipe (section 6.1). It plays the role of "the JPEG/PNG the
  user has". The pipeline sees only the render and camera-model constants.
- **Delivered outputs.** The three gated outputs of section 5.4: a 50 MP RGGB
  mosaic DNG (8640x5760) and 4K (3840x2560) and 2K (1920x1280) linear
  camera-RGB DNGs, all with 16-bit samples, each with a developed 16-bit TIFF
  beside it. "Delivered DNG" means any of the three DNGs.
- **16-bit scale.** The A1 II records 14 bits: black 512, clip 15360, so 14,848
  DN of signal. Every v6 output stores unsigned 16-bit samples on that scale
  multiplied by exactly 4: black 2048, white 61440, 59,392 DN of signal. The
  reference is put on the same scale by the same multiplication, which is
  lossless, so the gate compares like with like.
- **Linear RAW space.** Black-subtracted camera values in DN, before white
  balance, colour matrix or tone curve. At 50 MP these are photosite values,
  before demosaicing. At 2K and 4K they are camera-RGB values on the output
  grid (section 6.2).
- **Camera-model constants.** Facts true of every A1 II frame: CFA pattern and
  phase, black and white levels, clip point, colour matrix, conversion gain,
  read noise, full well. The pipeline may use these.
- **Frame-specific facts.** Facts true only of the reference frame: its as-shot
  white balance, exposure, pixel values. The pipeline must not use these, except
  in clearly labelled oracle diagnostics (section 7.2).

## 5. Pipeline

### 5.1 Stage decisions

Each v5 stage is classified by its expected effect on accuracy against the
reference. "Expected" means reasoned from the measurements we have; every run
measures the ablation (section 6.5), and M5 is the first such measurement.

| v5 stage | Expected effect on accuracy | v6 decision |
| --- | --- | --- |
| Input decode, ICC handling, geometry planning to 8640x5760 | Required | Keep |
| Exact inverse sRGB transfer (piecewise W3C) to linear light | Helps; any approximation is pure error | Keep, exact form only |
| Camera matrix: identity (virtual sRGB camera) | Hurts; R and B planes systematically wrong | **Replace** with the measured A1 II matrix (`src/synthraw/camera_profile.py`) |
| White balance [1,1,1] | Hurts; wrong channel ratios in RAW | **Replace** with an illuminant estimate (section 5.3) |
| Fidelity SR, HAT-L (x2 or x4), PSNR-trained | Small help; +0.31 dB best case in cycle 1 | Keep for the 50 MP output: HAT-L x4 with x8 self-ensemble, both measured by the ablation |
| GAN or perceptual SR (Real-ESRGAN class) | Hurts; invents texture that is not in the scene | **Drop** |
| Image-domain grain | Hurts; adds error that is not sensor noise | **Drop** |
| Back-projection consistency (output, downsampled by the render model, must equal the input) | Helps; removes SR colour and level drift | **Add** |
| Synthetic optics: distortion, CA, PSF, defocus, motion, diffraction, vignetting | Hurts; the real lens already put these into the source, so adding them again double-applies them; distortion alone moves corners about 52 px | **Drop** |
| Analytic HDR expansion of unclipped tones (v5 `HDR_MAX_RADIANCE` 16) | Hurts outside clipped regions; changes tones the render already fixed | **Drop** outside clipped regions; keep clipped-region extension only if ablation shows no loss |
| HDRUNet | Unverified on this camera; changes unclipped tones | **Drop** from max preset |
| Learned RGB-to-RAW for other cameras (RAW-Flow Nikon/Canon/PASCALRAW, RAW-Diffusion, SpiralDiff) | Hurts; wrong spectral response, no A1 II checkpoint exists | **Drop** |
| Mosaic to RGGB at the reference CFA phase | Required | Keep |
| Saturation at the A1 II clip (15360) | Helps; matches real highlight behaviour | Keep, at the measured clip, not 16383 |
| Shot and read noise | Costs accuracy by design: the 50 MP DNG's independent noise adds directly to the P1 error (about 2-3% NRMSE near the median, section 7.1); needed for a believable RAW | Keep in the 50 MP output, at **measured** A1 II values: gain about 2.08 e-/DN (G), read noise 4 e- (upper bound), full well about 33,000 e-; exempt from the rule below. Not added to the 2K and 4K outputs (section 5.4) |
| PRNU, DSNU, FPN, dark current, hot and dead pixels, nonlinearity | Hurts; fabricated patterns are not in the reference (its measured hot-pixel rate is 6e-6 and dead 2e-8) | **Drop** |
| ADC quantization to 14 bit, black 512 | Required | **Replace** with quantization on the 16-bit scale (A1 II levels x4: black 2048, clip 61440, section 4), so the outputs carry 16-bit data; finer steps cannot lower accuracy |
| DNG writer, LibRaw TIFF, `dng_validate` | Required | Keep |
| Edit sweeps (exposure, WB, demosaic, tone) | No effect on the gate; useful evidence of editability | Keep, outputs excluded from default downloads |
| Colour references, LPIPS, MTF, render-space metrics | Reported only; not RAW accuracy | Keep as secondary report |

Rule for the frozen max preset: a stage stays enabled only if the ablation on
the benchmark shows it changes the primary accuracy score A (section 6.3) by
-0.1 percentage points or better on every plane. A stage that helps one plane
and hurts another by more than that is dropped or made plane-neutral.

Shot and read noise are exempt from this rule. They are a property of the
deliverable, not an accuracy stage: the 50 MP DNG is a noisy camera RAW
(Q6 default) and the gate scores that file, so noise stays on even though it
lowers A. Its cost is reported every run as ladder rung L1 (section 7.2).

The preset is frozen in code from the reasoning in this table before any Colab
measurement exists. Every run measures the ablation (section 6.5) and flags in
the verdict any stage that breaks the rule; it never changes the preset at run
time. Changing the frozen preset after a measurement needs a new commit
(section 11, M3).

### 5.2 v6 order

```
source RGB (8-bit sRGB render)
  -> decode, exact inverse sRGB -> linear RGB
  -> resample to each output grid
       2K / 4K output at or below the source size: area-downsample (none when
                  equal, which is the gated case)
       larger output: fidelity SR (HAT-L x4, fp32, tiled, x8 self-ensemble),
                  then area-downsample (or Lanczos for a final step up) to the
                  output grid, then back-projection consistency against the
                  source
  -> undo illuminant (white-balance gains from the illuminant prior)
  -> linear sRGB(D65) -> A1 II camera RGB (measured matrix, the inverse of the
     LibRaw rendering transform)
  -> exposure placement (camera-model prior; see 6.2 for how the gate treats it)
  -> 2K / 4K: three-channel camera RGB on the output grid
       -> 16-bit scale, clip 61440             [2K and 4K DNG]
  -> 50 MP: clean mosaic, RGGB at reference phase   [clean DNG written here]
       -> electrons, shot + read noise at measured values, clip at full well
       -> 16-bit scale, black 2048, clip 61440      [50 MP DNG]
  -> DNG, developed 16-bit TIFF, LibRaw parse, dng_validate, reports
```

### 5.3 Illuminant

A white-balanced render does not contain the scene illuminant. The pipeline has
to assume one, and the reference shows how much this matters: the A1 II as-shot
neutral is [0.4883, 1, 0.5636] and its daylight neutral is
[0.3744, 1, 0.7402]. Assuming daylight on this frame puts the R plane about
23% low and the B plane about 31% high before any other error.

v6 estimates the illuminant from the render using a camera-model prior
and records the estimate and its uncertainty. It must not read the reference
frame's as-shot neutral. Whether the captain allows a user-declared illuminant
in the gated run is open question Q2 (answered: estimated).

A white-balanced render carries no measurable cue to the illuminant: the
renderer divided it out, and highlight clipping removes the only physical
trace (which channel saturated first). The v6 estimator is therefore a fixed
camera-model prior, not a per-image measurement: the Planckian-locus point at
the mired midpoint of the common-illuminant range 2850 K (CIE A, tungsten) to
7500 K (shade, D75), about 4130 K, mapped to a camera neutral through the
measured A1 II matrix. It is about [0.503, 1, 0.550], and it bounds the
worst-case per-channel error over that range rather than betting on daylight.
On the reference frame (as-shot [0.4883, 1, 0.5636]) it is about 3% high on R
and 2.4% low on B. That closeness is the luck of one frame, and the rule was
chosen with that frame's as-shot neutral already known, so the score is
in-sample (section 13). Ladder rung L2 measures the prior's cost on every run.

### 5.4 Three outputs

Every run delivers three outputs, and the gate scores each of them
(section 6.3):

| Output | Size | DNG layout | Made from (gated run) |
| --- | --- | --- | --- |
| 2K | 1920x1280 | LinearRaw: three samples per pixel, camera RGB, already demosaiced | the 2K render, inverted at its own size |
| 4K | 3840x2560 | LinearRaw, as 2K | the 4K render, inverted at its own size |
| 50 MP | 8640x5760 | CFA: one RGGB sample per photosite, a true mosaic like the ARW | the 4K render through HAT-L x4 |

All three DNGs store unsigned 16-bit samples on the 16-bit scale (section 4):
the A1 II's 14-bit levels multiplied by exactly 4, so black 2048 and white
61440, with BlackLevel and WhiteLevel tags to match. The camera colour matrix
and the estimated neutral (as AsShotNeutral) go into the tags. 16-bit samples
give the outputs a finer step than the camera's 14 bits; they cannot add
information the render lacks, and the gate measures what is really there.

Each DNG also gets a developed 16-bit TIFF: sRGB with an ICC profile, white
balanced with the estimated neutral, AHD demosaic for 50 MP, and no tone curve
beyond the sRGB transfer function. Photoshop opens the DNG through Adobe Camera Raw, which is the
editable RAW path, and opens the TIFF directly as a 16 bits/channel document.

Why 2K and 4K are LinearRaw rather than mosaics: a 2K or 4K Bayer mosaic would
throw away two thirds of each pixel's colour, and the only real A1 II data at
those sizes is the full-colour, binned image. LinearRaw keeps all three
samples per pixel and is still a RAW to Adobe Camera Raw: linear, unrendered,
with the camera matrix and white balance applied in the editor.

The 2K and 4K outputs carry no added noise. The render is a binned image of the
reference, so it already carries the reference's own binned noise, and the
inverse reproduces it; adding independent noise would count it twice. The
50 MP output adds independent noise because SR cannot recover per-photosite
noise (section 5.1, Q6).

In production (section 8.2) each source makes all three outputs: an output at
or below the source size is area-downsampled from the inverted source, and a
larger one goes through HAT-L x4 as above. A 2K source therefore reaches 4K
and 50 MP through SR, with no accuracy claim.

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

References at each output size, built from the same ARW:

- 50 MP: the ARW visible area 8640x5760, per photosite, no processing.
- 4K and 2K: the ARW decoded with the same rawpy and AHD demosaic as R1 but in
  camera space with unit white balance (raw colour output, linear gamma, no
  auto-bright, highlight clip, 16-bit), put back on the DN scale, then
  area-downsampled by exactly 2.25 and 4.5. This is the camera's own colour
  data at those sizes, before white balance and colour matrix. It shares R1's
  demosaic, which the pipeline does not invert; after a 2.25x or 4.5x bin the
  demosaic's own error is far below the 5% budget, and section 13 lists the
  shared step as a limitation.

All references are multiplied by 4 to the 16-bit scale before scoring.

The camera matrix LibRaw used for R1 is recorded in the benchmark manifest. If
it differs from the measured matrix in `camera_profile.py`, the run reports both
and the gap is attributed in the error budget; the pipeline keeps using the
measured matrix because it is a camera-model constant.

### 6.2 Comparison space

Each output is scored against the reference at its own size (section 6.1),
on the 16-bit scale:

- Reference x: at 50 MP, the ARW visible area 8640x5760, black-subtracted, per
  photosite. At 4K and 2K, the binned camera-RGB reference, per pixel and
  channel.
- Candidate y: the output DNG's stored samples, black-subtracted using the
  DNG's own black level. The 50 MP DNG is decoded with the same rawpy; the 2K
  and 4K LinearRaw samples are read directly from the file, and LibRaw must
  also parse the file (integrity gate).
- The 50 MP DNG visible area must be 8640x5760 at the reference CFA phase, and
  the 4K and 2K DNGs must be exactly 3840x2560 and 1920x1280. A size or phase
  mismatch makes the run INVALID, not a low score.
- Exposure: a single global scalar k = sum(x*y) / sum(y*y), fitted over the
  mask and shared by all three planes, once per output. Absolute exposure (scene radiance times
  shutter, aperture and ISO) cannot be recovered from a render whose exposure
  was normalized, so one scalar is allowed and reported in EV. Because it is a
  single scalar, white balance, colour matrix and plane-ratio errors still
  count in full. Whether the captain allows even this scalar is Q1; the
  unfitted (k = 1) score is always reported.
- Mask: the visible area minus a 16 px border (8608x5728 photosites), minus
  sensor-clipped reference photosites, defined as raw value (before black
  subtraction) >= 0.98 x 15360 = 15052.8 DN; no truth is available there. At
  4K and 2K the border is the same 16 photosites, rounded up to 8 and 4 output
  pixels, and an output pixel is masked when any photosite in its footprint is
  sensor-clipped. The
  source-clipped pixels of the 8-bit render are kept in the mask and reported
  separately, because recovering them is part of the job. If sensor-clipped
  photosites exceed 1% of the visible area excluding the border, the reference
  frame is unsuitable and the run is INVALID. The border is not counted
  towards that 1%.

### 6.3 Primary gate

Both conditions must hold for each of the three outputs (2K, 4K and 50 MP,
section 5.4, R1 render) against the reference at that output's size, for each
colour plane p in {R, G, B} (at 50 MP the two green sites are pooled into G):

- **P1, energy accuracy.**
  A_p = 1 - sqrt( sum((k*y - x)^2) / sum(x^2) ) >= 0.95.
  Equivalently, the normalized RMS error in linear RAW is at most 5% of the
  reference signal energy in that plane.
- **P2, photosite agreement.** At least 95% of masked photosites (pixels, at 2K
  and 4K) in the plane satisfy |k*y - x| <= max(0.05 * x, 3 * sigma), where
  sigma = sqrt(sigma_ref(x)^2 + k^2 * sigma_syn(y)^2), sigma_ref uses the
  measured A1 II gain and read noise (at 2K and 4K divided by the number of
  same-colour photosites binned into one pixel), and sigma_syn uses the
  pipeline's own noise parameters (zero for the 2K, 4K and clean outputs).

The accuracy of one output is the min over planes of A_p, shown as a
percentage with the P2 fractions next to it. The notebook prints all three, and
the run headline is the lowest of the three. PASS needs P1 and P2 on all three
planes of all three outputs; the verdict also states each output's own result,
so a run where 2K and 4K reach 95% and 50 MP does not is MISSED and says
exactly that.

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
| Scoring only on heavily downsampled images | PR #13 showed downsampled comparisons hid high-frequency differences. v6 scores the 50 MP output at every photosite; the 2K and 4K scores are about those outputs and never stand in for the 50 MP score. |
| Clean mosaic as the gated output | The 50 MP DNG includes noise; the user gets that file. The clean score is reported as a diagnostic. |

Sanity checks, run on every benchmark for each output at its own size and
shipped in the report.

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
- Shift by 1 photosite of the same colour (2 raw photosites); 1 pixel at 4K
  and 2K.
- R plane gain +10%.
- Gaussian blur of each CFA plane with sigma 2 raw photosites (1 same-colour
  site); sigma 1 pixel at 4K and 2K.

### 6.5 Reported, not gated

- A 50 MP output made from the 2K render (HAT-L x4 then Lanczos x1.125),
  scored like the gated 50 MP output.
- R2 stress render results for all three outputs.
- Clean mosaic (noise-free) score.
- Ablation table on the 50 MP clean mosaic: x8 self-ensemble against its
  identity member, HAT-L against Lanczos, back-projection on and off, and the
  illuminant prior against daylight. A row that breaks the section 5.1 rule is
  flagged in the verdict.
- Unfitted score (k = 1) and k in EV.
- Diagnostic per-plane white-balance-fitted score, labelled as diagnostic.
- L1 variant of P1.
- Band split: error energy below and above the source Nyquist frequency, so
  detail loss is separated from colour and level error.
- Tone zones: shadows, midtones and highlights by reference value, and the
  source-clipped region on its own.
- Error maps per plane (PNG, full resolution in the data bundle, downscaled in
  the summary), and error histograms.
- Secondary noise gate (pass/fail, not part of the 95% target): the 50 MP
  DNG's G-plane conversion gain, measured by photon transfer
  (`src/synthraw/realism.py`), is within 15% of the reference's. Read noise is
  reported only, because single-frame estimates are scene-contaminated upper
  bounds.

### 6.6 Run status

- **PASS**: integrity gates pass, hard sanity checks pass, primary gate
  passes.
- **MISSED**: integrity and hard sanity checks pass, primary gate fails on at
  least one output. The report states the gap per output and plane and the
  largest contributor from the error budget (section 7.2).
- **INVALID**: integrity or a hard sanity check fails, the reference is
  unsuitable, one of driver steps 1-4 raised (section 9.2), the process was
  lost before step 4 finished (recorded by `download_results` as "process
  lost during step N", section 9.2), or a required stage silently did
  nothing. No accuracy claim is made.

A failure in driver step 5 (production) or step 6 (sweeps and secondary
reports) does not change the PASS or MISSED verdict already computed by the
benchmark; the verdict and the banner mark it "completed with errors" and
name the failed step. A process loss in step 5 or 6 likewise keeps PASS or
MISSED, marked "completed with errors: process lost during step N".

Thresholds are fixed in code and in this PRD. The notebook never relaxes a
threshold, never retries with different settings to reach a pass, and never
reports a diagnostic score as the headline. In every status, results are
packaged and downloaded, unless the runtime process itself is lost
(section 9.2).

### 6.7 Integrity gates kept from v5

Each of the three DNGs parses with LibRaw and passes `dng_validate`, and
each TIFF reads back as 16-bit; CFA pattern and phase match at 50 MP;
metadata is coherent (levels, matrices, crop); no tile seams; provenance and
hashes are complete; every enabled stage changes its input (no silent no-op).

## 7. Feasibility

### 7.1 Is 95% reachable?

Honest answer, per output:

- **2K and 4K: plausible.** No resolution is lost by construction, because the
  render is the reference binned to the output's size. What remains is the
  illuminant prior, 8-bit quantization (largest in deep shadows), out-of-gamut
  and highlight clipping in the sRGB render, and any gap between the measured
  matrix and the one LibRaw used. On this frame the prior is within about 3%
  per channel, which alone uses most of the R budget. Expected: G passes, R and
  B near the line.
- **50 MP: unlikely.** On top of the 2K and 4K errors it must recover detail
  the 4K render does not have (x2.25 per axis), and it carries independent
  shot and read noise by design. Detail loss is estimated at 3-6% NRMSE and
  noise at 2-3%; in quadrature that is about 4-7% against a 5% budget, before
  the illuminant. P2 is hardest in fine texture. The target stays at 95%, it is
  not lowered; M5 measures the gap and the ladder names its causes.
- v5 as shipped is expected to fail by construction at every size; this has
  not been measured yet.

Error sources, with current estimates to be replaced by measurement in M5:

| Source | Estimated cost in NRMSE | Notes |
| --- | --- | --- |
| Detail lost to downsampling, 4K render to 50 MP (x2.25) | about 3-6% | 50 MP only; SR recovers a fraction, cycle 1 suggests a small one |
| Detail lost, 2K render to 50 MP (x4.5) | about 6-10% | Reported-only 50 MP variant; alone likely exceeds the budget |
| Illuminant error | up to 23% (R) and 31% (B) with a daylight assumption; about 3% (R) and 2.4% (B) with the v6 prior on this frame | Every output; the small figure is in-sample luck (section 5.3) |
| 8-bit quantization and sRGB clipping | small in midtones, large in deep shadows, clipped highlights and saturated colours | Every output; shadows are where P2 bites |
| Demosaic and colour-matrix mismatch between R1 and the inverse | unknown until M5 | Recorded in the benchmark manifest |
| Noise (two independent realizations) | about 2-3% near the median, less in highlights | 50 MP only; bounded by the noise-floor check |

So the target is plausible at 2K and 4K and a real stretch at 50 MP. Each
output is held to the same 95%.

### 7.2 Error budget ladder

Every benchmark run also scores a ladder of oracle variants. Oracles use
frame-specific facts and are labelled as diagnostics; they never feed the
delivered outputs or the headline.

| Rung | What changes | Cost it isolates |
| --- | --- | --- |
| L0 | Delivered output | Headline |
| L1 | Clean mosaic (no noise) | Noise |
| L2 | L1 with the oracle illuminant (reference as-shot neutral) | Illuminant estimation |
| L3 | Full-resolution R1 render (no downsampling) through the same inverse | Resolution and SR |
| L4 | L3 with the oracle illuminant | Remaining inverse ISP error (matrix, demosaic, quantization) |

The full ladder runs for the 50 MP output. The 2K and 4K outputs have no
added noise and no SR, so their ladder is L0 and L2 (L1 equals L0 there).

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
2. If L3 vs L1 dominates at 50 MP: accept 2K and 4K as the claim with 50 MP
   reported only, or evaluate stronger fidelity SR.
3. If L4 is itself below 95%: the analytic inverse is the limit; train or
   fine-tune an A1 II-specific RGB-to-RAW model on paired A1 II frames. This
   needs many more reference frames and is a separate PRD.

## 8. Inputs

### 8.1 Benchmark inputs

- Reference ARW: the publisher's own sample file from the Photography Blog
  Sony A1 II review,
  `https://img.photographyblog.com/reviews/sony_a1_ii/sample_images/sony_a1_ii_65.arw`,
  106,033,152 bytes, SHA-256
  `dc87580c1d6111cbaef530478fa1f7863a02a5482cd7cab9a902d731c20b4893`. The
  notebook fetches it from that URL at run time and checks size and hash;
  nothing is redistributed (Q4). If the URL stops working, `REFERENCE_ARW_URL`
  in the settings cell can point at another copy, which must match the same
  hash.
- HAT-L checkpoint `HAT-L_SRx4_ImageNet-pretrain.pth` (XPixelGroup HAT,
  Apache-2.0), 165,774,123 bytes, SHA-256
  `5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043`, from
  the Hugging Face mirror `anchuang/HAT-L_SRx4_ImageNet-pretrain` pinned at
  revision `cf5cb1c188096b795e608629afd9a864646dbffa`. The official Google
  Drive folder is the source of truth but is not used at run time because
  Drive quotas fail intermittently. The mirror is a third party, so the pinned
  hash is what makes it usable: the hash was taken from the mirror, the file
  is loaded with spandrel (pinned) through `torch.load(weights_only=True)` so
  it cannot run code, and the run checks that spandrel reports a HAT x4 model
  of HAT-L size. The first draft named `HAT_SRx4_ImageNet-pretrain.pth`, which
  is the smaller HAT model, not HAT-L. Only the x4 model is used: every SR path
  in v6 is x4 followed by area or Lanczos resampling.

### 8.2 Production inputs (optional)

`SOURCE_2K_URL` and `SOURCE_4K_URL` in the config cell. When set, the frozen
pipeline also runs on them and produces the three outputs for each source
(section 5.4). Each source is centre-cropped to 3:2 first. There is no reference
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
  v6 adds the ladder, three outputs, the reference and archive staging).
- Expected wall time on A100: about 2.5 to 4.5 hours. The largest items are
  HAT-L x4 with x8 self-ensemble on the 4K R1 and R2 renders (about 70 tiles of
  512 px, so about 560 tile passes each, estimated 40-50 minutes per render)
  and on the 2K render (about 10 minutes). The 2K and 4K outputs need no SR and
  take minutes. Each production source adds about 50 minutes (4K source) or
  10 minutes (2K source). These are estimates; M5 replaces them with measured
  stage times. SR outputs are computed once and shared by every rung and
  ablation row that uses the same input.

### 9.2 Cell contract

The notebook has two code cells. Colab's Run all stops at the first cell that
raises, and a `try/finally` cannot span cells, so all work that can fail runs
inside one driver cell.

1. **Settings cell.** Literal assignments only, so it cannot fail, with
   defaults that need no edits: `PRESET = "max"`, `REPO_REF` pinned to a
   release commit (not `main`), `SOURCE_*_URL = None`,
   `REFERENCE_ARW_URL = None` (the publisher URL of section 8.1),
   `DOWNLOAD_ALL = False`, `DRIVE_MIRROR = False`.
2. **Driver cell.** At the start of the run, before any heavy step, it prints
   the recovery line: `run_dir`, the clone path, and the import-and-call line
   for `download_results(run_dir)`. Then one `try/finally`. The `try` runs
   these steps in order, and before each step starts it writes a step-started
   record (the current step number) to the run folder:
   1. **Drive mount**, only if `DRIVE_MIRROR` is true. It is the first step, so
      the one authorization prompt happens at the start and never later.
   2. **Setup.** Clone at `REPO_REF`, install pinned packages that do not
      require a runtime restart (a restart would break Run all; the notebook
      checks this and fails with a clear message if a package would need one),
      build `dng_validate`.
   3. **Preflight.** GPU, VRAM, RAM, disk, package versions; download the
      reference ARW and checkpoints and verify every SHA-256. Any failure stops
      here, before heavy work.
   4. **Benchmark.** Build R1 and R2 at 4K and 2K and the references at 2K,
      4K and 50 MP, run the frozen pipeline to make the three outputs, the
      ladder and the ablation, score, sanity checks. It writes `verdict.json` to the run
      folder as soon as step 4 finishes.
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
   (section 10) and ends with a printed verdict banner. It calls the same
   standard-library-only packaging code as `download_results`, imported from
   the clone on disk without any installed package; if the clone is missing or
   that import fails, a minimal fallback defined in the notebook itself zips
   the run folder with the standard library and downloads it through
   `google.colab.files`, so it works even when the clone or install failed.

   After each driver step, the driver cell rebuilds the summary archive and,
   when `DRIVE_MIRROR` is true, copies it to Drive at that point, not only in
   the `finally` block.

   This guarantee covers raised exceptions only. A `finally` block does not
   run when the runtime process itself is lost (an out-of-memory kernel crash,
   a runtime disconnect or a recycle). For those cases the mitigations are:
   - the summary archive rebuilt after each driver step, and its Drive copy
     when `DRIVE_MIRROR` is true;
   - `download_results(run_dir)`, which lives in the repo package, so after a
     kernel crash a new cell can import it from the clone on disk. It always
     rebuilds the summary archive and reads N from the step-started record.
     When `verdict.json` is present it keeps that PASS or MISSED, marked
     "completed with errors: process lost during step N" unless the `finally`
     block already wrote the final verdict; when it is absent (process lost
     before step 4 finished) it writes the verdict as INVALID with the reason
     "process lost during step N". It builds any missing archives from the
     run folder, labels any archive built from a step that did not finish as
     partial, then triggers the downloads (section 10.3).

   After a runtime recycle only what was already mirrored to Drive survives;
   nothing survives a recycle when `DRIVE_MIRROR` is false.

Never used anywhere: `files.upload`, `input()`, widgets that wait for a click,
`/content` paths that assume a manual upload, or silent fallbacks (for example,
Lanczos instead of HAT-L) in the max preset. If HAT-L cannot be loaded the run
is INVALID for the max preset (Q7).

### 9.3 Maximum quality settings

"Max options" in v6 means every stage that helps or holds accuracy is on, at
its highest-quality setting:

- HAT-L x4, fp32, tile 512 with 128 px overlap and feathered blending, x8
  self-ensemble (its identity member is kept for the ablation).
- Back-projection consistency, 3 iterations.
- Full-resolution float32 processing; no preview-scale shortcuts in scored
  outputs.
- Measured A1 II camera model; 16-bit output scale.
- All three outputs, full ladder, both renders, all sweeps.
- Shot and read noise at measured A1 II values in the 50 MP DNG (the one
  stage kept although it lowers A, section 5.1).

Stages dropped in section 5.1 are not "options" in the max preset because they
lower accuracy. Q9 default: they are removed from v6, with no opt-in look mode.

## 10. Outputs and automatic download

### 10.1 Run folder

`/content/exp50mp_v6/<run_id>/` with `run_id = <UTC timestamp>_<repo short
sha>`. Per output (`benchmark/2k/`, `benchmark/4k/`, `benchmark/50mp/`, and the
same under `production/<source>/`): the output DNG, its developed 16-bit TIFF,
`preview.jpg`, the mask, `metrics.json` and error maps; the 50 MP output adds
`clean.dng`. Run-level: `verdict.json`, `report.html`, `manifest.json` (every
file with size and SHA-256), `camera_model.json`, `config.json`,
`environment.json`.

### 10.2 Download archives

Archives use ZIP_STORED for DNG and TIFF (already dense; compression only costs
time). The notebook calls `google.colab.files.download` once per archive, in
this order, and prints each archive's size and SHA-256 next to the call:

| Order | Archive | Contents | Approx size |
| --- | --- | --- | --- |
| 1 | `<run_id>_01_summary.zip` | Verdict, report, all JSON, manifest, downscaled error maps, previews | under 200 MB |
| 2 | `<run_id>_02_benchmark_2k.zip` | 2K DNG (15 MB), 16-bit TIFF, full-size error maps | about 0.04 GB |
| 3 | `<run_id>_03_benchmark_4k.zip` | 4K DNG (59 MB), 16-bit TIFF, full-size error maps | about 0.13 GB |
| 4 | `<run_id>_04_benchmark_50mp.zip` | 50 MP DNG and clean DNG (100 MB each), 16-bit TIFF (300 MB), full-size error maps | about 0.55 GB |
| 5 | `<run_id>_05_production_4k_source.zip` | The three outputs made from `SOURCE_4K_URL` (only if set) | about 0.7 GB |
| 6 | `<run_id>_06_production_2k_source.zip` | The three outputs made from `SOURCE_2K_URL` (only if set) | about 0.7 GB |

The benchmark archives go in ascending size, so the smaller outputs arrive
first. Excluded by default: sweep renders, NumPy arrays, ladder and ablation
outputs, the reported-only variants and the reference ARW.
`DOWNLOAD_ALL = True` adds a seventh archive with everything (several GB; slow
through a browser).

The summary archive goes first because it is small and holds the verdict; if
anything after it fails, the captain still has the answer.

### 10.3 When the browser blocks or misses a download

- Chrome asks "This site is trying to download multiple files" on the second
  download. If it is blocked, only archive 1 arrives. The notebook prints this
  warning before the downloads start and again in the banner.
- `files.download` only works while the Colab tab is open and connected when
  the run reaches the driver cell's `finally` block. A closed tab means no
  download, but the run folder and archives stay on the runtime.
- Recovery, printed at the start of the run and again in the banner:
  1. Allow multiple downloads for `colab.research.google.com`, then in a new
     cell import `download_results` from the repo package in the clone on disk
     and run `download_results(run_dir)`. It always rebuilds the summary
     archive, builds any missing archives from the run folder without
     recomputing results, then triggers the downloads. When `verdict.json`
     from step 4 is present it keeps that PASS or MISSED, marked "completed
     with errors: process lost during step N" after a process loss in step 5
     or 6; when it is absent it writes the verdict as INVALID with the reason
     "process lost during step N". It labels any archive built from a step
     that did not finish as partial. This also works after a kernel crash,
     because the clone and the run folder stay on the runtime.
  2. Download the archives from the Files pane.
  3. If `DRIVE_MIRROR` is true, the summary archive is copied to
     `MyDrive/exp50mp_v6/<run_id>/` after each driver step, and all archives
     are copied there before the browser downloads start.
- Files on the runtime last only until Colab recycles it (idle timeout or
  session limit). After a recycle only what was already mirrored to Drive
  survives; nothing survives a recycle when `DRIVE_MIRROR` is false. The
  banner says so.

## 11. Milestones

M1 to M4 are built in the same change as this PRD and tested locally only
(unit tests and tiny-input smoke tests; no notebook, model or full-frame run on
the development Mac). Their first measurement is M5.

- **M0, artefacts.** Done: the reference ARW is pinned to the publisher URL and
  HAT-L to a pinned mirror revision, both with SHA-256 (section 8.1); no
  hosting needed.
- **M1, harness first.** Benchmark builder for the three output sizes, metric
  implementation, sanity and sensitivity checks, ladder, unit tests on small
  synthetic fixtures. Scoring v5-as-shipped is left to a later run; the v6
  ablation's daylight row stands in as the identity-free baseline.
- **M2, accuracy pipeline.** Measured matrix, illuminant prior, exact inverse,
  back-projection, measured noise and levels, stage removals, 16-bit DNG and
  TIFF writers for the three outputs.
- **M3, ablation and freeze.** The preset is frozen in code from section 5.1;
  every run measures the ablation and flags a rule break (section 6.5).
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
3. The headline numbers are computed in linear RAW space on the three
   delivered DNGs, each against the reference at its own size.
4. Every stage in the max preset except shot and read noise has an ablation
   row showing it meets the section 5.1 rule; the noise cost is reported as
   ladder rung L1.
5. A forced exception in any driver step (for example, a failed clone or a
   wrong checkpoint hash) still produces and downloads a summary archive that
   explains the failure, with no cell left unrun by Run all. A forced
   exception in step 5 or 6 keeps the benchmark's PASS or MISSED verdict,
   marked "completed with errors". The guarantee covers raised exceptions
   only. After a forced kernel crash, `download_results(run_dir)` imported
   from the clone in a new cell rebuilds the summary archive, with an INVALID
   verdict and the reason "process lost during step N" for a crash before
   step 4 finishes, or the step 4 PASS or MISSED marked "completed with
   errors: process lost during step N" for a crash in step 5 or 6, builds
   any missing archives from the run folder (labelled partial when their step
   did not finish) and downloads them. After a runtime recycle only what was
   already mirrored to Drive survives (the summary archive as of the last
   completed driver step when `DRIVE_MIRROR` is true; nothing when it is
   false).
6. Unit tests and lint pass locally; full-suite tests run through the shared
   heavy-lock script.
7. The 2K, 4K and 50 MP DNGs open in Photoshop through Adobe Camera Raw, and
   their TIFFs open directly as 16 bits/channel documents (checked by the
   captain on the first run).

## 13. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Illuminant not identifiable from a white-balanced render | R and B planes likely miss | Ladder isolates it; Q2 |
| One reference frame: tuning and gating on the same frame overfits | 95% may not hold on other scenes | Report the score as in-sample until held-out frames exist; Q4 asks for at least 5 frames with a held-out split |
| Browser blocks multiple downloads or the tab is closed | Captain gets only the summary or nothing | Summary first, recovery function, Files pane, optional Drive mirror |
| HAT-L or reference download fails (quota, link rot) | Run cannot start | SHA-256 pins, `REFERENCE_ARW_URL` override, preflight fails fast with a report |
| HAT-L mirror is a third party | Wrong or tampered model | Pinned revision and SHA-256, `weights_only` load, architecture and size check |
| 50 MP output misses 95% | Headline MISSED | Expected (section 7.1); 2K and 4K results stand on their own in the verdict; the ladder names the cause |
| 2K and 4K references share R1's demosaic | Their scores are slightly optimistic | Disclosed (section 6.1); the 50 MP reference is raw photosites with no shared step |
| Adobe Camera Raw renders a LinearRaw DNG unexpectedly | 2K or 4K DNG looks wrong in Photoshop | The 16-bit TIFF beside it opens directly; `dng_validate` checks every DNG |
| A100 not available or High-RAM not granted | Slower or rejected run | L4 accepted with warning; preflight states the actual tier |
| Runtime longer than estimates, Colab session recycled | Lost outputs | Measure in M5; summary archive rebuilt after each driver step and copied to Drive when `DRIVE_MIRROR` is true (only that survives a recycle); `download_results(run_dir)` after a kernel crash (rebuilds the summary: INVALID "process lost during step N" before step 4 finishes, otherwise the step 4 PASS or MISSED marked "completed with errors: process lost during step N"; partial archives labelled) |
| Disk exhaustion while archiving | Packaging fails | 80 GB preflight check, ZIP_STORED, exclusions by default |
| ARW licence unknown | Cannot host publicly | Fetched from the publisher's own URL with a pinned SHA-256; never redistributed or committed |
| rawpy or LibRaw version drift changes R1 or decode | Score shifts between runs | Pin versions; record LibRaw version and matrix in the manifest |
| Pip install needing a runtime restart | Breaks Run all | Pin versions known to install cleanly on Colab; preflight check |

## 14. Open questions for the captain

The captain accepted every default below ("your answers look good"), so the
defaults are now the decisions of this PRD. Q3 was then superseded by the
three-output requirement, and Q4 was settled without hosting.

1. **Exposure.** Is one global exposure scalar allowed in the gate (section
   6.2), or must the score be fully unfitted? Default in this PRD: allowed,
   unfitted score always reported.
2. **Illuminant.** May the gated run use a user-declared illuminant (for
   example, "daylight" or a colour temperature), or must it be estimated from
   the image? Default: estimated.
3. **2K gate.** Should 2K also have to reach 95%, or is 2K reported only?
   Default: reported only, because 2K is unlikely to pass (section 7.1).
   **Superseded:** the captain asked for 2K, 4K and 50 MP outputs, all aiming
   at 95%, so all three are gated (sections 5.4 and 6.3).
4. **Reference frames.** Can the reference ARW be hosted at a stable URL, and
   under what licence? Can we get at least 5 more A1 II frames of varied scenes
   and light so that tuning and the gate use different frames? Default: a
   captain-owned private URL with a pinned SHA-256 and no redistribution; until
   more frames exist, the gate uses the single frame and the score is labelled
   in-sample (section 13). **Settled:** the publisher's own URL serves the
   file (section 8.1), so v6 fetches it from there with a pinned SHA-256 and
   does not host or redistribute it. More frames are still wanted.
6. **Delivered noise.** Are there specific 2K/4K source images to deliver,
   or is benchmark-only the default run? Default: benchmark-only.
6. **Delivered noise.** Should the primary 50 MP DNG include sensor noise
   (as now), or should the noise-free mosaic be primary and the noisy one
   secondary? Default: the noisy DNG is primary and gated, exempt from the
   section 5.1 rule, with its cost reported as rung L1; the clean mosaic is
   reported only.
7. **Missing model.** If HAT-L cannot be loaded, should the max run fail
   (default) or fall back to Lanczos and say so?
8. **Downloads.** Is the default bundle set right, or should every artefact
   download (tens of GB)? Should the Drive mirror be on by default? Default:
   the bundles in section 10.2, Drive mirror off.
9. **Look mode.** Keep the dropped v5 stages (optics, grain, fabricated sensor
   effects) as an opt-in, ungated "camera look" mode, or remove them from v6?
   Default: remove them from v6, no look mode.
10. **If MISSED.** Ship the best measured pipeline with a MISSED verdict, or
    fund an A1 II-specific RGB-to-RAW model (section 7.4, step 3)? Default:
    ship the best measured pipeline with a MISSED verdict and open a separate
    PRD for an A1 II RGB-to-RAW model.
