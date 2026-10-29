# Reference ARW versus synthetic run: measured comparison

One real Sony Alpha 1 II raw (`ILCE-1M2`, 8672 x 5784 ARW, ISO 100, 1/250 s, f/8, FE 28-70 mm F2 GM at
28 mm, WB Auto) was measured plane by plane on the development laptop with LibRaw/rawpy and compared
against the synthetic run `run_20261029_094146` (PRD_v3 notebook, SONY_A1II_LIKE profile, 6336 x 2688
source, OUTPAINT_VERTICAL). The ARW is not committed: it is a 106 MB file of unknown licence. Only the
derived numbers below are recorded, as provenance class M (measured), with the source stated as "one
ILCE-1M2 sample frame".

What one frame can establish: the decoder-exposed identity (dimensions, CFA, levels, matrices, camera
white balance), where a real exposure sits in the ADC range, the shape of the dark-noise tail, the
banding amplitude ceiling, and a defect-pixel ceiling. What one frame cannot establish: a conversion
gain or read-noise value that is free of scene texture (a photon-transfer curve needs flat fields or
frame pairs), the temporal versus fixed-pattern split of the noise, PRNU, DSNU, dark current, and any
per-ISO behaviour. Every noise number below taken from the frame is therefore an upper bound and is
recorded as such.

## Identity and levels

| Quantity | ILCE-1M2 sample (M) | run_20261029_094146 | Adopted in SONY_A1II_LIKE |
|---|---|---|---|
| Raw dimensions | 8672 x 5784 (visible 8640 x 5760) | 8640 x 5760 | unchanged |
| CFA | RGGB | RGGB | unchanged |
| Bit depth | 14 (lossless 14-bit ARW) | 14 | unchanged |
| BlackLevel | 512 (all four channels) | 512 | unchanged, now with measured confirmation |
| WhiteLevel | 16383 (camera white 15360) | 16383 | 16383 |
| AsShotNeutral | [0.4883, 1, 0.5636] (camera WB [2097, 1024, 1817]) | [1.4609, 1, 0.7305] | camera-space neutral from the measured matrix, typically [0.37-0.50, 1, 0.56-0.75] |
| Daylight neutral | [0.3744, 1, 0.7402] (LibRaw daylight multipliers) | not modelled | reference record |
| XYZ-to-camera (D65) | [[0.8161, -0.2947, -0.0739], [-0.4811, 1.2668, 0.2389], [-0.0437, 0.1229, 0.6524]] | sRGB XYZ-to-RGB (virtual sRGB camera, identity camera-to-working) | measured matrix as ColorMatrix2; ColorMatrix1 by Bradford adaptation |

The old run recorded a neutral with R above 1, which is the inverse of a real Sony neutral (R and B
both below 1). The cause was the virtual sRGB camera plus a user_override WB record: the linear-sRGB
illuminant neutral was written straight into AsShotNeutral. Pushing that neutral through the measured
matrix gives a camera neutral of about [0.49, 1, 0.61], which is close to the real AsShotNeutral, so
the fix is the matrix, not the illuminant estimate. The measured matrix is written as ColorMatrix1/2 and
the inverse ISP uses it scaled so that a D65 white reaches 1 in the brightest camera channel, so the
mosaic is in real ILCE-1M2 camera space and no user_override WB record is written unless the owner
sets one.

## Exposure placement (black-subtracted, EV below the camera white point)

| Plane percentile | ILCE-1M2 sample | run_20261029_094146 |
|---|---|---|
| G p50 | -3.66 EV (1690 DN) | about -5.4 EV (876 DN) |
| G p99 | -1.71 EV (5065 DN) | about -3.9 EV (1556 DN) |
| G p99.9 | -0.83 EV (8849 DN) | not reached |
| R p50 / p99 | -4.91 / -2.64 EV | -5.0 / -3.9 EV |
| B p50 / p99 | -4.58 / -2.49 EV | -5.6 / -4.0 EV |
| Pixels at WhiteLevel | 26 / 123 / 118 / 28 (R, G1, G2, B) | 39 |
| Fraction at or below black | about 0 | 0.23 % |

The synthetic frame sat 1.7 to 2.2 EV darker than the real exposure because `applied_exposure_scale`
reserved the whole configured HDR headroom (2^-3 = 0.125) even though the reconstructed radiance never
used it; the TIFF development then needed roughly a 6x alignment gain before the round trip matched.
The fix places the reconstructed HDR maximum that was actually produced half a stop below full well,
records the headroom used as `BaselineExposure`, and the TIFF developer applies that exposure so the
unaligned round trip is comparable without a large alignment matrix.

## Noise (single-frame photon transfer, lower-envelope fit on detrended flat blocks)

| Quantity | ILCE-1M2 sample | run_20261029_094146 | Adopted |
|---|---|---|---|
| Conversion gain (R / G1 / G2) | 2.37 / 2.08 / 2.08 e-/DN (B unreliable) | 3.28 e-/DN | 2.08 e-/DN (M, from the green planes) |
| Full well at camera white | 35.2 k / 30.9 k e- | 52 000 e- | 33 000 e- (I, so full well lands at WhiteLevel) |
| Read noise (intercept) | 4.1 / 8.2 / 8.4 e- (scene-texture inflated) | 3.0 e- | 4.0 e- (I, upper bound; interval 0.3-10) |
| Darkest blocks sigma (MAD) | 11.9 / 23.7 / 23.7 / 16.3 DN | not measured | reference record |
| Dark tail skew / excess kurtosis | 3.6-6.3 / 42-177 | Gaussian | reference record (shot-noise dominated tail) |
| Positive tail beyond 4 sigma | 0.6-0.7 % | Gaussian 0.006 % | reference record |
| Row banding rms (dark blocks) | 6.7 / 14.5 / 15.0 / 10.3 DN (upper bound) | 0.5 e- (0.15 DN) | 0.5 e- kept (F); measured ceiling recorded |
| Column banding rms (dark blocks) | 4.8 / 10.6 / 9.8 / 6.2 DN (upper bound) | 0.5 e- | 0.5 e- kept (F); measured ceiling recorded |
| PSD low/high off-axis ratio | 500-1470 (scene dominated) | n/a | reference record |
| Hot pixels in dark regions | 77 in R (6e-6 of the plane), 0 elsewhere | 1e-5 | 2e-6 (I) |
| Dead pixels | 1 in B (2e-8) | 1e-5 | 2e-8 (I) |

The banding numbers from this frame are dominated by scene gradients inside the darkest blocks, so they
are ceilings. They are stored as reference records next to the fabricated 0.5 e- prior rather than being
adopted, and the realism report compares the run's own banding rms against them.

## Geometry and generated content

| Quantity | ILCE-1M2 sample | run_20261029_094146 | Fix |
|---|---|---|---|
| DefaultCropOrigin / Size | 0, 0 / 8640 x 5760 (whole frame is real) | 0, 0 / 8640 x 5760 | unchanged, so the full outpainted frame stays available |
| DefaultUserCrop (DNG 1.6) | absent | absent | content box [1048/5760, 0, 4712/5760, 1] (top, left, bottom, right), rounded inward to even rows; older readers ignore it |
| Outpainted rows | none | rows 0-1047 and 4713-5760 (36.35 %) | `confidence/outpaint_mask.npy` separate from the generated-SR-detail mask |
| Generated SR detail mask | n/a | counted the outpaint bands as SR detail | learned-backend content only |

For evaluation crops without any invented edge extension, run with `SENSOR_FIT_MODE = "CROP_HORIZONTAL"`.

## Honest no-ops and the 2K/4K comparison

- DISTORTION was on with k1 = k2 = 0: no lens is calibrated, so every profile keeps k1 = k2 = 0 and no
  geometry is invented; the run report now states that distortion is enabled but not modelled, and lists
  any optical effect that is enabled while OPTICAL_SIMULATION is off as not applied.
- C2PA unavailable was recorded in provenance.json but not in the run report fallbacks; it is now a
  fallback with its reason. The notebook configures no C2PA signer, so signing stays skipped and reported.
- The noise gates compared the simulator with itself; they are now labelled self-consistency, and a
  REFERENCE_RAW option produces the realism rows (histograms, noise versus signal, PSD with banding,
  WB and matrix) side by side with the reference.
- The 2K/4K comparison analysed 640 x 960 views (a 9x downsample) and so could not see detail above the
  2K Nyquist. Crops are now full-resolution windows (analysis factor 1), shown 1:1, with the spectral
  energy fraction above the low-resolution source's Nyquist frequency.

- The validation HTML now opens with a "Read this first" block: the unaligned round-trip numbers next
  to the aligned ones, the alignment gain (and its EV), every gate that passed within 10 % of its
  threshold, and every skipped or no-op feature (fallbacks such as C2PA unavailable or zero distortion,
  and checks that were not measured).

## Reproduction

The measurements were made with `rawpy` on the single ARW; the script reads the file once, keeps one
plane at a time, computes 32 x 32 detrended block statistics, a photon-transfer fit (weighted least
squares of variance against signal over the 5 % lower variance envelope of the blocks in each signal
bin, weighted by the square root of the bin's block count), dark-block tail statistics, row and column banding, a dark-window PSD and isolated-defect counts. The same
procedure is implemented in `synthraw.realism` and runs on Colab against any REFERENCE_RAW:
`measure_raw` reads a raw with rawpy (levels, camera white, AsShotNeutral, XYZ-to-camera matrix and the
D65 camera neutral) and keeps only the photosites inside a DNG's `DefaultUserCrop` within its
`DefaultCropOrigin`/`DefaultCropSize` (even edges, so the CFA phase holds), which leaves the synthetic
outpaint bands and sensor margins out of every statistic; `measure_mosaic` computes per-plane black-subtracted EV histograms, the
photon-transfer envelope and fit, dark-block tail and banding statistics (about each block's fitted
plane), the dark-window banding PSD and hot/dead counts, and `write_realism` writes
`validation/realism_comparison.json` and `.png` with reference, synthetic and ratio rows. The module
removes each block's fitted plane before measuring noise and banding, so its numbers on this frame are
somewhat lower than the scene-inflated ceilings in the tables above. On this frame several mid-tone
signal bins hold no flat block at all (envelope variance up to about 100 000 DN^2 near 1000 DN on the
greens), which drives a least-squares slope negative, so the module fits the envelope with Theil-Sen
(median pairwise slope, median intercept) instead. It reports R 2.36 e-/DN, 35.1 k e- full well and
5.0 e- read noise; G1 / G2 1.95 / 1.98 e-/DN, 28.9 k / 29.4 k e- and about 10 e-; B stays unreliable
(1.22 e-/DN). The realism rows are report only: no gate reads them.
