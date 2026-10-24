# Phase 0 to 2 report: test_image_1.jpg

> Synthetic camera-grade masters reconstructed from an AI-generated source by classical interpolation (baselines) and by pretrained super-resolution (synthesised detail). None is a camera capture or camera RAW file and none contains recovered sensor measurements.

Status: **complete**

## 1. Run summary

| Item | Value |
|---|---|
| Run ID | 20261024T090241Z_40e30440_baselines |
| Source | `test_image_1.jpg` |
| Source SHA-256 | `40e304406633ec29dce11d9ca381d93b4ed24b4096c95a5cbf0b40973d0f6329` |
| Input dimensions | 6336 x 2688 |
| Output dimensions | 10852 x 4604 (49.963 MP) |
| Pipeline version | synthetic_raw_lab 0.1.0, commit `c3c1abbfa045` |
| Notebook version | 00_Feasibility_Baselines/1.1 |
| Models | swinir_m_x4_classical (SwinIR-M x4, pixel-shuffle upsampler (11.9M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)) on NVIDIA L4 (float32); swinir_m_x4_realsr_gan (SwinIR-M x4, nearest+conv upsampler (11.7M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)) on NVIDIA L4 (float32) |
| GPU | NVIDIA L4 (22.49 GB) |
| Platform | Linux-6.6.122+-x86_64-with-glibc2.39, Python 3.13.16 |
| Total runtime | 1121.9 s |
| Peak resident memory | 5009.9 MB |
| Storage used | 1,693.3 MB |

## 2. Scientific limitations

- The source is a rendered, display-referred image. Linearising it gives linear-light display-referred values, not scene-referred radiance; the EXR master is synthetic linear RGB.
- Interpolation estimates samples between source pixels. It adds no information beyond the source sampling limit; every output pixel is an estimate.
- Pretrained super-resolution predicts detail above the source sampling limit from priors learned on other photographs. That detail is synthesised, not recovered, and may be wrong even when it looks plausible; a sharper result is not accepted for looking sharper (PRD 10.5).
- 16-bit and floating-point encodings add numerical precision, not dynamic range or recovered detail.
- Source-consistency metrics show agreement with the source after reduction; they cannot prove that detail above the source resolution is correct.

## 3. Input validation (PRD 3.4)

| Check | Result |
|---|---|
| Format | JPEG (7,470,965 bytes), decoded by opencv |
| Read-only stored copy | `40e304406633_test_image_1.jpg` (read-only: True) |
| Dimensions | 6336 x 2688 (17.03 MP), aspect 33:14 |
| Bit depth | 8-bit uint8 (effective 8-bit) |
| Channels | 3 in file, 3 colour |
| Alpha | none |
| ICC profile | none |
| Source interpretation | sRGB IEC 61966-2-1 (untagged_default) |
| Working signal | linear-light display-referred RGB (not scene-referred radiance) |
| EXIF | none |
| Embedded text keys | none |
| Clipping (at min / at max) | R 0.014% / 0.389%, G 1.706% / 0.017%, B 1.291% / 0.009% |
| JPEG quality estimate | 99.1 |
| 8x8 blockiness ratio | 1.007 (about 1.0 means no block artefacts) |
| Integrity | structure verified and fully decoded |
| Duplicates | none |

Warnings:

- no embedded ICC profile; assumed sRGB (override with source_color_space)
- possible shadow clipping in G: 1.71% at minimum
- possible shadow clipping in B: 1.29% at minimum

![Source histograms](figures/source_histograms.png)

## 4. Target resolution (PRD 3.5)

| Quantity | Value |
|---|---|
| Mode | tolerance |
| Requested pixels | 50,000,000 |
| Achieved pixels | 49,962,608 |
| Pixel-count error | -0.0748% |
| Aspect-ratio error | -0.00263% |
| Nominal linear scale | 1.71342 |
| Scale x / y | 1.71275 / 1.71280 |

## 5. Processing log (PRD 27.2)

| Stage | Class (PRD 5.2) | Enabled | Seconds | Precision | Peak RSS MB | Parameters | Warnings, errors, notes |
|---|---|---|---|---|---|---|---|
| ingest_and_validate | Preservation | yes | 2.48 | uint8 (8-bit) -> float32 linear working space | 1102 | source_color_space=auto, working_color_space=linear_rec709 | no embedded ICC profile; assumed sRGB (override with source_color_space); possible shadow clipping in G: 1.71% at minimum; possible shadow clipping in B: 1.29% at minimum |
| memory_benchmark | Measurement (reads data, produces no image data) | yes | 1.60 | - -> - | 605 | width=10852, height=4604 | - |
| save_working_source | Packaging | yes | 5.84 | float32 -> float32 | 813 | format=OpenEXR float32 | - |
| baseline_lanczos | Interpolation | yes | 10.71 | float32 linear -> float32 linear | 1754 | method=Lanczos-3 windowed sinc (a = 3, 6x6 taps when upscaling), Pillow float32, width=10852, height=4604, scale_x=1.712753, scale_y=1.712798, space=linear_rec709 | - |
| export_exr_lanczos | Packaging | yes | 9.85 | float32 linear -> half linear | 1537 | precision=half, compression=zip | - |
| encode_srgb16_lanczos | Packaging | yes | 2.29 | float32 linear -> uint16 sRGB | 1599 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_lanczos | Packaging | yes | 6.48 | uint16 sRGB -> uint16 sRGB TIFF | 1634 | compression=zlib | - |
| source_consistency_lanczos | Measurement (reads data, produces no image data) | yes | 36.05 | uint16 sRGB -> metrics | 3546 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| baseline_bicubic | Interpolation | yes | 10.17 | float32 linear -> float32 linear | 1792 | method=Keys cubic convolution (a = -0.5, 4x4 taps when upscaling), Pillow float32, width=10852, height=4604, scale_x=1.712753, scale_y=1.712798, space=linear_rec709 | - |
| export_exr_bicubic | Packaging | yes | 9.72 | float32 linear -> half linear | 1575 | precision=half, compression=zip | - |
| encode_srgb16_bicubic | Packaging | yes | 2.31 | float32 linear -> uint16 sRGB | 1651 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_bicubic | Packaging | yes | 6.30 | uint16 sRGB -> uint16 sRGB TIFF | 1634 | compression=zlib | - |
| source_consistency_bicubic | Measurement (reads data, produces no image data) | yes | 35.67 | uint16 sRGB -> metrics | 3539 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| super_resolution_setup | Setup (loads model weights, produces no image data) | yes | 9.85 | - -> - | 1177 | backend=swinir | - |
| load_model_swinir_m_x4_classical | Setup (loads model weights, produces no image data) | yes | 1.50 | weights file -> network | 1423 | weights=swinir_m_x4_classical, device=auto, precision=auto | - |
| super_resolution_conservative | Inference | yes | 395.07 | float32 linear (sRGB-encoded for the network) -> float32 linear | 4208 | weights=swinir_m_x4_classical, native_scale=4, then=Lanczos-3 x0.4282 (downscale), tile=640 (60% of free CUDA memory at 33 kB per input pixel (SwinIR, float32)), overlap=64, strength=1.0, device=cuda, precision=float32 | - |
| source_projection_conservative | Interpolation | yes | 6.13 | float32 linear -> float32 linear | 3602 | strength=1.0, reduction=area average, upsampling=Keys bicubic | - |
| seam_check_conservative | Measurement (reads data, produces no image data) | yes | 4.01 | - -> - | 2688 | boundaries=54 | - |
| interpolation_difference_conservative | Measurement (reads data, produces no image data) | yes | 16.46 | - -> - | 3007 | reference=Lanczos-3 | - |
| export_exr_sr_conservative | Packaging | yes | 9.78 | float32 linear -> half linear | 2935 | precision=half, compression=zip | - |
| encode_srgb16_sr_conservative | Packaging | yes | 2.47 | float32 linear -> uint16 sRGB | 3041 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_sr_conservative | Packaging | yes | 6.24 | uint16 sRGB -> uint16 sRGB TIFF | 3024 | compression=zlib | - |
| source_consistency_sr_conservative | Measurement (reads data, produces no image data) | yes | 35.73 | uint16 sRGB -> metrics | 4899 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| load_model_swinir_m_x4_realsr_gan | Setup (loads model weights, produces no image data) | yes | 2.92 | weights file -> network | 2094 | weights=swinir_m_x4_realsr_gan, device=auto, precision=auto | - |
| super_resolution_balanced | Inference | yes | 401.67 | float32 linear (sRGB-encoded for the network) -> float32 linear | 4328 | weights=swinir_m_x4_realsr_gan, native_scale=4, then=Lanczos-3 x0.4282 (downscale), tile=640 (60% of free CUDA memory at 33 kB per input pixel (SwinIR, float32)), overlap=64, strength=1.0, device=cuda, precision=float32 | - |
| source_projection_balanced | Interpolation | yes | 6.03 | float32 linear -> float32 linear | 3716 | strength=1.0, reduction=area average, upsampling=Keys bicubic | - |
| seam_check_balanced | Measurement (reads data, produces no image data) | yes | 2.92 | - -> - | 2804 | boundaries=54 | - |
| interpolation_difference_balanced | Measurement (reads data, produces no image data) | yes | 15.74 | - -> - | 3157 | reference=Lanczos-3 | - |
| export_exr_sr_balanced | Packaging | yes | 9.74 | float32 linear -> half linear | 3058 | precision=half, compression=zip | - |
| encode_srgb16_sr_balanced | Packaging | yes | 2.46 | float32 linear -> uint16 sRGB | 3141 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_sr_balanced | Packaging | yes | 6.15 | uint16 sRGB -> uint16 sRGB TIFF | 3040 | compression=zlib | - |
| source_consistency_sr_balanced | Measurement (reads data, produces no image data) | yes | 35.67 | uint16 sRGB -> metrics | 5010 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| crop_comparison | Measurement (reads data, produces no image data) | yes | 1.84 | uint16 sRGB -> uint8 sRGB review PNG | 2088 | crops=4 | - |

## 6. Memory (PRD 21.3, 28.1)

Base-image estimates at the output size (decimal MB, excluding temporaries):

| Representation | Bytes/pixel | Estimated MB |
|---|---|---|
| RGB 8-bit | 3 | 149.9 |
| RGB 16-bit integer | 6 | 299.8 |
| RGB float16 | 6 | 299.8 |
| RGB float32 | 12 | 599.6 |
| RGBA float32 | 16 | 799.4 |
| Single-channel float32 mask | 4 | 199.9 |

Measured allocation benchmark:

| Representation | Estimated MB | Measured RSS MB | Allocate s |
|---|---|---|---|
| RGB 8-bit | 149.9 | 150.0 | 0.032 |
| RGB 16-bit integer | 299.8 | 299.8 | 0.066 |
| RGB float16 | 299.8 | 299.8 | 0.069 |
| RGB float32 | 599.6 | 599.6 | 0.136 |
| RGBA float32 | 799.4 | 799.4 | 0.179 |
| Single-channel float32 mask | 199.9 | 199.9 | 0.046 |

Pipeline peak resident memory: 5009.9 MB; process lifetime peak: 5013.2 MB.

## 7. Visual results (PRD 27.3)

Original source:

![Source](figures/source_preview.jpg)

Lanczos baseline at output size (review downscale):

![lanczos](figures/lanczos_preview.jpg)

Bicubic baseline at output size (review downscale):

![bicubic](figures/bicubic_preview.jpg)

Sr_conservative pretrained super-resolution at output size (review downscale):

![sr_conservative](figures/sr_conservative_preview.jpg)

Sr_balanced pretrained super-resolution at output size (review downscale):

![sr_balanced](figures/sr_balanced_preview.jpg)

Crop `center` (source, every method at 100% output pixels, amplified difference):

![center](figures/crop_center.png)

Crop `detail_1` (source, every method at 100% output pixels, amplified difference):

![detail_1](figures/crop_detail_1.png)

Crop `detail_2` (source, every method at 100% output pixels, amplified difference):

![detail_2](figures/crop_detail_2.png)

Crop `detail_3` (source, every method at 100% output pixels, amplified difference):

![detail_3](figures/crop_detail_3.png)

Difference heatmap, sr_conservative vs Lanczos at output resolution (where detail was synthesised or drifted):

![sr_conservative vs Lanczos](figures/sr_difference_conservative.png)

Difference heatmap, sr_balanced vs Lanczos at output resolution (where detail was synthesised or drifted):

![sr_balanced vs Lanczos](figures/sr_difference_balanced.png)

Difference map, lanczos reduced to source resolution vs source:

![delta E lanczos](figures/delta_e_lanczos.png)

Difference map, bicubic reduced to source resolution vs source:

![delta E bicubic](figures/delta_e_bicubic.png)

Difference map, sr_conservative reduced to source resolution vs source:

![delta E sr_conservative](figures/delta_e_sr_conservative.png)

Difference map, sr_balanced reduced to source resolution vs source:

![delta E sr_balanced](figures/delta_e_sr_balanced.png)

## 8. Quantitative results (PRD 27.4)

Source consistency, measured on the 16-bit TIFF master after area reduction to source size:

| Metric | lanczos | bicubic | sr_conservative | sr_balanced |
|---|---|---|---|---|
| PSNR (dB) | 55.23 | 54.93 | 66.91 | 53.21 |
| SSIM | 0.99853 | 0.99835 | 0.99987 | 0.99793 |
| MS-SSIM (luma) | 0.99981 | 0.99980 | 0.99998 | 0.99978 |
| Edge displacement mean (px) | 0.450 | 0.451 | 0.099 | 0.317 |
| Edge displacement p95 (px) | 1.000 | 1.000 | 0.000 | 1.000 |
| Edge pixel ratio | 0.992 | 0.985 | 1.003 | 1.011 |
| delta E 2000 mean | 0.1380 | 0.1456 | 0.0417 | 0.1624 |
| delta E 2000 p95 | 0.3617 | 0.3824 | 0.1119 | 0.4148 |
| L* mean shift | 0.0017 | 0.0021 | -0.0005 | -0.0005 |
| L* std ratio | 0.9999 | 0.9998 | 1.0000 | 1.0000 |
| L* Wasserstein-1 | 0.0047 | 0.0051 | 0.0023 | 0.0045 |

SR modes with a low-frequency correction were pulled towards the source under the same area reduction these metrics use, so their PSNR, SSIM and delta E are partly true by construction. Edge displacement and the interpolation-difference statistics below remain independent checks.

LPIPS: not computed: LPIPS needs pretrained network weights, which Phase 0/1 does not download.

Image statistics and clipping:

| Quantity | lanczos | bicubic | sr_conservative | sr_balanced |
|---|---|---|---|---|
| Linear mean R/G/B | 0.3598 / 0.1862 / 0.1255 | 0.3598 / 0.1862 / 0.1255 | 0.3598 / 0.1862 / 0.1255 | 0.3598 / 0.1862 / 0.1255 |
| Linear min | -0.0019 | -0.0012 | -0.0033 | -0.1701 |
| Linear max | 1.1002 | 1.0714 | 1.0459 | 1.0443 |
| Samples below 0 | 0.4226% | 0.4111% | 0.5068% | 0.4047% |
| Samples above 1 | 0.0528% | 0.0476% | 0.0734% | 0.0450% |
| Clipped in TIFF | 0.4754% | 0.4587% | 0.5802% | 0.4497% |
| Float32 pixel SHA-256 | `7f156f58af5a1966...` | `9f84bb4803921a82...` | `2239a0449fcbe183...` | `774401ce9b48a011...` |

Crop high-frequency energy (variance of the luma Laplacian, encoded domain):

| Crop | lanczos | bicubic | sr_conservative | sr_balanced |
|---|---|---|---|---|
| center | 0.000067 | 0.000061 | 0.000116 | 0.000325 |
| detail_1 | 0.000258 | 0.000243 | 0.000357 | 0.002292 |
| detail_2 | 0.000237 | 0.000222 | 0.000333 | 0.002136 |
| detail_3 | 0.000180 | 0.000169 | 0.000255 | 0.002816 |

Export validation (PRD 23.6):

| File | Kind | Size | Compression | Checks | SHA-256 |
|---|---|---|---|---|---|
| `test_image_1_source_lin_rec709_f32.exr` | exr_float_linear_rec709 | 92.3 MB | ZIP_COMPRESSION | 8/8 passed | `2f7a56235f239dce...` |
| `test_image_1_lanczos_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 161.5 MB | ZIP_COMPRESSION | 8/8 passed | `fc19fe3d01cdf4e0...` |
| `test_image_1_lanczos_10852x4604_srgb16.tif` | tiff16_srgb | 231.3 MB | ADOBE_DEFLATE | 11/11 passed | `56f45f4b42171c65...` |
| `test_image_1_bicubic_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 159.4 MB | ZIP_COMPRESSION | 8/8 passed | `e313759ed062ce69...` |
| `test_image_1_bicubic_10852x4604_srgb16.tif` | tiff16_srgb | 228.8 MB | ADOBE_DEFLATE | 11/11 passed | `dd75fdfd8e3319d7...` |
| `test_image_1_sr_conservative_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 165.8 MB | ZIP_COMPRESSION | 8/8 passed | `ea70ff40fdb9478d...` |
| `test_image_1_sr_conservative_10852x4604_srgb16.tif` | tiff16_srgb | 234.6 MB | ADOBE_DEFLATE | 11/11 passed | `4061d61a5ee895bc...` |
| `test_image_1_sr_balanced_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 165.5 MB | ZIP_COMPRESSION | 8/8 passed | `677df02566a6e9a0...` |
| `test_image_1_sr_balanced_10852x4604_srgb16.tif` | tiff16_srgb | 235.0 MB | ADOBE_DEFLATE | 11/11 passed | `d383c1628a9a1f73...` |

## 8b. Pretrained super-resolution (PRD 10, experiment E02)

Backend `swinir`; scale handling: network at its native x4, then Lanczos-3 in linear light to the exact target size.

| Output | Mode | Weights | Device | Tile / overlap px | Tiles | Inference s | Peak device MB | Memory fallbacks | Blend error p99 (tile disagreement low-freq. / raw), 8-bit | Hard-seam height max, 8-bit | Seam check | Low-frequency residual (linear) before -> after | vs Lanczos mean / p99 (8-bit luma) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sr_conservative | Mode A, conservative | swinir_m_x4_classical | NVIDIA L4 float32 | 640 / 64 | 55 (55 calls) | 345 | 4400 | none | 0.005 (0.02 / 0.06) | 0.001 | pass | 0.000875 -> 0.000241 | 0.2807 / 1 |
| sr_balanced | Mode B, balanced | swinir_m_x4_realsr_gan | NVIDIA L4 float32 | 640 / 64 | 55 (55 calls) | 352 | 4400 | none | 0.704 (2.76 / 9.35) | 0.016 | pass | 0.007733 -> 0.000926 | 0.4611 / 4 |

Blend error: inside each crossfade, the low-frequency (Gaussian of 2 source pixels) difference between the two tiles times the smaller blending weight, which bounds how far blending moves the image from the better-placed tile (the soft-seam measure). Tile disagreement itself, low-frequency and raw, also counts texture the networks synthesised differently where one tile has almost no weight. Hard-seam height: the luma step across each tile-boundary line beyond the step across nearby lines, on samples where the content is flat (0 means no hard seam). Both gates are provisional until calibrated in pilot runs (PRD 25.3); a mode passes when the blend error p99 is at most 2.0 and the hard-seam height at most 1.0 code values.

Mode intents:

- `sr_conservative`: Source consistency first: the fidelity network (pixel loss, no adversarial texture) and full low-frequency correction. Minimal invented texture; preferred for faces.
- `sr_balanced`: Moderate detail reconstruction: the perceptual (adversarial) network for sharper edges and inferred texture, with full low-frequency correction so tone and colour stay on the source. Default for general scenes.

## 9. Information provenance (PRD 27.5)

| Contribution | In this run |
|---|---|
| Original source data | All image structure. The source holds 17,031,168 samples per channel, 34.1% of the output pixel count. |
| Interpolated values | Every baseline sample (classical, deterministic kernels); in the SR masters, the final resample from the x4 grid and the low-frequency correction. |
| Learned inference | Detail above the source sampling limit in the SR masters (sr_conservative: swinir_m_x4_classical, sr_balanced: swinir_m_x4_realsr_gan) is synthesised by pretrained networks from learned priors; it is not recovered information. The baselines contain none. |
| Simulated sensor effects | None. |
| Added noise or grain | None. |
| File encoding | 16-bit sRGB integer TIFF (clips interpolation overshoot); half or float linear EXR (keeps overshoot). Precision only, no new information. |

## 10. Conclusions (PRD 27.6)

- Output size 10852x4604 (49,962,608 pixels) meets the configured pixel-count target.
- Phase 0 exit criterion met: a valid 50 MP master was created and exported from the source by classical interpolation, and every export passed read-back validation.
- Reproducibility was not re-run in this preset; compare the recorded working_pixel_sha256 values across runs (the research preset repeats each baseline).
- Phase 1 baselines are measurable: source-consistency metrics are reported for every baseline.
- Highest source PSNR: lanczos; lowest mean delta E 2000: lanczos.
- Kernel overshoot outside [0, 1] (a ringing indicator) is largest for lanczos (0.4754% of samples); the TIFF clips it, the EXR keeps it.
- Phase 2 exit criterion met by the automated checks: full-resolution SR ran for every mode, every export passed validation and no tile boundary exceeded the seam gates. Confirm visually.
- sr_conservative: source PSNR 66.91 dB vs 55.23 dB for lanczos; edge displacement 0.099 px vs 0.450 px; mean departure from Lanczos 0.2807 8-bit luma codes (p99 1).
- sr_balanced: source PSNR 53.21 dB vs 55.23 dB for lanczos; edge displacement 0.317 px vs 0.450 px; mean departure from Lanczos 0.4611 8-bit luma codes (p99 4).
- Visual acceptance (PRD 25.2: seams, ringing, halos, banding) needs human review of the crop panels at 100% and 200%.
- Detail in the SR masters above the source sampling limit was inferred by pretrained networks; source consistency shows it agrees with the source after reduction, not that it is correct.

## 11. Provenance record (PRD 6.2)

| Item | Value |
|---|---|
| Source SHA-256 | `40e304406633ec29dce11d9ca381d93b4ed24b4096c95a5cbf0b40973d0f6329` |
| Decoded pixel SHA-256 | `4c1b2455300e835903774e712fc59bcad3e9b655ea7ff5c8286e6f947afdfcb6` |
| Package | synthetic_raw_lab 0.1.0, commit `c3c1abbfa045` |
| Notebook version | 00_Feasibility_Baselines/1.1 |
| Models and weight hashes | swinir_m_x4_classical (SwinIR-M x4, pixel-shuffle upsampler (11.9M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)), SHA-256 `4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215`, https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth on NVIDIA L4 (float32); swinir_m_x4_realsr_gan (SwinIR-M x4, nearest+conv upsampler (11.7M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)), SHA-256 `b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6`, https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth on NVIDIA L4 (float32) |
| GPU | NVIDIA L4 |
| Random seeds | {"random_seed": 42, "used_by": "torch.manual_seed before SR inference (inference itself draws no random numbers)"} |
| Runtime | 1121.9 s |

Dependency versions:

| Package | Version |
|---|---|
| OpenEXR | 3.5.2 |
| PyYAML | 6.0.3 |
| matplotlib | 3.10.0 |
| numpy | 2.1.3 |
| opencv-python | 5.0.0.93 |
| opencv-python-headless | 5.0.0.93 |
| pillow | 11.3.0 |
| psutil | 5.9.5 |
| scikit-image | 0.25.2 |
| scipy | 1.16.3 |
| spandrel | 0.4.2 |
| tifffile | 2026.10.15 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |

Output checksums (SHA-256):

| File | SHA-256 |
|---|---|
| `test_image_1_source_lin_rec709_f32.exr` | `2f7a56235f239dcedba024766bc35fecd00956405527f428f2a1fb665507ce64` |
| `test_image_1_lanczos_10852x4604_lin_rec709_half.exr` | `fc19fe3d01cdf4e029ac72cab66d700f5c87c8766504050374211e8e8b01fca0` |
| `test_image_1_lanczos_10852x4604_srgb16.tif` | `56f45f4b42171c6572bf0f7d4aecbc8aff06a6b487f209cf6cf51bddc97b2752` |
| `test_image_1_bicubic_10852x4604_lin_rec709_half.exr` | `e313759ed062ce69cf270e6223e557e5df1a594bc80ac8375f21bdde8dff044f` |
| `test_image_1_bicubic_10852x4604_srgb16.tif` | `dd75fdfd8e3319d739ab3dcc94dc06adfa4971ad5796be34b3d58e0dc0fcbb94` |
| `test_image_1_sr_conservative_10852x4604_lin_rec709_half.exr` | `ea70ff40fdb9478d3360e5fd0a9e876d7e5cd252afd9014f9fce353545766424` |
| `test_image_1_sr_conservative_10852x4604_srgb16.tif` | `4061d61a5ee895bc93a7a417abda4c9750475cd7f46746f250526572e17cecae` |
| `test_image_1_sr_balanced_10852x4604_lin_rec709_half.exr` | `677df02566a6e9a0603a5301d61ac65aa4f5d0b892cc5f93a47bc470a724d3ce` |
| `test_image_1_sr_balanced_10852x4604_srgb16.tif` | `d383c1628a9a1f732e5e2dfebfcc40f870c9c6931fad0dca903117052e28f031` |

Configuration:

```json
{
  "profile": "baselines",
  "input_category": "universal",
  "runtime_preset": "standard",
  "quick_target_megapixels": 2.0,
  "target_megapixels": 50.0,
  "resolution_mode": "tolerance",
  "pixel_count_tolerance": 0.001,
  "dimension_multiple": 2,
  "preserve_aspect_ratio": true,
  "max_aspect_ratio_error": 0.001,
  "source_color_space": "auto",
  "working_color_space": "linear_rec709",
  "processing_precision": "float32",
  "baselines": [
    "lanczos",
    "bicubic"
  ],
  "enable_super_resolution": true,
  "sr_backend": "swinir",
  "sr_modes": [
    "conservative",
    "balanced"
  ],
  "sr_tile_size": 0,
  "sr_tile_overlap": 64,
  "sr_strength": 1.0,
  "preserve_source_structure": true,
  "sr_device": "auto",
  "sr_precision": "auto",
  "sr_weights_dir": "",
  "export_tiff16": true,
  "tiff_compression": "zlib",
  "export_exr": true,
  "exr_precision": "half",
  "exr_compression": "zip",
  "export_review": true,
  "review_max_edge": 2048,
  "save_intermediates": true,
  "crop_size": 256,
  "crop_count": 4,
  "crops": [],
  "random_seed": 42,
  "output_root": "",
  "reports_root": "",
  "inputs_root": ""
}
```
