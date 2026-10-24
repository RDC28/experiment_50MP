# Phase 0 to 2 report: test_image_2.jpg

> Synthetic camera-grade masters reconstructed from an AI-generated source by classical interpolation (baselines) and by pretrained super-resolution (synthesised detail). None is a camera capture or camera RAW file and none contains recovered sensor measurements.

Status: **complete**

## 1. Run summary

| Item | Value |
|---|---|
| Run ID | 20261024T093611Z_5fe78cd3_baselines |
| Source | `test_image_2.jpg` |
| Source SHA-256 | `5fe78cd34888767d3691db81c288e7492800f40032ffe8f22cf66b63f730bd07` |
| Input dimensions | 6336 x 2688 |
| Output dimensions | 10852 x 4604 (49.963 MP) |
| Pipeline version | synthetic_raw_lab 0.1.0, commit `9d3765f9d718` |
| Notebook version | 00_Feasibility_Baselines/1.1 |
| Models | swinir_m_x4_classical (SwinIR-M x4, pixel-shuffle upsampler (11.9M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)) on NVIDIA L4 (float32); swinir_m_x4_realsr_gan (SwinIR-M x4, nearest+conv upsampler (11.7M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)) on NVIDIA L4 (float32) |
| GPU | NVIDIA L4 (22.49 GB) |
| Platform | Linux-6.6.122+-x86_64-with-glibc2.39, Python 3.13.16 |
| Total runtime | 1114.6 s |
| Peak resident memory | 5011.6 MB |
| Storage used | 1,794.4 MB |

## 2. Scientific limitations

- The source is a rendered, display-referred image. Linearising it gives linear-light display-referred values, not scene-referred radiance; the EXR master is synthetic linear RGB.
- Interpolation estimates samples between source pixels. It adds no information beyond the source sampling limit; every output pixel is an estimate.
- Pretrained super-resolution predicts detail above the source sampling limit from priors learned on other photographs. That detail is synthesised, not recovered, and may be wrong even when it looks plausible; a sharper result is not accepted for looking sharper (PRD 10.5).
- 16-bit and floating-point encodings add numerical precision, not dynamic range or recovered detail.
- Source-consistency metrics show agreement with the source after reduction; they cannot prove that detail above the source resolution is correct.

## 3. Input validation (PRD 3.4)

| Check | Result |
|---|---|
| Format | JPEG (7,949,019 bytes), decoded by opencv |
| Read-only stored copy | `5fe78cd34888_test_image_2.jpg` (read-only: True) |
| Dimensions | 6336 x 2688 (17.03 MP), aspect 33:14 |
| Bit depth | 8-bit uint8 (effective 8-bit) |
| Channels | 3 in file, 3 colour |
| Alpha | none |
| ICC profile | none |
| Source interpretation | sRGB IEC 61966-2-1 (untagged_default) |
| Working signal | linear-light display-referred RGB (not scene-referred radiance) |
| EXIF | none |
| Embedded text keys | none |
| Clipping (at min / at max) | R 0.023% / 0.000%, G 0.006% / 0.000%, B 0.000% / 0.000% |
| JPEG quality estimate | 99.1 |
| 8x8 blockiness ratio | 1.011 (about 1.0 means no block artefacts) |
| Integrity | structure verified and fully decoded |
| Duplicates | none |

Warnings:

- no embedded ICC profile; assumed sRGB (override with source_color_space)

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
| ingest_and_validate | Preservation | yes | 2.42 | uint8 (8-bit) -> float32 linear working space | 2689 | source_color_space=auto, working_color_space=linear_rec709 | no embedded ICC profile; assumed sRGB (override with source_color_space) |
| memory_benchmark | Measurement (reads data, produces no image data) | yes | 1.59 | - -> - | 2149 | width=10852, height=4604 | - |
| save_working_source | Packaging | yes | 5.83 | float32 -> float32 | 2353 | format=OpenEXR float32 | - |
| baseline_lanczos | Interpolation | yes | 10.55 | float32 linear -> float32 linear | 3251 | method=Lanczos-3 windowed sinc (a = 3, 6x6 taps when upscaling), Pillow float32, width=10852, height=4604, scale_x=1.712753, scale_y=1.712798, space=linear_rec709 | - |
| export_exr_lanczos | Packaging | yes | 9.93 | float32 linear -> half linear | 3034 | precision=half, compression=zip | - |
| encode_srgb16_lanczos | Packaging | yes | 2.38 | float32 linear -> uint16 sRGB | 3104 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_lanczos | Packaging | yes | 6.13 | uint16 sRGB -> uint16 sRGB TIFF | 3029 | compression=zlib | - |
| source_consistency_lanczos | Measurement (reads data, produces no image data) | yes | 36.54 | uint16 sRGB -> metrics | 5003 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| baseline_bicubic | Interpolation | yes | 10.34 | float32 linear -> float32 linear | 3252 | method=Keys cubic convolution (a = -0.5, 4x4 taps when upscaling), Pillow float32, width=10852, height=4604, scale_x=1.712753, scale_y=1.712798, space=linear_rec709 | - |
| export_exr_bicubic | Packaging | yes | 10.07 | float32 linear -> half linear | 3036 | precision=half, compression=zip | - |
| encode_srgb16_bicubic | Packaging | yes | 2.34 | float32 linear -> uint16 sRGB | 3118 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_bicubic | Packaging | yes | 6.14 | uint16 sRGB -> uint16 sRGB TIFF | 3031 | compression=zlib | - |
| source_consistency_bicubic | Measurement (reads data, produces no image data) | yes | 36.35 | uint16 sRGB -> metrics | 5012 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| super_resolution_setup | Setup (loads model weights, produces no image data) | yes | 0.00 | - -> - | 2086 | backend=swinir | - |
| load_model_swinir_m_x4_classical | Setup (loads model weights, produces no image data) | yes | 0.63 | weights file -> network | 2104 | weights=swinir_m_x4_classical, device=auto, precision=auto | - |
| super_resolution_conservative | Inference | yes | 395.07 | float32 linear (sRGB-encoded for the network) -> float32 linear | 4348 | weights=swinir_m_x4_classical, native_scale=4, then=Lanczos-3 x0.4282 (downscale), tile=640 (60% of free CUDA memory at 33 kB per input pixel (SwinIR, float32)), overlap=64, strength=1.0, device=cuda, precision=float32 | - |
| source_projection_conservative | Interpolation | yes | 6.12 | float32 linear -> float32 linear | 3736 | strength=1.0, reduction=area average, upsampling=Keys bicubic | - |
| seam_check_conservative | Measurement (reads data, produces no image data) | yes | 4.04 | - -> - | 2824 | boundaries=54 | - |
| interpolation_difference_conservative | Measurement (reads data, produces no image data) | yes | 16.59 | - -> - | 3146 | reference=Lanczos-3 | - |
| export_exr_sr_conservative | Packaging | yes | 10.07 | float32 linear -> half linear | 3075 | precision=half, compression=zip | - |
| encode_srgb16_sr_conservative | Packaging | yes | 2.51 | float32 linear -> uint16 sRGB | 3175 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_sr_conservative | Packaging | yes | 6.14 | uint16 sRGB -> uint16 sRGB TIFF | 3047 | compression=zlib | - |
| source_consistency_sr_conservative | Measurement (reads data, produces no image data) | yes | 36.67 | uint16 sRGB -> metrics | 5011 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| load_model_swinir_m_x4_realsr_gan | Setup (loads model weights, produces no image data) | yes | 0.59 | weights file -> network | 2109 | weights=swinir_m_x4_realsr_gan, device=auto, precision=auto | - |
| super_resolution_balanced | Inference | yes | 402.06 | float32 linear (sRGB-encoded for the network) -> float32 linear | 4353 | weights=swinir_m_x4_realsr_gan, native_scale=4, then=Lanczos-3 x0.4282 (downscale), tile=640 (60% of free CUDA memory at 33 kB per input pixel (SwinIR, float32)), overlap=64, strength=1.0, device=cuda, precision=float32 | - |
| source_projection_balanced | Interpolation | yes | 6.08 | float32 linear -> float32 linear | 3743 | strength=1.0, reduction=area average, upsampling=Keys bicubic | - |
| seam_check_balanced | Measurement (reads data, produces no image data) | yes | 4.03 | - -> - | 2829 | boundaries=54 | - |
| interpolation_difference_balanced | Measurement (reads data, produces no image data) | yes | 16.30 | - -> - | 3152 | reference=Lanczos-3 | - |
| export_exr_sr_balanced | Packaging | yes | 9.79 | float32 linear -> half linear | 3073 | precision=half, compression=zip | - |
| encode_srgb16_sr_balanced | Packaging | yes | 2.46 | float32 linear -> uint16 sRGB | 3160 | transfer=sRGB, clip=[0, 1] | - |
| export_tiff16_sr_balanced | Packaging | yes | 6.04 | uint16 sRGB -> uint16 sRGB TIFF | 3060 | compression=zlib | - |
| source_consistency_sr_balanced | Measurement (reads data, produces no image data) | yes | 36.76 | uint16 sRGB -> metrics | 5011 | reduction=area average (box filter over each output pixel footprint), Pillow float32, compared=16-bit sRGB master vs source | - |
| crop_comparison | Measurement (reads data, produces no image data) | yes | 1.72 | uint16 sRGB -> uint8 sRGB review PNG | 2109 | crops=4 | - |

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
| RGB 8-bit | 149.9 | 149.9 | 0.033 |
| RGB 16-bit integer | 299.8 | 299.8 | 0.069 |
| RGB float16 | 299.8 | 299.8 | 0.07 |
| RGB float32 | 599.6 | 599.6 | 0.135 |
| RGBA float32 | 799.4 | 799.4 | 0.179 |
| Single-channel float32 mask | 199.9 | 199.9 | 0.046 |

Pipeline peak resident memory: 5011.6 MB; process lifetime peak: 5021.1 MB.

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
| PSNR (dB) | 51.75 | 51.37 | 64.22 | 44.03 |
| SSIM | 0.99811 | 0.99789 | 0.99986 | 0.99381 |
| MS-SSIM (luma) | 0.99964 | 0.99960 | 0.99998 | 0.99908 |
| Edge displacement mean (px) | 0.279 | 0.290 | 0.069 | 0.266 |
| Edge displacement p95 (px) | 1.000 | 1.000 | 0.000 | 1.000 |
| Edge pixel ratio | 0.982 | 0.973 | 1.003 | 1.000 |
| delta E 2000 mean | 0.1972 | 0.2085 | 0.0579 | 0.3000 |
| delta E 2000 p95 | 0.5491 | 0.5799 | 0.1694 | 0.8485 |
| L* mean shift | 0.0074 | 0.0096 | -0.0015 | -0.0062 |
| L* std ratio | 0.9996 | 0.9994 | 1.0001 | 1.0011 |
| L* Wasserstein-1 | 0.0155 | 0.0179 | 0.0084 | 0.0282 |

SR modes with a low-frequency correction were pulled towards the source under the same area reduction these metrics use, so their PSNR, SSIM and delta E are partly true by construction. Edge displacement and the interpolation-difference statistics below remain independent checks.

LPIPS: not computed: LPIPS needs pretrained network weights, which Phase 0/1 does not download.

Image statistics and clipping:

| Quantity | lanczos | bicubic | sr_conservative | sr_balanced |
|---|---|---|---|---|
| Linear mean R/G/B | 0.0167 / 0.0246 / 0.0332 | 0.0167 / 0.0246 / 0.0332 | 0.0167 / 0.0246 / 0.0332 | 0.0167 / 0.0246 / 0.0332 |
| Linear min | -0.0010 | -0.0003 | -0.0046 | -0.3586 |
| Linear max | 1.0294 | 1.0272 | 1.0353 | 1.0255 |
| Samples below 0 | 0.0031% | 0.0024% | 0.0047% | 0.1140% |
| Samples above 1 | 0.0000% | 0.0000% | 0.0001% | 0.0000% |
| Clipped in TIFF | 0.0032% | 0.0024% | 0.0048% | 0.1141% |
| Float32 pixel SHA-256 | `38d93baae6eebaa9...` | `f661a3b4690479bf...` | `81d1b865346b47f7...` | `bf3ef4894977fd49...` |

Crop high-frequency energy (variance of the luma Laplacian, encoded domain):

| Crop | lanczos | bicubic | sr_conservative | sr_balanced |
|---|---|---|---|---|
| center | 0.000051 | 0.000046 | 0.000097 | 0.000796 |
| detail_1 | 0.000770 | 0.000725 | 0.000964 | 0.070264 |
| detail_2 | 0.000666 | 0.000625 | 0.000839 | 0.096265 |
| detail_3 | 0.000522 | 0.000491 | 0.000669 | 0.086380 |

Export validation (PRD 23.6):

| File | Kind | Size | Compression | Checks | SHA-256 |
|---|---|---|---|---|---|
| `test_image_2_source_lin_rec709_f32.exr` | exr_float_linear_rec709 | 87.2 MB | ZIP_COMPRESSION | 8/8 passed | `6b5eda62f2d3701a...` |
| `test_image_2_lanczos_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 183.4 MB | ZIP_COMPRESSION | 8/8 passed | `a52b637541c06f17...` |
| `test_image_2_lanczos_10852x4604_srgb16.tif` | tiff16_srgb | 234.6 MB | ADOBE_DEFLATE | 11/11 passed | `6be871126545cfa1...` |
| `test_image_2_bicubic_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 182.1 MB | ZIP_COMPRESSION | 8/8 passed | `82083e2da3df41e4...` |
| `test_image_2_bicubic_10852x4604_srgb16.tif` | tiff16_srgb | 231.9 MB | ADOBE_DEFLATE | 11/11 passed | `44665d82ecedf9c8...` |
| `test_image_2_sr_conservative_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 186.0 MB | ZIP_COMPRESSION | 8/8 passed | `9c3ee6ae27b8014a...` |
| `test_image_2_sr_conservative_10852x4604_srgb16.tif` | tiff16_srgb | 237.3 MB | ADOBE_DEFLATE | 11/11 passed | `798f149d09d7504d...` |
| `test_image_2_sr_balanced_10852x4604_lin_rec709_half.exr` | exr_half_linear_rec709 | 191.3 MB | ZIP_COMPRESSION | 8/8 passed | `31d392c71d9dcc2e...` |
| `test_image_2_sr_balanced_10852x4604_srgb16.tif` | tiff16_srgb | 240.5 MB | ADOBE_DEFLATE | 11/11 passed | `f4ea9262e89696cf...` |

## 8b. Pretrained super-resolution (PRD 10, experiment E02)

Backend `swinir`; scale handling: network at its native x4, then Lanczos-3 in linear light to the exact target size.

| Output | Mode | Weights | Device | Tile / overlap px | Tiles | Inference s | Peak device MB | Memory fallbacks | Blend error p99 (tile disagreement low-freq. / raw), 8-bit | Hard-seam height max, 8-bit | Seam check | Low-frequency residual (linear) before -> after | vs Lanczos mean / p99 (8-bit luma) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sr_conservative | Mode A, conservative | swinir_m_x4_classical | NVIDIA L4 float32 | 640 / 64 | 55 (55 calls) | 345 | 4400 | none | 0.002 (0.01 / 0.05) | 0.000 | pass | 0.000443 -> 0.000108 | 0.3547 / 2 |
| sr_balanced | Mode B, balanced | swinir_m_x4_realsr_gan | NVIDIA L4 float32 | 640 / 64 | 55 (55 calls) | 352 | 4400 | none | 2.234 (10.07 / 17.87) | 0.000 | FAIL | 0.004484 -> 0.000844 | 1.2667 / 20 |

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
- Kernel overshoot outside [0, 1] (a ringing indicator) is largest for lanczos (0.0032% of samples); the TIFF clips it, the EXR keeps it.
- Phase 2 exit criterion not met: a seam gate failed.
- sr_conservative: source PSNR 64.22 dB vs 51.75 dB for lanczos; edge displacement 0.069 px vs 0.279 px; mean departure from Lanczos 0.3547 8-bit luma codes (p99 2).
- sr_balanced: source PSNR 44.03 dB vs 51.75 dB for lanczos; edge displacement 0.266 px vs 0.279 px; mean departure from Lanczos 1.2667 8-bit luma codes (p99 20).
- Visual acceptance (PRD 25.2: seams, ringing, halos, banding) needs human review of the crop panels at 100% and 200%.
- Detail in the SR masters above the source sampling limit was inferred by pretrained networks; source consistency shows it agrees with the source after reduction, not that it is correct.

## 11. Provenance record (PRD 6.2)

| Item | Value |
|---|---|
| Source SHA-256 | `5fe78cd34888767d3691db81c288e7492800f40032ffe8f22cf66b63f730bd07` |
| Decoded pixel SHA-256 | `ba0e30d126203e093dd6b00c0d97a19dff0b426e40a1a5f5c306a34821f1c6d3` |
| Package | synthetic_raw_lab 0.1.0, commit `9d3765f9d718` |
| Notebook version | 00_Feasibility_Baselines/1.1 |
| Models and weight hashes | swinir_m_x4_classical (SwinIR-M x4, pixel-shuffle upsampler (11.9M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)), SHA-256 `4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215`, https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth on NVIDIA L4 (float32); swinir_m_x4_realsr_gan (SwinIR-M x4, nearest+conv upsampler (11.7M parameters), Apache-2.0 (SwinIR code and released weights, Liang et al. 2021)), SHA-256 `b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6`, https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth on NVIDIA L4 (float32) |
| GPU | NVIDIA L4 |
| Random seeds | {"random_seed": 42, "used_by": "torch.manual_seed before SR inference (inference itself draws no random numbers)"} |
| Runtime | 1114.6 s |

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
| rawpy | 0.27.1 |
| scikit-image | 0.25.2 |
| scipy | 1.16.3 |
| spandrel | 0.4.2 |
| tifffile | 2026.10.15 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |

Output checksums (SHA-256):

| File | SHA-256 |
|---|---|
| `test_image_2_source_lin_rec709_f32.exr` | `6b5eda62f2d3701ac3e20c94812a6dfdf0a5ae3958f88cf4b88f68031eff92a0` |
| `test_image_2_lanczos_10852x4604_lin_rec709_half.exr` | `a52b637541c06f1797e7e1cc831273cb0e22f7bca2a924e43debf2a05187166d` |
| `test_image_2_lanczos_10852x4604_srgb16.tif` | `6be871126545cfa132364b0c70e89c97a1633c04cbd125046188d52ca19f70ff` |
| `test_image_2_bicubic_10852x4604_lin_rec709_half.exr` | `82083e2da3df41e412a6facda67abe6b5171449c7c4daacc4aaea4aea36f9d6a` |
| `test_image_2_bicubic_10852x4604_srgb16.tif` | `44665d82ecedf9c8ff7105453d9ad3dd8654c00c6df8bb3cf7ce59b074ce500a` |
| `test_image_2_sr_conservative_10852x4604_lin_rec709_half.exr` | `9c3ee6ae27b8014a9591914d685c06e685ca71218a4ff33fb2f32b3a06ee299a` |
| `test_image_2_sr_conservative_10852x4604_srgb16.tif` | `798f149d09d7504d81d8923195e2a43c24ae10418eb80834bfca436135a53c18` |
| `test_image_2_sr_balanced_10852x4604_lin_rec709_half.exr` | `31d392c71d9dcc2ec393448d21ef6ce16d640c4e626b37b339a8b8cb6f15cc29` |
| `test_image_2_sr_balanced_10852x4604_srgb16.tif` | `f4ea9262e89696cf8b5b9a9aee1236b0ccfe5b0e5d13479ca372bf4ec08f2525` |

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
