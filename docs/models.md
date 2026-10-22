# Pretrained super-resolution models (PRD 10.1)

Phase 2 uses existing pretrained networks for inference only: no training, fine-tuning or distillation. This
page records which networks were evaluated, which one is the default and why, and the exact versions, weight
sources, checksums, licences and commercial-use terms. `src/sr_models.py` holds the same registry in code; a
run records the weights it used, with their checksum, in `manifest.json`, the report and every master's
sidecar.

## Decision

| Setting | Value |
|---|---|
| Default backend | `swinir` |
| Mode A, conservative | SwinIR-M x4 classical (fidelity network) |
| Modes B and C, balanced and aggressive | SwinIR-M x4 real-world GAN (perceptual network) |
| Faster alternative (`sr_backend: realesrgan`) | RealESRNet_x4plus (fidelity), RealESRGAN_x4plus (perceptual) |
| Loader | [spandrel](https://github.com/chaiNNer-org/spandrel) 0.4.2 (MIT, maintained) builds the architecture from the state dict |
| Native scale | x4 for every network, then Lanczos-3 in linear light to the exact target size |
| Tiles | automatic size, at least 256 px, starts on the 8 px attention-window grid; overlap 64 px (PRD 6.1), not less |

SwinIR was chosen because it was the more faithful family on every measure below, and it changed the
astronaut's face visibly less than Real-ESRGAN (PRD 11: identity before sharpness). Its cost is speed: about
2.5 times slower than Real-ESRGAN on Apple MPS, and it has no float16 support in spandrel, so it runs in
float32 on CUDA too. Real-ESRGAN stays available as the faster backend.

## Weights

All files are downloaded at run time from the projects' own GitHub releases into `<project>/weights` (or
`sr_weights_dir`), which Git ignores. The SHA-256 is checked after every download and before every load, and
`torch.load(weights_only=True)` reads them as plain tensors.

| Registry name | File (release) | Size | SHA-256 |
|---|---|---:|---|
| `swinir_m_x4_classical` | [`001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth) (SwinIR v0.0) | 67,869,037 B | `4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215` |
| `swinir_m_x4_realsr_gan` | [`003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth`](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth) (SwinIR v0.0) | 67,129,861 B | `b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6` |
| `realesrnet_x4plus` | [`RealESRNet_x4plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth) (Real-ESRGAN v0.1.1) | 67,040,989 B | `a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df` |
| `realesrgan_x4plus` | [`RealESRGAN_x4plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) (Real-ESRGAN v0.1.0) | 67,040,989 B | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |

Source repositories at the time of the check (2026-10-22): [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
(last commit `6545850` on 2022-12-04, not archived) and [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
(last commit `a4abfb2` on 2024-04-02, not archived). Neither is actively developed, which is why the code that
builds the networks comes from spandrel rather than from either repository.

## Licences and commercial use

| Network | Code and weights licence | Training data | Commercial use |
|---|---|---|---|
| SwinIR-M x4 classical | Apache-2.0 (SwinIR repository, which builds on Swin Transformer and KAIR, both MIT) | DF2K: DIV2K + Flickr2K | Licence allows it; see the data caveat |
| SwinIR-M x4 real-world GAN | Apache-2.0 | DIV2K + Flickr2K + OST | Licence allows it; see the data caveat |
| RealESRNet / RealESRGAN x4plus | BSD-3-Clause (Copyright (c) 2021 Xintao Wang) | DF2K + OST | Licence allows it; see the data caveat |
| spandrel (loader code) | MIT | none | Allowed |

Data caveat: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) states "This dataset is made available for
academic research purpose only", and Flickr2K consists of Flickr photographs under their owners' terms.
Whether weights trained on research-only data may be used commercially is legally unsettled. This experiment
is research; any commercial use of its SR outputs needs a separate legal review. SwinIR-L (trained also on
FFHQ, licensed CC BY-NC-SA 4.0) was evaluated but is not used, partly for that reason.

## Evaluation

### Controlled reference test

Six reference crops (four scikit-image photographs, two regions of the synthetic test image) were reduced 4x by
area averaging, restored with each network at x4, and compared with the reference. Lanczos is the classical
baseline. Apple M1, MPS, float32, 2026-10-22; scratch script, not part of the package.

| Network | Parameters | PSNR dB (Lanczos 25.00) | SSIM (Lanczos 0.761) | Edge displacement px | High-frequency energy / reference | Source PSNR after re-reduction dB |
|---|---:|---:|---:|---:|---:|---:|
| SwinIR-M x4 classical | 11.9M | 25.01 | **0.789** | **0.95** | 0.46 | **40.1** |
| SwinIR-M x4 real-world GAN | 11.7M | 23.51 | 0.721 | 4.60 | 0.52 | 31.2 |
| SwinIR-L x4 real-world GAN | 28.0M | 23.68 | 0.734 | 5.17 | 0.46 | 32.3 |
| RealESRNet_x4plus | 16.7M | **25.29** | 0.769 | 5.22 | 0.08 | 36.2 |
| RealESRGAN_x4plus | 16.7M | 23.81 | 0.729 | 5.00 | 0.29 | 32.1 |
| realesr-general-x4v3 | 1.2M | 24.07 | 0.744 | 5.46 | 0.15 | 32.1 |

Edge displacement averages are dominated by the synthetic zone plate, whose fine rings every real-world model
erases (about 21 px there) and only the classical SwinIR keeps (0.8 px). On the photographs alone, SwinIR
classical stays the lowest (0.7 to 1.4 px against 0.9 to 3.3 px for the Real-ESRGAN networks).

What the numbers and the side-by-side crops show:

- **Real-world models smooth clean renders.** Real-ESRGAN networks are trained to remove blur, noise and JPEG
  damage, so on a clean input they also remove real fine texture: RealESRNet keeps 8% of the reference's
  high-frequency energy and paints skin, fur and fabric flat. AI-generated sources are clean renders, so this
  matters.
- **SwinIR classical is the most faithful.** Best SSIM and edge placement, the highest source consistency, and
  natural-looking texture. It suits Mode A ("minimal invented texture, strong edge preservation").
- **GAN networks add sharpness and alter faces.** Both GAN families score lower against the reference; on the
  astronaut, RealESRGAN visibly changed the face more than SwinIR's GAN, which also kept more texture
  (0.52 against 0.29 of the reference's high-frequency energy).

### Speed and memory per tile (Apple M1, MPS, float32)

| Network | Seconds per input megapixel | Device memory per input pixel |
|---|---:|---:|
| SwinIR-M (either) | 120 to 230 | about 33 kB (1.1 GB at 192 px, 2.1 GB at 256 px) |
| RRDBNet (Real-ESRGAN) | about 62 | about 17 kB at 256 px |

A 3840 x 2160 source is 8.3 input megapixels, plus the tile overlaps. Colab GPU throughput was not measured
here (no CUDA on the development machine).

### Tile alignment, overlap and size

SwinIR attends within 8 x 8 windows counted from the tile origin. Tiles whose starts are not multiples of 8
partition the overlap differently, and the perceptual network then synthesises different texture there: on a
1280 x 720 run the worst low-frequency tile disagreement fell from 6.1 to 2.9 code values once starts were
aligned (median 2.1 to 1.0), and from 1.05 to 0.44 for the classical network. The tiler aligns tile starts to
the network's window size (1 for the convolutional Real-ESRGAN).

Near a tile edge the networks also lack context. The tiler gives the outer quarter of each overlap zero weight
and crossfades the middle half. Measured with aligned tiles against an untiled reference (sRGB 8-bit code
values; "photo" is a 256 x 384 crop of the scikit-image astronaut, "chart" the synthetic test chart):

| Network | Tile / overlap | Tiled vs untiled p99 (max) | Overlap disagreement p99, raw | Low-frequency p99 |
|---|---|---:|---:|---:|
| classical | 128 / 32 | 0.24 to 0.86 (15 to 22) | 5.5 to 14.4 | 0.31 to 1.84 |
| classical | 128 / 64 | 0.11 to 0.35 (3 to 4) | 1.6 to 4.6 | 0.15 to 0.62 |
| classical | 192 / 64 | 0.05 to 0.18 (2 to 9) | 0.9 to 3.7 | 0.10 to 0.88 |
| classical | 256 / 64 | 0.01 to 0.07 (0.3 to 1.9) | 0.2 to 0.9 | 0.01 to 0.20 |
| real-world GAN | 128 / 32 | 1.4 to 5.4 (18 to 61) | 8.6 to 44 | 2.2 to 8.9 |
| real-world GAN | 128 / 64 | 1.1 to 1.8 (20 to 21) | 5.5 to 17.7 | 1.4 to 2.6 |
| real-world GAN | 192 / 64 | 0.26 to 0.90 (8 to 18) | 3.2 to 17.6 | 0.68 to 2.52 |
| real-world GAN | 256 / 64 | 0.14 to 0.48 (4.6 to 9.8) | 1.5 to 5.3 | 0.48 to 0.88 |

So the overlap floor is 64 px (the PRD's initial value) and automatic tiles are at least 256 px, at about
2.1 GB of MPS memory per tile; out-of-memory splitting can still go lower. Before alignment, a 96 px overlap
gave the same disagreement as 64 px on the 1280 x 720 run, which ruled out border effects and led to the
window-grid finding.

### Seam gates

Two gates are reported for every SR mode (provisional, PRD 25.3). Both are in sRGB 8-bit code values.

- **Soft seams, blend error at most 2.0 (p99).** Inside each crossfade the low-frequency difference between the
  two tiles (Gaussian of 2 source pixels) is multiplied by the smaller blending weight: the crossfaded result
  lies within that distance of the better-placed tile. Raw tile disagreement is not a seam measure: where one
  tile has almost no weight it counts texture the network synthesised from less context, which the crossfade
  discards. Measured against untiled references, blend error tracks the real tiling error and raw disagreement
  does not:

  | Network, content, tile / overlap | Blend error p99 | Low-frequency disagreement p99 | Tiled vs untiled p99 |
  |---|---:|---:|---:|
  | GAN, colour patches, 128 / 32 | 7.06 | 16.7 | 12.3 |
  | GAN, colour patches, 192 / 64 | 3.46 | 13.1 | 3.23 |
  | GAN, colour patches, 256 / 64 | 0.79 | 6.21 | 0.70 |
  | GAN, dense chart, 128 / 32 | 1.15 | 3.49 | 3.04 |
  | GAN, dense chart, 256 / 64 | 0.13 | 0.88 | 0.47 |
  | GAN, photo, 256 / 64 | 0.07 | 0.48 | 0.14 |
  | classical, any of the above, 256 / 64 | 0.00 to 0.03 | 0.00 to 0.20 | 0.00 to 0.07 |

- **Hard seams, height at most 1.0.** Across each tile-boundary line, luma is smoothed along the line (texture
  averages out, an offset does not) and the step across the line, three pixels wide, is compared with the step
  across lines 4 to 12 pixels away. The mean excess over samples where the Lanczos interpolation of the source
  is flat (so content edges on the line do not count) is the seam height. Seam-free SR output measures 0 to
  0.18; a step of k code values injected at a boundary measures about k (0.76 to 0.92 for 1, 1.75 to 1.92 for 2).

An earlier, ratio-based hard-seam score flagged a chart edge that happened to lie on a boundary and was
dominated by tiny steps in smooth gradients; the height above replaced it.
