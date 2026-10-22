# Synthetic RAW 50MP Lab

A Google Colab research experiment that turns approximately 4K AI-generated images into 50-megapixel,
high-bit-depth image masters, and measures what that process can and cannot recover. The full specification
is [PRD.md](PRD.md).

Outputs are **synthetic camera-grade masters**: 16-bit TIFF and linear-light OpenEXR files reconstructed from
a rendered source. They are never camera RAW captures, and every report keeps preserved, interpolated,
inferred, simulated and encoded information apart.

## Status

| PRD phase (section 30) | State |
|---|---|
| Phase 0: feasibility and environment | Implemented |
| Phase 1: classical baselines | Implemented |
| Phase 2: pretrained super-resolution | Implemented |
| Phases 3 to 6: specialised notebooks, universal notebook, sensor simulation, evaluation | Not started |

Phase 0 exit criterion: a valid 50 MP master created and exported from a 4K input by classical
interpolation. Phase 1 exit criterion: baselines that are reproducible and measurable. Phase 2 exit criterion:
full-resolution SR runs successfully and produces no major seams. The report of every standard or research run
states whether its run met them.

## Repository layout

The layout follows PRD section 4.4. Modules are added when the phase that needs them is built.

```text
PRD.md                         specification
README.md
requirements.txt               exact versions the tests were run with
pyproject.toml                 package metadata (minimum versions) and tool settings
configs/baseline_default.yaml  Phase 0/1 parameters, every key documented in src/config.py
configs/super_resolution_default.yaml  Phase 2 profile: the baselines plus pretrained SR
docs/models.md                 SR model evaluation, weights, checksums, licences, commercial-use terms
notebooks/00_Feasibility_Baselines.ipynb  Phases 0 to 2
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
  comparison.py                identical evaluation crops across methods (PRD 9.2, 26)
  environment.py               runtime inspection, memory benchmark, storage estimate (PRD 21, 28.1)
  report_builder.py            Markdown report and figures (PRD 27)
  pipeline.py                  the Phase 0 to 2 run with per-stage logging (PRD 5.2, 22.3)
  test_images.py               deterministic synthetic test image
tests/                         pytest suite, including a notebook execution test
inputs/{human,night}/          read-only copies of ingested sources (not in Git)
outputs/{human,night,universal}/  full-resolution masters per run (not in Git)
weights/                       downloaded model weights (not in Git)
reports/                       per-run report, manifest and figures (not in Git)
```

Notebooks `01_Human_Crowd_50MP`, `02_Night_Scenery_50MP` and `03_Universal_50MP` arrive with Phases 3 and 4.

## Running in Google Colab

1. Open `notebooks/00_Feasibility_Baselines.ipynb` in Colab (File > Open notebook > GitHub, or upload it).
2. The repository is private: add a Colab secret named `GITHUB_TOKEN` holding a read-only token for it, or
   copy the repository to Google Drive and set `PROJECT_DIR`. The token is sent with each Git request only and
   is never stored in the clone.
3. To keep the masters, enable `MOUNT_DRIVE` and point `OUTPUT_ROOT` at a Drive folder.
4. Choose the input (`upload`, a `path`, or the `synthetic` test image) and a preset, then run all cells.
5. Super-resolution (`ENABLE_SUPER_RESOLUTION`, on by default) needs a GPU runtime (Runtime > Change runtime
   type) to finish in minutes. The first run downloads about 135 MB of SwinIR weights into `WEIGHTS_DIR`; point
   it at Drive to keep them.

Without `PROJECT_DIR`, the setup cell keeps a clone in `/content/experiment_50MP`, checks out `REPO_REF` there
on every run and reinstalls the package from it with `pip install`, so re-running the cell picks up new
commits. The package declares minimum versions only, so Colab keeps its preinstalled NumPy, SciPy, Pillow and
OpenCV; each run records the versions and the Git commit it actually used.

## Running locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .     # includes PyTorch and spandrel for super-resolution

python -m pytest                          # unit tests plus a run of the notebook on a small synthetic input
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
Apple MPS, then the CPU. The SR tests that need the real networks run only when the weights are already cached
in `weights/` or in `$SYNTHETIC_RAW_LAB_WEIGHTS`; they never download. On a shared machine, run a full 50 MP
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
   image), and the overlaps are crossfaded with the quarter next to each tile edge left out. Each strip is resampled with Lanczos-3 in
   linear light to its rows of the exact target, and a low-frequency correction pulls tone and colour back to
   the source (Mode A conservative: fidelity network; Mode B balanced: perceptual network; Mode C aggressive,
   experimental: perceptual network uncorrected). Out-of-memory tiles are split, then moved to the CPU. Seam
   gates and a |SR - Lanczos| heatmap show where detail was synthesised. Choices and licences:
   [docs/models.md](docs/models.md).
6. **Export** (Packaging). OpenEXR (half by default, linear Rec.709, chromaticities and a scene-referred
   caveat in the header) and 16-bit sRGB TIFF (embedded sRGB ICC profile, lossless Deflate, provenance JSON).
   Each file is re-read and checked: dimensions, sample type, colour metadata, provenance and a pixel hash.
7. **Measure** (Measurement). Reduces each TIFF master to source size by area averaging and reports PSNR,
   SSIM, MS-SSIM, symmetric Canny edge displacement, CIEDE2000 and the L* distribution shift. LPIPS needs
   pretrained weights and is not computed in Phase 0/1.
8. **Compare and report.** Saves identical crops from every baseline (centre plus the most detailed
   regions) next to the source and an amplified difference, previews, delta E maps, and writes
   `reports/<run>/report.md` plus `manifest.json` with the provenance record of PRD 6.2.
