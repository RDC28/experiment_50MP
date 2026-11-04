# Video to synthetic RAW - video v1

One Colab notebook: [video_to_raw_v1.ipynb](notebooks/video_to_raw_v1.ipynb). It converts one video clip into
a sequence of synthetic Sony A1 II style RAW frames with the PRD v6 pipeline, imported from `prd_v6` (inverse
ISP to a linear RGGB mosaic, A1 II shot and read noise, 16-bit DNG writer). Nothing from v6 is copied or
changed, and the v6 notebook and its 95% gate are untouched. There is no accuracy claim (section 4).

## Run in Colab

1. Open the notebook from GitHub in Colab and choose **Runtime > Change runtime type > A100 GPU** with
   **High-RAM**, or **L4 GPU**. The conversion runs on the CPU, and these runtimes bring more CPU cores, RAM
   and disk. A CPU-only runtime also runs, with a warning, and is slower.
2. **Runtime > Run all**. With `VIDEO = None` (the default) the driver cell shows a **Choose Files** button:
   pick one clip, for example a 1080p MP4 of about 5 seconds. Instead, `VIDEO` can be a direct URL, a Google
   Drive share link of a file shared as "Anyone with the link", or a path on the runtime such as
   `/content/drive/MyDrive/clip.mp4` (Drive is then mounted first, one prompt). `MAX_FRAMES = 10` converts only
   the first 10 frames, as a quick test.
3. A 1080p clip of about 5 seconds should finish well under an hour, setup included. When the run ends, also
   after a failure, the browser downloads `<run>_01_summary.zip`, then the DNG sequence as
   `<run>_02_dng_part1ofN.zip` and onwards. Allow multiple downloads if Chrome asks. If a download was
   blocked or the kernel crashed, paste the recovery line that the driver cell printed first into a new cell.

`DRIVE_MIRROR = True` mirrors the summary to Google Drive after every step, and the archives at the end.

The verdict is **COMPLETE** (every frame converted and validated, stability passed), **FLAGGED** (it says
which frames failed validation, which stability ratio failed, or that step 5 raised) or **INVALID** (a gating
step 1-4 failed; it says which and why).

## Defaults

### 1. Input

- One video from the settings cell: an upload, a URL, a Drive share link or a runtime path.
- ffprobe records the codec, profile, pixel format, size, display rotation, frame rate, frame count and the
  colour tags. ffmpeg decodes every frame to the stream's own planes (8 to 16 bits, no scaling and no colour
  conversion). `video_v1/ycbcr.py` then turns them into linear light in BT.709 primaries in four exact steps:
  - chroma upsampling at the stream's chroma siting;
  - the YCbCr matrix with the stream's range, limited or full;
  - the inverse transfer function;
  - a primaries change when the stream is not BT.709.
- Untagged streams are read the way players read them: HD as BT.709, SD as BT.601, limited range. The report
  lists every field that was assumed.
- HDR is decoded but not tone mapped:
  - HLG uses its inverse OETF, so the nominal peak becomes the sensor clip.
  - PQ maps 1000 cd/m2 to the sensor clip.
- A display rotation, such as a phone's portrait clip, is applied so the frames match what players show.
  Odd sizes lose their last row or column, because an RGGB mosaic needs even sides.
- Audio is dropped, because a DNG sequence has none. The report says whether the source had any.

### 2. Output

- One 16-bit RGGB DNG per frame at the source resolution: no super-resolution, no 4K or 50 MP upscaling.
- Files are named as a CinemaDNG-style sequence, `dng/<clip>/<clip>_000000.dng` and onwards, plus a manifest,
  `<clip>_manifest.json`, that holds each frame's time code, seed, timings and validation results.
- Each DNG carries the v6 tag set plus ImageNumber, FrameRate and, at 30 fps or below, a SMPTE TimeCodes tag.
  29.97 fps uses drop-frame time code.
- Every frame must open in LibRaw at the right size and pattern and pass Adobe `dng_validate`, as in v6.
- `timings.csv` records the seconds per frame for decode, RAW, write, LibRaw and dng_validate.
- The report estimates what a 4K UHD version would cost (see Notes).

### 3. Temporal consistency

- The illuminant, white balance and exposure are fixed once per clip, never per frame. They come from the
  v6 MAX preset's frozen illuminant prior (Planckian at the mired midpoint of 2850-7500 K), and render white
  is the sensor clip. `calibration.json` records them.
- Every frame gets fresh v6 shot and read noise from its own seed (the v6 noise seed plus the frame index), so
  a frame is reproducible but grain is never repeated between frames.

### 4. Proof

- Step 5 develops every DNG back through LibRaw exactly as v6 develops its 50 MP output (as-shot neutral,
  AHD demosaic), and compares each developed frame with its decoded source. It checks three ratios of developed to source: mean luma,
  R/G and B/G.
- The run is stable when:
  - every frame-to-frame step of each ratio is at most **0.5%**;
  - the clip-wide span of each ratio, (max - min) / median, is at most **2%**.
- Frames darker than linear mean luma 0.002 are reported but not scored. Fades and black frames have no
  colour to compare.
- `check/preview_side_by_side.mp4` shows the source beside the developed DNGs at the source frame rate. The
  report shows it with first, middle and last frame thumbnails and a stability plot.
- The report also has the per-frame timings and the total wall time.
- **No accuracy claim.** There is no real RAW of the clip to score against. The v6 benchmark result, each
  colour plane within the 95% target against one real A1 II RAW, does not transfer to video frames.

### 5. Deliverables

- `<run>_01_summary.zip`: report.html, summary.txt, the verdict, the manifest, convert.json, check.json,
  timings.csv, the preview MP4, the plot and the thumbnails.
- `<run>_02_dng_part1ofN.zip` and onwards: the DNG sequence in frame order, at most 500 MB each. Each part
  includes the sequence manifest. A run that ends early marks its last part `_partial`.
- The source clip stays on the runtime. Only its small text records are packaged.

### 6. Runtime

- Colab A100 or L4, or CPU-only with a warning. Nothing in this notebook uses the GPU.
- Preflight (step 3) stops the run before any frame is converted if:
  - host RAM is under 8 GB;
  - ffmpeg or ffprobe is missing;
  - dng_validate was not built;
  - the clip cannot be probed or its first frame cannot be decoded;
  - free disk is short for the DNGs, their zip copies and a reserve.

## Layout

```text
notebooks/   the notebook (two code cells: settings, driver; step 1 input and step 2 setup run inline)
source.py    ffprobe metadata and the ffmpeg frame stream (planes as decoded, rotation, early stop)
ycbcr.py     decoded planes to linear BT.709: chroma siting, matrix, range, transfer, primaries
sequence.py  CinemaDNG-style names, SMPTE time codes and the frame DNG (the v6 tag set plus frame tags)
preflight.py step 3: runtime tier, tools, the clip (upload, URL, Drive or path), probe, free disk
convert.py   step 4: clip calibration, per-frame RAW and validation, timings, the 4K estimate
check.py     step 5: develop back, stability check, side-by-side preview, plot
driver.py    steps 3-5 with gating, guarding and checkpoints
delivery.py  run state, verdict, split archives, Drive mirror, downloads and recovery (standard library only)
report.py    report.html and summary.txt
tests/       tests on tiny synthetic clips (a few 64x48 frames, no models and no real video)
```

## Tests

```sh
.venv/bin/python -m pytest video_v1/tests
```

These tests are not in the default `testpaths` in `pyproject.toml`. Adding them there would change
`pyproject.toml`, which the v6 notebook pins. The tests need ffmpeg and ffprobe on the PATH (they skip
otherwise), and they stand in a script for dng_validate, which is built only on Colab. The notebook runs only
on Colab.

## Notes

- **4K estimate.** The estimate is computed, not measured. Super-resolution uses the v6 run
  20261104T064058Z_ad151cd: on an A100, HAT-L x4 took 2386 s for 70 tiles of 512 px with the x8
  self-ensemble. The other stages scale with pixel count from this run's own measured times. The report
  gives the cost with and without the ensemble.
- **Per-frame SR.** HAT-L has no temporal model, so a 4K version would need its own flicker check.
- **Known v6 limits carry over unchanged.**
  - The illuminant is a fixed prior, not an estimate from the frame, and that prior was v6's largest
    error.
  - The 50 MP grain was cleaner than the real camera. The noise model is the same one at source resolution.
