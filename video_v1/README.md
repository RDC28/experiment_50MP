# Video to synthetic RAW - video v1

One Colab notebook: [video_to_raw_v1.ipynb](notebooks/video_to_raw_v1.ipynb). It converts one video clip into
synthetic Sony A1 II style RAW with the PRD v6 pipeline, imported from `prd_v6` (inverse ISP to a linear RGGB
mosaic, A1 II shot and read noise, 16-bit DNG writer). It writes three outputs: one 14-bit MLV raw video file
per clip, one developed 16-bit sRGB TIFF per frame and, optionally (`WRITE_DNG = True`), one 16-bit DNG per
frame. Nothing from v6 is copied or changed: the v6 notebook, its DNGs and its 95% gate are untouched. There is
no accuracy claim (section 4).

## Run in Colab

1. Open the notebook from GitHub in Colab and choose **Runtime > Change runtime type > A100 GPU** with
   **High-RAM**, or **L4 GPU**. The conversion runs on the CPU, and these runtimes bring more CPU cores, RAM
   and disk. A CPU-only runtime also runs, with a warning, and is slower.
2. **Runtime > Run all**. With `VIDEO = None` (the default) the driver cell shows a **Choose Files** button:
   pick one clip, for example a 1080p MP4 of about 5 seconds. Instead, `VIDEO` can be a direct URL, a Google
   Drive share link of a file shared as "Anyone with the link", or a path on the runtime such as
   `/content/drive/MyDrive/clip.mp4` (Drive is then mounted first, one prompt). Every frame of the clip is
   converted.
3. A 1080p clip of about 5 seconds should finish well under an hour, setup included. When the run ends, also
   after a failure, the browser downloads `<run>_01_summary.zip`, then the MLV clip, the TIFFs and (with
   `WRITE_DNG = True`) the DNGs as `<run>_NN_mlv_partKofN.zip`, then `<run>_NN_tiff_partKofN.zip`, then
   `<run>_NN_dng_partKofN.zip`, with NN numbered on continuously from 02 (section 5). Allow multiple
   downloads if Chrome asks. If a download was blocked or the kernel crashed, paste the recovery line that
   the driver cell printed first into a new cell.

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
- Audio is dropped, because none of the outputs carry it. The report says whether the source had any.

### 2. Output

Every output is at the source resolution: no super-resolution, no 4K or 50 MP upscaling. All three come
from the same per-frame RGGB mosaic, noise included.

- **MLV raw video**, `mlv/<clip>.MLV`: one Magic Lantern raw video file (MLV v2.0) per clip, 14-bit packed
  RGGB, with the RAWI, RAWC, IDNT, EXPO and WBAL blocks, one VIDF block per frame and the clip frame rate in
  the file header. The 14-bit samples are the DNG samples rounded to 14 bits (black 512, white 15360). The
  ColorMatrix1, the D65 white balance and a 180 degree shutter are recorded. A long clip is split into
  chunks of at most 480 MB, named `.MLV`, `.M00`, `.M01` as a camera names them; keep them
  in one folder.
  - [MLV App](https://mlv.app/) opens the clip (open the `.MLV`; it reads the `.Mnn` chunks beside it) and
    exports it to ProRes or CinemaDNG for DaVinci Resolve or Premiere Pro.
  - The pipeline reads the clip back in Python before delivery: the headers, the frame count and every
    frame decoded bit-exact against the mosaic it wrote (`mlv/<clip>_mlv_check.json`).
  - MLV needs a width that is a multiple of 8, so a width that is not loses up to 7 columns on the right in
    the MLV only.
- **Developed TIFFs**, `tiff/<clip>/<clip>_000000.tif` and onwards: each frame's DNG developed through LibRaw
  exactly as v6 develops its 50 MP output (as-shot neutral, AHD demosaic), as 16-bit sRGB RGB with an
  embedded sRGB ICC profile and lossless deflate. Each is read back (size, 16 bits, ICC profile).
- **DNG sequence** (optional, off by default; set `WRITE_DNG = True`), `dng/<clip>/<clip>_000000.dng` and
  onwards: one 16-bit RGGB DNG per frame, named as a CinemaDNG-style sequence. Each DNG carries the v6 tag
  set plus ImageNumber, FrameRate and, at 30 fps or below, a SMPTE TimeCodes tag (29.97 fps uses drop-frame
  time code), and BaselineExposure -0.5 EV. Every frame must open in LibRaw at the right size and pattern
  and pass Adobe `dng_validate`, as in v6. With `WRITE_DNG = False` (the default) the DNG is written to a
  temporary folder only to develop the TIFF, then deleted, and no DNG goes into the archives; the TIFFs and
  the MLV are identical either way.
- The TIFF manifest, `tiff/<clip>/<clip>_manifest.json`, holds each frame's time code and TIFF read-back
  result. With `WRITE_DNG = True` the DNG manifest, `dng/<clip>/<clip>_manifest.json`, adds each frame's seed,
  timings and LibRaw and dng_validate results.
- `timings.csv` records the seconds per frame for decode, RAW, write, LibRaw, dng_validate, develop, TIFF and
  MLV.
- The report estimates what a 4K UHD version would cost (see Notes).

### 3. Temporal consistency

- The illuminant, white balance and exposure are fixed once per clip, never per frame. A video is display
  referred, so its white is the sRGB (D65) white: the as-shot neutral is the camera's response to D65, not
  v6's still-photo illuminant prior, and render white is the sensor clip. `calibration.json` records them.
- Every frame gets fresh v6 shot and read noise from its own seed (the v6 noise seed plus the frame index), so
  a frame is reproducible but grain is never repeated between frames.

### 4. Proof

- Step 5 reads every developed TIFF (section 2) and compares it with its decoded source. It checks three
  ratios of developed to source: mean luma, R/G and B/G.
- The run is stable when:
  - every frame-to-frame step of each ratio is at most **0.5%**;
  - the clip-wide span of each ratio, (max - min) / median, is at most **2%**.
- Frames darker than linear mean luma 0.002 are reported but not scored. Fades and black frames have no
  colour to compare.
- The MLV must read back with the expected headers and frame count, and every frame bit-exact (section 2).
- `check/preview_side_by_side.mp4` shows the source beside the developed TIFFs at the source frame rate. The
  report shows it with first, middle and last frame thumbnails and a stability plot.
- The report also has the per-frame timings and the total wall time.
- **No accuracy claim.** There is no real RAW of the clip to score against. The v6 benchmark result, each
  colour plane within the 95% target against one real A1 II RAW, does not transfer to video frames.

### 5. Deliverables

- `<run>_01_summary.zip`: report.html, summary.txt, the verdict, the manifest, convert.json, check.json,
  timings.csv, the preview MP4, the plot and the thumbnails.
- Then one set of split archives per output, at most 500 MB each. NN is numbered on continuously in
  download order from 02 across all three sets, and K counts the parts of one set:
  - `<run>_NN_mlv_partKofN.zip`: the MLV clip, its chunks in play order, and its read-back check. Each chunk
    fits whole in one part; extract every part into one folder before opening the `.MLV`.
  - `<run>_NN_tiff_partKofN.zip`: the TIFFs in frame order, with their manifest.
  - `<run>_NN_dng_partKofN.zip`: the DNGs in frame order, with their manifest (only with `WRITE_DNG = True`).
- A run that ends early marks its last parts `_partial`.
- The source clip stays on the runtime. Only its small text records are packaged.

### 6. Runtime

- Colab A100 or L4, or CPU-only with a warning. Nothing in this notebook uses the GPU.
- Preflight (step 3) stops the run before any frame is converted if:
  - host RAM is under 8 GB;
  - ffmpeg or ffprobe is missing;
  - dng_validate was not built;
  - the clip cannot be probed or its first frame cannot be decoded;
  - free disk is short for the outputs (the TIFFs counted uncompressed, the DNGs only when written), their
    zip copies and a reserve.

## Layout

```text
notebooks/   the notebook (two code cells: settings, driver; step 1 input and step 2 setup run inline)
source.py    ffprobe metadata and the ffmpeg frame stream (planes as decoded, rotation, early stop)
ycbcr.py     decoded planes to linear BT.709: chroma siting, matrix, range, transfer, primaries
sequence.py  CinemaDNG-style names, SMPTE time codes and the frame DNG (the v6 tag set plus frame tags)
mlv.py       the MLV v2.0 writer (14-bit packed RGGB, chunks) and its Python reader and check
preflight.py step 3: runtime tier, tools, the clip (upload, URL, Drive or path), probe, free disk
convert.py   step 4: clip calibration, per-frame RAW, DNG, TIFF and MLV, validation, timings, the 4K estimate
check.py     step 5: the TIFFs against the source, stability check, side-by-side preview, plot
driver.py    steps 3-5 with gating, guarding and checkpoints
delivery.py  run state, verdict, split archives, Drive mirror, downloads and recovery (standard library only)
report.py    report.html and summary.txt
tests/       tests on tiny synthetic clips (a few 64x48 frames, no models and no real video)
```

## Tests

```sh
.venv/bin/python -m pytest                 # the whole project suite, video_v1/tests included
.venv/bin/python -m pytest video_v1/tests  # only these tests
```

A plain `pytest` run adds `video_v1/tests` through the root `conftest.py`, not through `testpaths` in
`pyproject.toml`. The v6 notebook test requires `pyproject.toml` to stay as it was at the v6 notebook's
`REPO_REF`. The project has no CI workflow, so a plain `pytest` run is where every test runs. The tests need
ffmpeg and ffprobe on the PATH (they skip otherwise), and they stand in a script for dng_validate, which is
built only on Colab. The notebook runs only on Colab.

## Notes

- **4K estimate.** The estimate is computed, not measured. Super-resolution uses the v6 run
  20261104T064058Z_ad151cd: on an A100, HAT-L x4 took 2386 s for 70 tiles of 512 px with the x8
  self-ensemble. A 4K version still decodes the source, so the decode stage (with the YCbCr to linear
  conversion) keeps this run's measured time. RAW build, DNG write, LibRaw, dng_validate, develop, TIFF and
  MLV scale with the UHD to source pixel ratio from this run's measured times. The report gives the cost
  with and without the ensemble.
- **Per-frame SR.** HAT-L has no temporal model, so a 4K version would need its own flicker check.
- **Why the first DNGs looked off in other apps.** Run 20261105T110627Z_c030e67 wrote DNGs whose data
  matched the source (the pipeline's own develop was within dE 0.4), but macOS Preview and Adobe rendered
  them with off colours and a soft, noisy look:
  - Colour: the DNGs were encoded the way LibRaw renders (white balance in camera space, then a fixed D65
    camera matrix) with a non-D65 as-shot neutral and only ColorMatrix1. A DNG-spec reader then adapts
    from the white that neutral implies, which shifts saturated colours. The video neutral is now the
    source's own D65 white, so ColorMatrix1 alone gives DNG-spec readers the colours LibRaw renders. This is
    a video_v1 calibration choice; the shared v6 DNG writer is unchanged.
  - Look: default raw rendering adds sharpening and a contrast tone curve that lifts mid-grey and the grain.
    BaselineExposure -0.5 EV brings that default rendering close to the source. The pixel data is
    unchanged; a raw editor can still set any rendering.
  - For a fixed rendering, use the TIFFs, which are the pipeline's own develop.
- **No PyPI MLV parser in the tests.** The only PyPI package named `mlv` has no description, author or
  project page, so it is not a maintained parser; the tests use the project's own reader, written from
  Magic Lantern's `mlv.h` and `raw.h`.
- **Known v6 limit carries over.** The 50 MP grain was cleaner than the real camera. The noise model is the
  same one at source resolution.
