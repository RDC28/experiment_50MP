# Synthetic 50 MP RAW - PRD v6

One Colab notebook: [synthetic_50mp_v6.ipynb](notebooks/synthetic_50mp_v6.ipynb), specified by
[PRD_v6.md](PRD_v6.md). It turns rendered images of one real Sony A1 II frame into three synthetic RAW
outputs (2K, 4K and 50 MP, each a 16-bit DNG plus a 16-bit TIFF that Photoshop opens) and scores each one
against that frame's real RAW in linear RAW space. Each output passes only when every colour plane reaches
95% (PRD section 6).

## Run in Colab

1. Open the notebook from GitHub in Colab and choose **Runtime > Change runtime type > A100 GPU** with
   **High-RAM**. T4 and CPU runtimes are rejected at preflight; L4 runs with a warning.
2. **Runtime > Run all**. The defaults are the maximum-quality run and need no edits. A run takes about
   2.5 to 4.5 hours on an A100.
3. When the run ends, also after a failure, the browser downloads the archives: `<run>_01_summary.zip`
   (verdict, report, logs and JSON), then one archive per output folder (`_02_benchmark_2k` to
   `_04_benchmark_50mp`). Allow multiple downloads if Chrome asks. If a download was blocked or the kernel
   crashed, paste the recovery line that the driver cell printed first into a new cell.

`SOURCE_4K_URL` and `SOURCE_2K_URL` add 2K, 4K and 50 MP outputs of your own images, given as a direct URL or a Google Drive share link (file shared as "Anyone with the link"; no accuracy claim,
because there is no reference RAW for them). A source that fails to download or decode is skipped and reported
next to the verdict; it never stops the benchmark. `DRIVE_MIRROR = True` mirrors the summary to Google Drive after
every step.

## Layout

```text
PRD_v6.md          specification
notebooks/         the notebook (two code cells: settings, driver)
install.py         step 2: pinned install without a restart, dng_validate build (standard library only)
preflight.py       step 3: runtime tier, verified downloads of the reference ARW and HAT-L
benchmark.py       step 4: references R1-R3, ladder, the three outputs, gates and verdict
production.py      step 5: the frozen pipeline on your own sources
sweeps.py          step 6: edit sweeps, noise gate, render-space scores (reported, not gated)
driver.py          steps 3-6 with gating, guarding and checkpoints
delivery.py        run state, archives, downloads and recovery (standard library only)
report.py          report.html and summary.txt (standard library only)
pipeline.py, sr.py, colour.py, dng.py, metrics.py, reference.py, resample.py, constants.py
tests/             tiny-scale tests of the whole flow with a bicubic stand-in for HAT-L
```

The tests run locally with `python -m pytest prd_v6/tests`; the notebook itself runs only in Colab.
