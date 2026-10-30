# Synthetic 50 MP camera experiment - PRD v5

One new Colab notebook: [synthetic_50mp_v5.ipynb](notebooks/synthetic_50mp_v5.ipynb).
The supplied [PRD](PRD_v5.md) is unchanged. [EVALUATION.md](EVALUATION.md) records
source/licence checks, resource estimates, implementation coverage and deferrals.

Outputs describe one hypothetical synthetic exposure. They do not recover the
original camera RAW, sensor calibration, spectra or private metadata.

## Run in Colab

1. Open the notebook from this branch after validation publishes it. Select a GPU
   runtime. Setup clones the repository into `/content/experiment_50MP`, fetches
   `REPO_REF`, records its actual commit and installs the package plus SR extras.
   Pin `REPO_REF` to a full commit SHA for a reproducible release experiment.
2. Supply the rendered source variants at `SOURCE_2K` and `SOURCE_4K`. They must
   depict the same aligned scene; filenames alone cannot prove registration.
   The notebook defaults to a declared sRGB assumption for unprofiled images.
3. Review the full configuration and editable field catalogue. Defaults use
   full 50 MP, HAT requested, HYBRID requested, experimental analytic HDR,
   nonzero physical sensor/optics priors, and all outputs. The physical-effect
   defaults are modest hypotheses, not maximum noise or blur amplitudes.
4. HAT and Real-ESRGAN accept an `SR_CHECKPOINT` dictionary containing `path`,
   `sha256`, `backend`, full `repository_revision`, `source`, `licence`, and
   explicit `licence_accepted`. Bytes, RGB architecture and x4 scale are checked
   before loading. Actual inference is Colab/CUDA-only. No model is downloaded
   automatically. Without supplied weights, the allowed Lanczos/analytic fallback
   is prominently reported. Invalid supplied weights fail.
5. Run all in Colab. Hardware/RAM/disk admission precedes reconstruction. Two
   sources execute sequentially with the same immutable camera/settings/seeds.
   Outputs go under `OUTPUT/pair/2k` and `OUTPUT/pair/4k`. The final cell displays
   the validation and comparison reports and prints DNG/TIFF paths.

A100 40 GB / about 89 GB RAM is a plausible runtime, not a guarantee. Neural
inference is tiled; arrays are disk-backed. Reserve at least 30 GB of runtime
disk, more for retaining all intermediates and later editability sweeps. Runtime,
quality, checkpoints and maximum GPU/RAM use still require measurement in Colab.

## Configuration and scientific limits

`presets.full_quality_config()` provides the implemented full-scale preset.
`AnalyticRunConfig` defaults remain conservative for library users. Its strict
JSON serializer/parser and `analytic-config.schema.json` support replay. The
notebook also exposes every PRD parameter-table entry with field modes. Unsupported
estimators/overrides are rejected; a visible control does not claim implementation.

- Geometry supports aspect-preserving 50 MP and native 8640x5760 with explicit
  crop/margin policies. Unsupported required outpainting fails before allocation.
- ADC hypotheses support 12-16 bits in uint16 storage; Sony-like defaults are
  14 bits. Exposure/white balance/profile estimates retain provenance and
  uncertainty. Generic/custom synthetic identity and user D65 matrices are supported.
- Sensor simulation samples CFA first, then electrons, body/thermal/defect
  effects, stochastic shot/read noise, optional charge compression, gain and ADC.
  ISO-switched effective electronics require a supplied profile. They do not
  represent simultaneous dual-gain readout or measured Sony electronics.
- Optics are applied before CFA: distortion, vignetting, lateral/axial CA,
  spatial Gaussian PSF, approximate diffraction/defocus, and global line motion.
  Axial CA has no reconstructed depth field; PSFs are not measured lens MTF.
- A caller-loaded `ResidualAdapter` can add camera-bound confidence-gated learned
  residuals. RAW-Diffusion/SpiralDiff/RAW-Flow loaders, camera adaptation and halo
  inference remain deferred. Their absence is reported; strict learned mode
  fails without a compatible adapter. A supplied residual currently requires
  `RAW_TILE=1024`, `RAW_OVERLAP=0`; the default requested overlap of 128 is rejected
  when such an adapter is supplied, until its halo loader is implemented.
- Private identity/history remains unresolved. Experimental acceptance defaults
  do not grant licence acceptance. No GPS, serial or timestamp is invented.

## Outputs and interpretation

Each run contains canonical `synthetic_50mp.dng`, `synthetic_50mp_16bit.tif`,
`synthetic_50mp_preview.jpg`, clean/noisy Bayer NPY arrays, complete field registry,
camera/config/source/checkpoint JSON, uncertainty/generated-region PNG masks,
validation JSON/HTML, completion manifest and per-artifact hashes. Original
`capture.dng`, `developed.tiff`, config and mask names remain for API compatibility.
Canonical DNG/TIFF names use hard links where supported to avoid disk duplication.
The pair folder adds `pair_manifest.json` and `2k_vs_4k_report.html`.

Clean mosaic values are expected electrons before body/noise effects. Noisy values
are ADC-referred electrons after pedestal removal and include clipping/quantisation.
The generated-region mask marks potentially generated/unsupported content, not a
truth oracle. Original-source clipping uncertainty is independent of SR endpoints.

Independent LibRaw reopening checks pixel samples, CFA, levels and matrices; TIFF
reopening checks uint16 samples. Small flat/dark measurements assess only baseline
simulator consistency. Reports keep acceptance **conditional** until real RAW/HDR/
colour benchmarks, seam measurements, learned-SR versus Lanczos truth comparisons,
full editability sweeps and full-scale Colab execution provide evidence.

## Development

Do not execute this notebook, download models, run inference/training, or allocate
50 MP arrays on the 8 GB development Mac. Tiny unit fixtures and lint only.

```sh
python -m ruff check .
/Users/rohit/Projects/firstmate/data/exp50mp-heavy-lock.sh python -m pytest -q
```

The root pytest configuration includes legacy and v5 tests. New code is isolated
under `prd_v5`; existing notebooks and root PRDs are unchanged. Development-only
proof/logs/checkpoints belong outside Git. Never commit ARW files or large binaries.
