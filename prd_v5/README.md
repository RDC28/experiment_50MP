# Synthetic 50 MP camera experiment - PRD v5

One new Colab notebook: [synthetic_50mp_v5.ipynb](notebooks/synthetic_50mp_v5.ipynb).
The supplied [PRD](PRD_v5.md) is unchanged. [EVALUATION.md](EVALUATION.md) records
source/licence checks, resource estimates, implementation coverage and deferrals.

Outputs describe one hypothetical synthetic exposure. They do not recover the
original camera RAW, sensor calibration, spectra or private metadata.

## Run in Colab

1. Open the notebook from `main`. Select a GPU
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
5. For calibrated colour measurements, set `COLOUR_REFERENCES` using the example
   in the settings cell. Supply independent native-output-size uint16 sRGB TIFFs
   for `"2k"` and/or `"4k"` and actual camera/profile, trusted-development and
   alignment evidence. The empty default requests reporting without inventing
   truth: absent references stay not measured. `COMPARISON_REGION` also requests
   native-pixel reference crop measurements. Allow disk for supplied references.
6. Run all in Colab. Hardware/RAM/disk admission precedes reconstruction. Two
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
  residuals with bounded halo context and core-only writes. `RAW_TILE=1024`
  controls core size and `RAW_OVERLAP=128` supplies context on each side in the
  notebook. Library adapters expose `tile` and `halo` with those same defaults.
  The predictor receives the context window origin and owns physical-edge
  padding; its receptive field must fit the halo. No model seam guarantee is
  implied. RAW-Diffusion/SpiralDiff/RAW-Flow loaders and camera adaptation remain
  deferred (see EVALUATION.md for the verified artifact status of each). Their absence is reported; strict learned mode fails without a
  compatible adapter.
- Private identity/history remains unresolved. Experimental acceptance defaults
  do not grant licence acceptance. No GPS, serial or timestamp is invented.

## Outputs and interpretation

Each run contains canonical `synthetic_50mp.dng`, `synthetic_50mp_16bit.tif`,
`synthetic_50mp_preview.jpg`, clean/noisy Bayer NPY arrays, complete field registry,
camera/config/source/checkpoint JSON, uncertainty/generated-region PNG masks,
validation JSON/HTML, completion manifest and per-artifact hashes. Original
`capture.dng`, `developed.tiff`, config and mask names remain for API compatibility.
Canonical DNG/TIFF names use hard links where supported to avoid disk duplication.
The pair folder adds `pair_manifest.json` and `2k_vs_4k_report.html`. Both include automatic full-resolution pre/post-noise RAW disagreement in electron-equivalent units, with separate CFA-phase metrics and optional native-pixel crops. These unaligned comparisons do not establish scene registration or ground-truth accuracy.

Render comparisons also report five native-pixel frequency bands, including each
render's power and spectral difference power. FFTs use at most 256x256 pixels,
with mean removal and Hann windows. Inspect measured/omitted pixel counts for
small crops and edge slivers. These encoded-sRGB luma spectra measure disagreement;
they do not establish MTF, correct texture or learned-SR improvement against truth.

Clean mosaic values are expected electrons before body/noise effects. Noisy values
are ADC-referred electrons after pedestal removal and include clipping/quantisation.
The generated-region mask marks potentially generated/unsupported content, not a
truth oracle. Original-source clipping uncertainty is independent of SR endpoints.
The automatic HDR gate counts original-source endpoints and unsupported pixels.
Its claim-policy audit fails when analytic HDR expansion or the learned HDRUNet residual (`hdr_claims.npy`) claims radiance there, passes otherwise, and is `not measured` when neither ran, while numerical
HDR fidelity remains `not measured`. On Colab, inspect `validation.json` under
`HDR validity` for these counts. For external benchmark assertions, call
`hdr_validation.validate_hdr_claims(source_clipped, source_support, claimed_truth)`
with aligned boolean masks; any claimed pixel at an endpoint or unsupported
margin fails the gate. A claim elsewhere does not establish HDR truth.

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

The Colab bundle's Metadata coherence gate now compares exported DNG tags with the manifest and complete registry, including AsShotNeutral, exposure aliases and matrix groups. Inspect `checked`, `mismatches` and `unmapped_registry_fields` in `validation.json`: passing covers emitted standard metadata, not physical calibration or all simulator dependencies. Scientific truth gates remain conditional. Small local fixtures for this path: `.venv/bin/python -m pytest prd_v5/tests/test_coherence.py prd_v5/tests/test_delivery.py -q` (no model execution).

Generated-DNG exposure edit fixture/API:

```python
from prd_v5.roundtrip import develop_exposure_sweep

edits = develop_exposure_sweep(
    "capture.dng",
    "exposure_edits",  # must be a new directory
    baseline_exposure_ev=0.0,  # use the run's isp.baseline_exposure_ev
    max_working_bytes=8 * 1024**3,  # explicitly admit full-size Colab development
)
```

This sequentially exports full-resolution linear-light uint16 TIFFs at
-3/-2/0/+2/+3 EV using camera WB, linear demosaicing and fixed highlight clipping.
The JSON-safe return value records brightness multipliers, mean code and
zero/saturation counts. These are developer response measurements, not recovered
highlight truth or physical dynamic range. Files have linear transfer, unlike
the ordinary sRGB-transfer `developed.tiff`. Allow approximately 1.5 GB extra
storage per 50 MP capture (3 GB for the pair). The maximum preset and notebook
request these edits automatically via `exposure_sweep=True` and
`ENABLE_EXPOSURE_SWEEP=True`. Conservative library defaults disable them. Each
child bundle retains `exposure_edits/`, hashes all five TIFFs, and includes the
response in its manifest and RAW editability validation evidence. A failed sweep
prevents completion. Full-size visual
review remain pending; RAW editability remains not measured. Test without models using
`.venv/bin/python -m pytest prd_v5/tests/test_roundtrip.py -q`.

For a relative white-balance edit sweep on Colab (also invoked automatically by the maximum notebook):

```python
from prd_v5.roundtrip import develop_wb_sweep

wb_edits = develop_wb_sweep(
    dng_path,
    output_directory / "wb_edits",
    baseline_exposure_ev=baseline_exposure_ev,
    strength=1.25,
    max_working_bytes=8 * 1024**3,
)
```

This sequentially exports cool/neutral/warm linear-light uint16 TIFFs. Neutral
uses green-normalized camera WB; cool divides red and multiplies blue by
`strength`, warm reverses those changes. LibRaw normalizes the gains, so relative
channel ratios describe the change rather than absolute brightness scaling.
These are relative developer settings, not calibrated Kelvin temperatures.
The fresh destination is protected from overwrite and removed on failure.
Allow another 1.8 GB disk for both 50 MP inputs. WB accuracy stays not measured;
highlight/shadow sweeps and full-size visual review remain pending.
The maximum preset enables `wb_sweep=True`; notebook `ENABLE_WB_SWEEP=True`
and `WB_SWEEP_STRENGTH=1.25` expose the controls. Conservative library defaults
disable the sweep. Both child bundles retain `wb_edits/`, per-TIFF hashes and
WB response in manifest, pair and JSON/HTML validation reports. A failed sweep
prevents completion. The existing development memory budget admits each render;
RAW editability remains not measured.

Generated-DNG demosaic sweep API (Colab; also invoked by the maximum notebook):

```python
from prd_v5.roundtrip import develop_demosaic_sweep

edits = develop_demosaic_sweep(
    dng_path,
    output_directory / "demosaic_edits",  # fresh directory required
    baseline_exposure_ev=baseline_exposure_ev,
    max_working_bytes=8 * 1024**3,
)
```

This sequentially exports LINEAR, VNG and AHD linear-light uint16 TIFFs with
identical camera WB, brightness and highlight clipping. Unavailable algorithms
are listed with reasons; execution/write errors abort and remove the new sweep
directory. Reports retain algorithm names, library versions, channel means and
clipping counts. Allow another 1.8 GB disk for two 50 MP inputs. Algorithm
response does not prove detail accuracy or visual editability. Highlight/shadow
sweeps and full-size visual review remain pending. Tiny fixtures:
`.venv/bin/python -m pytest prd_v5/tests/test_roundtrip.py -q`.

The maximum preset enables `demosaic_sweep=True`; notebook
`ENABLE_DEMOSAIC_SWEEP=True` exposes the control. Conservative library defaults
disable it. Both child bundles retain `demosaic_edits/`, per-TIFF SHA-256 and
response evidence in child manifests, paired summaries and JSON/HTML validation.
The ISP-derived BaselineExposure and existing development memory budget apply.
Unavailable methods retain precise skips; a runtime failure prevents completion.
RAW editability remains not measured pending visual acceptance and truth data.

Generated-DNG highlight/shadow sweeps (Colab; automatic in the maximum notebook):

```python
from prd_v5.roundtrip import develop_tone_sweep

edits = develop_tone_sweep(
    "outputs/synthetic_50mp.dng",
    "outputs/tone_edits",
    baseline_exposure_ev=0.0,  # use the actual ISP BaselineExposure
    shadow_lifts_ev=(0.5, 2.0),
    max_working_bytes=8 * 1024**3,
)
```

This exports four sequential uint16 TIFFs. LibRaw highlight reconstruction levels
3 and 9 favor whites and colors respectively, as described in the
[official rawpy documentation](https://letmaik.github.io/rawpy/api/enums.html#rawpy.HighlightMode).
They are low/high numbered settings, not calibrated recovery strengths. Shadow
edits use explicit post-development channel curves
`y=g*x/(1+(g-1)*x)`, with `g=2**EV` and normalized linear-light RGB `x`.
The curve preserves black and white endpoints, increases continuously with EV,
and changes color ratios. It operates after uint16 development and cannot
restore information already clipped or quantized. Camera WB, linear demosaic
and baseline brightness are held fixed; shadow renders use highlight clipping.

Header-only memory admission precedes pixel unpacking. The destination must be
new; failures remove partial sweep outputs. Evidence includes exact settings,
means, endpoint counts, paths and library versions. Highlight/shadow truth,
physical dynamic range and visual editability remain not measured. Allow about
1.2 GB extra disk per 50 MP DNG. Fixture check:
`.venv/bin/python -m pytest prd_v5/tests/test_roundtrip.py -q`.

The maximum preset enables `tone_sweep=True`. Notebook `ENABLE_TONE_SWEEP=True`
and `SHADOW_LIFTS_EV=(0.5, 2.0)` expose the switch and increasing shadow lifts
(up to 4 EV). Both generated DNGs receive four sequential edits using the actual
ISP BaselineExposure and development memory budget. Allow another 2.4 GB disk
for the pair. Child manifests, paired summaries and JSON/HTML validation retain
response evidence and hashes for each TIFF. Disabled runs record that state;
runtime failures prevent completion. Recovery accuracy and RAW editability stay
not measured pending aligned truth and full-size visual acceptance.

Calibrated colour-reference API: `prd_v5.colour_metrics.calibrated_delta_e`
accepts a candidate and supplied aligned reference as uncompressed, memory-mappable
uint16 RGB TIFFs. Pass `calibration` with exactly `camera_profile`,
`trusted_development`, `alignment` (nonempty evidence descriptions), and
`colour_space="sRGB_D65_2deg"`. Both images must already have trusted sRGB
colour conversion; arbitrary camera RGB, linear RGB and HDR are unsupported.
The API computes native-pixel CIEDE2000 mean/RMS/maximum in bounded row strips,
optionally over `region=(top,left,bottom,right)`. It performs no alignment or
exposure fitting. Missing reference/calibration stays not measured; evidence is
caller-attested rather than independently verified. `run_analytic_pair` now accepts optional `colour_references`, keyed by `"2k"`
and/or `"4k"`, each containing `{"path": reference_tiff, "calibration": contract}`.
The pair JSON/HTML records independent whole-raster and optional
`comparison_region` crop measurements for each generated `developed.tiff`, with
reference path, byte count and streamed SHA-256. Omitted references remain
explicitly not measured. References must match each output's native geometry;
no reference is implicitly shared between sources. An unreadable or invalid
reference prevents pair completion. These metrics do not set a scientific
acceptance threshold. Notebook `COLOUR_REFERENCES` passes these records directly to the paired API.
Child bundle Colour gates are populated from these references, as described below.


Calibrated references also populate each child bundle's Colour gate in
`validation.json`, its manifest and HTML reports, including the reference hash.
Single-source callers use `config.run(..., colour_reference={"path": reference,
"calibration": contract})`. Whole-raster measurement does not imply an acceptance
pass or WB accuracy; overall scientific acceptance stays conditional. References
remain external caller inputs and are not stored in replay configuration.

For explicitly measured edge profiles, `prd_v5.mtf.edge_mtf(profile,
sample_spacing_px=1.0, evidence={"edge": "target and extraction procedure",
"linearization": "linear-light conversion procedure"})` reports sampled MTF50
and MTF10 in cycles per native pixel. Supply 16-65,536 uniformly spaced linear
samples normal to an isolated straight edge, including both plateaus. Oversampled
profiles must declare their spacing in native pixels. Missing profiles/evidence,
flat edges and absent threshold crossings are explicit unavailable measurements.
This API does not select scene edges, establish calibrated lens MTF or claim
ISO slanted-edge compliance. The notebook exposes `MTF_EDGE` (default `None`, not measured).

For a selected axis-aligned edge in a developed uint16 sRGB TIFF, call
`prd_v5.mtf.tiff_edge_mtf(path, region=(top, left, bottom, right),
normal_axis="x", edge_evidence="isolated vertical edge with both plateaus")`.
Use `normal_axis="y"` for a horizontal edge. The API decodes sRGB before
luminance averaging, reads one bounded row at a time and limits the region to
1,048,576 pixels. Missing selection/evidence and flat edges stay not measured.
Selection is caller-attested; slanted edges, scene texture, sharpening and noise
can bias this sampled response. It is not calibrated optical MTF or recovered
detail truth. Pair reporting is available as described below, and the notebook exposes `MTF_EDGE`.


`run_analytic_pair(..., mtf_edge={"region": (top, left, bottom, right),
"normal_axis": "x", "edge_evidence": "isolated vertical edge, both plateaus"})`
measures each developed TIFF independently at the same native-pixel coordinates.
Pair JSON/HTML records `mtf_comparison` with sampled MTF50/MTF10, the response
curve and extraction evidence. Omit the selection or evidence to retain explicit
not-measured results. The caller must establish edge suitability in each output;
shared coordinates do not certify registration. This is sampled response, not
an acceptance pass or evidence of improved scene detail. Invalid selections or
extraction failures prevent publication of pair completion. Controls remain
external inputs rather than replay configuration; the notebook passes `MTF_EDGE` to the pair run.

## How to test on Colab

Run these only on a Colab A100 runtime, never on the development Mac.

1. Run the setup cell. It installs the repository with the `sr` and `metrics`
   extras (the `metrics` extra pins `lpips==0.1.4`). It also builds Adobe
   `dng_validate` on Linux (verified in an Ubuntu 22.04 container, not yet on
   Colab; if it fails the DNG conformance gate is a visible skip naming the error). A prebuilt binary can be
   given through `DNG_VALIDATE_PATH` (see EVALUATION.md).
2. Maximum defaults request every stage: full 50 MP, HAT, HYBRID, HDR, sensor and
   optics priors, all outputs, and the exposure (-3/-2/0/+2/+3 EV), WB,
   highlight/shadow and demosaic edit sweeps.
3. Set `COLOUR_REFERENCES` to independent aligned uint16 sRGB TIFFs plus evidence.
   Calibrated CIEDE2000 and LPIPS (`LPIPS_OPTIONS`: patch 64-1024, `alex`, `cuda`)
   use them. With no reference both stay `not measured`, never passed.
4. Set `MTF_EDGE` to a native-pixel region and edge evidence to measure sampled MTF.
   Leave it `None` to keep MTF `not measured`.
5. To use a HDRUNet residual, set the notebook `HDRUNET_CHECKPOINT`, `HDRUNET_SHA256`,
   `HDRUNET_LICENCE_ACCEPTED`, `HDRUNET_EXPERIMENTAL_ACCEPTED` and `HDRUNET_MAX_DELTA`
   (with `ENABLE_LEARNED_RGB2RAW` on); the notebook calls
   `prd_v5.hdrunet.load_hdrunet_adapter(path, expected_sha256, camera,
   licence_accepted=True, experimental_accepted=True)` and uses the result as
   `RESIDUAL_ADAPTER`. It verifies the digest before unpickling, loads with
   `weights_only=True` and strict keys, and pads tiles to multiples of four. Its unbounded
   gamma-2.24 output is linearized into a source-clip-relative residual, mapped into clean
   camera units by the run's analytic model (HDR headroom, matrix, gains) and gated by a
   change-size confidence;
   it is NTIRE2021 HDR, not Sony calibration or clipped-highlight truth.
6. To use learned weights, supply `SR_CHECKPOINT` (path, expected SHA-256, backend,
   revision, source, licence, licence acceptance). No checkpoint is downloaded
   automatically: no public artifact with a verified byte SHA-256 was found for
   HAT, RAW-Flow or HDRUNet. RAW-Diffusion and SpiralDiff publish no usable
   checkpoint. Without weights the Lanczos/analytic fallback is reported.
6. Inspect `validation.json`, the pair JSON/HTML and the edit-sweep folders. The
   RAW pre/post-noise comparison, frequency bands, metadata coherence and HDR
   validity gate run automatically.
