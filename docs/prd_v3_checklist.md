# PRD_v3 delivery checklist audit

This page maps every item of the PRD_v3 "final Claude Code delivery checklist" and every row of the toggle
acceptance table to its implementation and tests. Each claim was checked against the code on
`fm/exp50mp-v3-followups`. Modules are in `src/synthraw/`, tests in `tests/synthraw/`.

States:

- **Tested locally**: implemented and tested locally on tiny fixtures (no 50 MP arrays, no model inference).
- **Pending Colab**: implemented, but the part that matters can only be verified in Colab (full size, GPU,
  downloaded weights or tools Colab provides).
- **Gated stub**: an optional backend behind a licence gate. It is `UNPINNED`, warns about its licence, never
  downloads or imports a model, and falls back deterministically with the reason recorded
  (`synthraw.experimental`, `test_experimental.py`).
- **Not implemented**: with the reason.

The notebook "Run all" test (`test_notebook_run_all.py`) runs by default: it executes every cell after the pip
install on a 48 x 20 PNG with Colab mocked and FAST_DEV_RUN at 96 px. No 50 MP arrays are allocated on the
development laptop.

## Delivery checklist

| PRD item | State | Implementation | Tests |
|---|---|---|---|
| One notebook: `synthetic_camera_raw_50mp_colab.ipynb` | Tested locally | `notebooks/synthetic_camera_raw_50mp_colab.ipynb`; each section runs one `pipeline.PipelineRun` stage, the same code as `run_pipeline` | `test_notebook_run_all.py`, `test_notebook_structure.py`, `test_notebook_validation_cells.py` |
| Clean Colab `Runtime → Run all` | Pending Colab | Whole notebook; setup cell installs `REPO_REF` (default `main`, or a pinned 40-character SHA) with the `sr` extra (cell 2) | `test_notebook_run_all.py` (tiny PNG, Colab mocked); setup cell mocked in `test_notebook_structure.py` |
| 2K JPEG/PNG input | Tested locally (tiny PNG); full 2K pending Colab | `input_audit.InputInspector` | `test_input_audit.py`, `test_synthraw_pipeline.py` |
| 4K JPEG/PNG input | Tested locally (tiny PNG); full 4K pending Colab | `input_audit.InputInspector` | `test_input_audit.py`, `test_synthraw_pipeline.py` |
| α1 II-like 8640 × 5760 native mode | Tested locally (geometry arithmetic and tiny dev canvas); 50 MP allocation pending Colab | `geometry.TargetGeometryPlanner`, `camera_profile.CameraProfileManager` | `test_geometry.py`, `test_camera_profile.py`, `test_synthraw_pipeline.py` |
| 10,872 × 4,608 cinema 50.1 MP virtual mode | Tested locally (arithmetic and tiny dev canvas); 50 MP pending Colab | `geometry.TargetGeometryPlanner` | `test_geometry.py`, `test_synthraw_pipeline.py` |
| No aspect stretching | Tested locally | `geometry.TargetGeometryPlanner` (outpaint, crop or margins; policy recorded), `sensor_canvas.place_sensor_canvas` | `test_geometry.py`, `test_sr_run.py`, `test_sensor_canvas.py` |
| Fidelity SR backend | Pending Colab (fallback chain tested locally with mocks) | `super_resolution.SuperResolutionEngine`, `learned_sr`, `model_manifest` (SwinIR pinned by revision and sha256); 2K and 4K sources both run the pinned x4 checkpoint, then an exact resize | `test_synthraw_super_resolution.py`, `test_learned_sr.py`, `test_sr_run.py`, `test_sr_scale.py`, `test_model_manifest.py` |
| Real-ESRGAN fallback | Pending Colab (chain order tested locally) | `SuperResolutionEngine.reconstruct` chains SWINIR/HAT/SUPIR → REAL_ESRGAN → LANCZOS. A chosen pinned backend (SWINIR, REAL_ESRGAN) that cannot load, or any missing learned-SR package, raises `SRBackendLoadError` instead of falling back; only unpinned and gated backends fall back on a load failure | `test_synthraw_super_resolution.py`, `test_sr_run.py` |
| Optional SUPIR with licence warning | Gated stub | `experimental.request_experimental("SUPIR")`; SUPIR → SWINIR → REAL_ESRGAN → LANCZOS | `test_experimental.py` |
| Continuous float scene-linear representation | Tested locally | `hdr.HDRReconstructionEngine.linearize` (float32, inverse tone), `exporters.EXRWriter` | `test_hdr.py`, `test_synthraw_exporters.py` |
| HDR highlight/shadow synthesis masks | Tested locally | `HDRReconstructionEngine.expand` writes `highlight_fabricated_mask` and `shadow_fabricated_mask`; `validation.HDRValidator` | `test_hdr.py`, `test_validation.py`, `test_report.py` |
| Optional ExpandDiff backend | Gated stub | `experimental.request_experimental("EXPANDDIFF")`, used by `HDRReconstructionEngine` (also `VID2HDR`) | `test_experimental.py`, `test_hdr.py` |
| Brooks-style inverse ISP | Tested locally | `camera_colour.InverseISPEngine` (matrix, inverse WB, inverse digital gain) | `test_camera_colour.py` |
| Optional RAW-Flow/SpiralDiff | Gated stub | `USE_RAW_FLOW`, `USE_SPIRALDIFF` → `request_experimental` in `InverseISPEngine.reconstruct`; PHYSICS kept | `test_experimental.py`, `test_camera_colour.py` |
| Camera colour matrices | Tested locally | `camera_colour.CameraColourModel` (ColorMatrix1/2, ForwardMatrix), `validation.ColourValidator` | `test_camera_colour.py`, `test_validation.py` |
| WB inference/manual override | Tested locally | `white_balance.WhiteBalanceEstimator` (`manual_cct`, `manual_tint`, `user_override` recorded) | `test_white_balance.py` |
| Exposure posterior rather than fake certainty | Tested locally | `exposure.ExposurePosteriorEstimator` (candidate set with weights) | `test_exposure.py` |
| Optional geometry/lens inference | Tested locally | `camera_geometry.CameraGeometryEstimator` (FOV hypothesis, never lens identification) | `test_camera_geometry.py` |
| Expected-electron representation | Tested locally | `sensor.ElectronSignalSimulator`; `exposure.exposure_and_gain` places the brightest reconstructed HDR value 0.5 EV below full well, records the push as BaselineExposure and sets the analogue gain from the posterior ISO | `test_sensor.py`, `test_synthraw_pipeline.py` |
| Shot noise | Tested locally | `sensor.SensorNoiseSimulator` (`SHOT_NOISE`) | `test_sensor.py`, `test_validation.py` |
| Read noise | Tested locally | `sensor.SensorNoiseSimulator` (`READ_NOISE`) | `test_sensor.py` |
| PRNU/DSNU/FPN toggles | Tested locally | `PRNU`, `DSNU`, `ROW_FPN`, `COLUMN_FPN`, `DARK_CURRENT`, `HOT_PIXELS`, `DEAD_PIXELS` in `SensorNoiseSimulator` | `test_sensor.py` |
| Persistent virtual sensor-body map | Tested locally | Body pattern seeded by `SENSOR_BODY_SEED`, separate from `SENSOR_NOISE_SEED`; `validation.SensorSeedValidator` | `test_sensor.py`, `test_seed_metadata_validation.py` |
| Real CFA mosaic | Tested locally | `cfa.CFAMosaicer` (RGGB), `validation.CFAValidator` | `test_cfa.py`, `test_validation.py` |
| 14-bit Sony-like ADC | Tested locally | `adc.ADCSimulator`, `validation.ADCContractValidator` | `test_adc.py`, `test_validation.py` |
| Experimental 16-bit ADC | Tested locally | `ADC_BITS=16` or `EXPERIMENTAL_16BIT_SENSOR` → `VIRTUAL_16BIT_SENSOR` profile (`config`, `camera_profile`) | `test_synthraw_config.py`, `test_camera_profile.py`, `test_adc.py` |
| DNG stored compatibly as `uint16` | Tested locally | `dng_writer.DNGWriter` (16-bit storage; uncompressed strips by default, or lossless JPEG tiles with `DNG_COMPRESSION = "LOSSLESS"`) | `test_dng_writer.py`, `test_raw_gate.py`, `test_synthraw_pipeline.py` |
| Standards-based DNG metadata | Tested locally | `dng_metadata.DNGMetadataBuilder` (`Make = "Synthetic"`, NoiseProfile, matrices) | `test_dng_metadata.py` |
| Dynamic ExifTool Sony metadata registry | Tested locally (parser, fail-loud catalogue errors, pinned install with the download mocked); the real download pending Colab | The notebook setup cell calls `sony_registry.install_exiftool` on Colab: it downloads the pinned ExifTool release (`EXIFTOOL_VERSION`), verifies `EXIFTOOL_SHA256` before unpacking and runs it with the system Perl. `sony_registry.SonyMetadataRegistry` parses `exiftool -listx` and records the ExifTool version. Only an absent ExifTool, including a failed download or hash check, returns the bundled seed with `seed_only: true` and the reason in `fallback_reason`, and the run lists the fallback. An installed ExifTool that fails (`-ver` or `-listx` error, timeout, bad XML, no Sony fields) raises | `test_sony_registry.py`, `test_notebook_structure.py`, `test_synthraw_pipeline.py` |
| `sony_arw_emulation.json` | Tested locally | `sony_emulation.SonyARWEmulationBuilder`; `capture_inputs` fills the exposure, shutter and WB families from the run's records | `test_sony_emulation.py`, `test_synthraw_pipeline.py` |
| No fake proprietary ARW requirement | Tested locally | ARW emulation is JSON only; no MakerNote blocks, no realistic Sony serials | `test_sony_emulation.py`, `test_dng_metadata.py` |
| Independent LibRaw reopen | Tested locally | `libraw_validator.LibRawValidator` (hard gate, both DNG compressions and every noise realisation), `validation.libraw_reopen_check`, `validation.sensor_realisations_check` | `test_raw_gate.py`, `test_dng_writer.py`, `test_synthraw_pipeline.py`, `test_validation_runner.py` |
| Multiple demosaic tests | Tested locally | `validation.DemosaicValidator` (LibRaw demosaic variants; AHD and the additional algorithms are hard gates) | `test_validation.py` |
| ±4 EV stress test | Tested locally | `validation.EditabilityValidator` (-4..+4 EV, hard gate) | `test_validation.py` |
| WB stress test | Tested locally | `validation.EditabilityValidator` (cool, neutral, warm via LibRaw `user_wb`, hard gate) | `test_validation.py` |
| NoiseProfile consistency test | Tested locally | `validation.NoiseValidator` (flat-field mean-variance fit against the NoiseProfile at the applied analogue gain; hard gate, R² stays a QA target) | `test_validation.py`, `test_validation_runner.py`, `test_seed_metadata_validation.py` |
| 16-bit 50 MP TIFF | Tested locally (tiny); 50 MP pending Colab | `exporters.TIFFWriter` (LibRaw development, Adobe RGB) | `test_synthraw_exporters.py`, `test_synthraw_pipeline.py` |
| Optional scene-linear EXR | Tested locally | `exporters.EXRWriter` (`WRITE_EXR`) | `test_synthraw_exporters.py` |
| Provenance for every field | Tested locally | `provenance.ProvenanceWriter`, `FieldRecord` classes D/I/F/M; `validation.MetadataDependencyValidator` | `test_provenance.py`, `test_seed_metadata_validation.py` |
| 2K-versus-4K report | Tested locally (tiny); full resolution pending Colab | `comparison.SourceResolutionComparator`, `ComparisonReportGenerator` | `test_synthraw_comparison.py`, `test_notebook_validation_cells.py` |
| Synthetic-capture disclosure | Tested locally | DNG `ImageDescription`, `metadata.json` disclaimer, report disclosure banner | `test_synthraw_pipeline.py`, `test_dng_metadata.py`, `test_report.py` |
| Optional C2PA declaration | Tested locally (mocked signer); real signing needs `c2pa-python` and a signer | `provenance.write_c2pa`; records "C2PA unavailable" otherwise | `test_synthraw_pipeline.py` |
| Model licence/checksum manifest | Tested locally | `model_manifest.write_model_manifest` (`logs/model_manifest.json`, `UNPINNED` entries) | `test_model_manifest.py`, `test_experimental.py` |
| HTML validation report | Tested locally | `report.ReportGenerator`, `validation_runner.ValidationRunner`, `figures.ValidationFigureWriter` | `test_report.py`, `test_validation_runner.py`, `test_figures.py` |

## Toggle acceptance table

| Toggle | State | Acceptance evidence | Tests |
|---|---|---|---|
| `SONY_A1II_NATIVE` | Tested locally | 8640 × 5760 plan with the outpaint, crop or margin policy recorded in `geometry_plan.json` | `test_geometry.py`, `test_sr_run.py` |
| `CINEMA_50MP_VIRTUAL` | Tested locally | `config` forces `TARGET_PROFILE = "CINEMA_50MP_VIRTUAL"`; profile names say "Virtual Cinema" | `test_synthraw_config.py`, `test_camera_profile.py`, `test_dng_metadata.py` |
| `SR_BACKEND` | Pending Colab (learned backends); Lanczos tested locally | `SuperResolutionEngine` with `diagnose_source_fidelity` measuring the source-scale downsample | `test_synthraw_super_resolution.py`, `test_sr_run.py` |
| `GENERATIVE_DETAIL_STRENGTH` | Not implemented | No generative detail backend exists (SUPIR is a gated stub). A value above 0 is recorded as an SR fallback saying it was ignored. `generated_detail_mask.npy` is still written and marks learned-SR content only; outpainted pixels are in `outpaint_mask.npy` | `test_sr_run.py`, `test_sensor_canvas.py` |
| `HDR_ENABLED` | Tested locally | Created radiance tracked by `highlight_fabricated_mask` and `shadow_fabricated_mask`; `HDRValidator` | `test_hdr.py`, `test_validation.py` |
| `HDR_*_EV` | Tested locally | Nonnegative EV enforced in `config`; `NumericHealthValidator` rejects negative/NaN radiance | `test_synthraw_config.py`, `test_hdr.py`, `test_validation.py` |
| `USE_RAW_FLOW` | Gated stub | Falls back to PHYSICS; camera domain labelled "cross-camera prior; not a Sony alpha 1 II model" | `test_experimental.py` |
| `SPECTRAL_MODE` | Tested locally | `CALIBRATED_CURVES` raises without `CALIBRATION_ASSETS` | `test_synthraw_config.py` |
| `ADC_BITS=14` | Tested locally | `ADCSimulator` clips to 16383; `ADCContractValidator` is a hard gate | `test_adc.py`, `test_validation.py` |
| `EXPERIMENTAL_16BIT_SENSOR` | Tested locally | Profile renamed `VIRTUAL_16BIT_SENSOR` automatically | `test_synthraw_config.py`, `test_camera_profile.py` |
| `SHOT_NOISE` | Tested locally | Mean-variance slope fitted by `NoiseValidator` | `test_sensor.py`, `test_validation.py` |
| `READ_NOISE` | Tested locally | Dark-region variance with and without read noise | `test_sensor.py`, `test_adc.py` |
| `PRNU/DSNU/FPN` | Tested locally | Same `SENSOR_BODY_SEED` reproduces the pattern across noise seeds; `SensorSeedValidator` | `test_sensor.py`, `test_seed_metadata_validation.py` |
| `OPTICAL_SIMULATION` | Tested locally | `LensSimulator` runs in the `optics` stage before `cfa` (pipeline `STAGES`), never after demosaic | `test_optics.py` |
| `FABRICATE_UNOBSERVABLE_EXIF` | Tested locally | Temperature and battery fields need the toggle and must be class F. Stricter than the PRD: no serial or shutter count is ever written, even in a synthetic namespace; they stay null with the reason recorded | `test_sony_emulation.py` |
| `WRITE_DNG` | Tested locally | Reopened by LibRaw as a hard gate; `PipelineRun.create` (notebook and `run_pipeline`) refuses `WRITE_DNG=False` | `test_raw_gate.py`, `test_synthraw_pipeline.py` |
| `WRITE_C2PA` | Tested locally (mocked signer) | `digitalSourceType` declared when `c2pa-python` and a signer are available, otherwise "C2PA unavailable" is recorded | `test_synthraw_pipeline.py` |

## Experimental options

`SUPIR`, `EXPANDDIFF`, `VID2HDR`, `RAW_FLOW`, `SPIRALDIFF` and `INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE` are
gated stubs in `synthraw.experimental`, each tested in `test_experimental.py`. None is pinned, so none can
download or load a model. Pinning one needs a repository revision, a sha256 and a licence review in
`model_manifest.py`.

## Other configuration fields

These `ExperimentConfig` fields are not in the checklist or the toggle table:

| Field | State | Implementation | Tests |
|---|---|---|---|
| `DNG_COMPRESSION` | Tested locally | PRD optional validated lossless mode. `LOSSLESS` writes lossless JPEG tiles (`Compression = 7`, at most 256 x 256, never wider or taller than the image, which LibRaw needs). `DNGWriter` decodes every tile again, and the LibRaw gate requires pixels identical to the uncompressed mosaic. `UNCOMPRESSED` stays the default. The RAW gate cell tests both | `test_dng_writer.py`, `test_synthraw_pipeline.py`, `test_notebook_structure.py`, `test_synthraw_config.py` |
| `N_SENSOR_REALISATIONS` | Tested locally | `sensor_realisations` pipeline stage: realisation 1 is the primary DNG with `SENSOR_NOISE_SEED`; realisations 2..N rerun noise, CFA and ADC with seeds from `sensor.realisation_noise_seeds`, each a LibRaw-validated DNG in `raw/`. Seeds go to `validation/sensor_realisations.json`, `run_report.json` and the sensor profile; `validation.sensor_realisations_check` is a hard gate | `test_sensor.py`, `test_synthraw_pipeline.py`, `test_validation_runner.py` |
| `GENERATE_SYNTHETIC_EXIF` | Tested locally | `False` keeps inferred and fabricated capture EXIF out of the DNG (`DNGMetadataBuilder` rejects it) and the Sony sidecar (fields stay empty with the reason). User-supplied values are still written; `FABRICATE_UNOBSERVABLE_EXIF` requires it | `test_dng_metadata.py`, `test_sony_emulation.py`, `test_synthraw_pipeline.py`, `test_synthraw_config.py` |
