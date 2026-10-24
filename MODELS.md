# Notebook 01 inference artifacts

## Phase 2 super-resolution weights

Notebook 01 defaults to `swinir`; `realesrgan` is an alternative learned backend.
Both use the shared Phase 2 registry: conservative selects fidelity weights and
balanced selects perceptual weights. Conservative feeds final reconstruction;
balanced remains a comparison candidate. Native x4 inference is resampled with
Lanczos-3 to the target. These outputs are Inference and synthesize detail.

[docs/models.md](docs/models.md) is authoritative for the selected SR weight files,
SHA-256 checksums, licences and commercial-use caveats. Runs record model descriptions,
weight metadata, tiling and seam checks in their manifests and reports.
The explicit `lanczos_stand_in` backend uses no SR weights and is Interpolation.
Missing learned dependencies or weights fail the SR stage rather than silently
substituting interpolation.

## OpenCV frontal Haar face detector

- Operation: inference only, labelled Inference. No training, fine-tuning or model download.
- Artifact: `cv2.data.haarcascades/haarcascade_frontalface_default.xml`, bundled in some OpenCV Python wheels.
- Historical XML verification: `opencv-python-headless==4.13.0.92` (OpenCV runtime `4.13.0`). The project currently pins `opencv-python-headless==5.0.0.93`, whose wheel omits this XML.
- Verified XML SHA-256: `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0`.
- Every run records the actual installed opencv-python/headless package versions, OpenCV runtime version, XML filename, path and SHA-256 in `manifest.json`'s `models` list. `face_detector: none` uses no model.
- License: Intel Open Source Computer Vision Library license, BSD-3-style. The full authoritative license text is in the XML's opening comment, immediately before `<opencv_storage>`. The package's own license is separate from this artifact's license.
- Commercial use and redistribution are permitted, including modifications, subject to retaining copyright, conditions and disclaimer in source distributions and reproducing them in documentation/materials for binary distributions. Intel's name cannot endorse derived products without prior written permission. The artifact is provided as-is without warranty.

The project retains its shared OpenCV 5.0.0.93 pin; optional Haar availability depends on the installed build. Missing/unloadable Haar artifacts now retain manual boxes, record "unavailable on this OpenCV build" in observations, and require human review. The manifest records absence with a null SHA-256; it never claims an unavailable artifact was used.

`face_detector` accepts `yunet` (default), `haar` or `none`. Optional Haar inference runs on clipped, sRGB-encoded uint8 grayscale at source size with scaleFactor 1.1, minNeighbors 5 and minSize 24 x 24. Accepted rectangles get a fixed uncalibrated observation confidence of 0.6, not a probability. Default tier thresholds consequently cap Haar observations at tier 2. Manual confidence remains annotation certainty. Detection never proves identity, and zero detections never establishes absence of faces. Frontal, size, pose and occlusion limitations require human inspection and additional manual boxes.

Manual boxes keep their coordinates, scores, order and primary index. Automatic rectangles are appended in deterministic coordinate order only if IoU with every earlier observation is below 0.3. Manual boxes always take precedence over overlapping inference; inference duplicates are also suppressed. `none` retains manual observations only. Primary selection then follows the configured manual override or the highest-tier/largest-area rule. All source detections and counts are recorded in the face-observations stage and summarized in the report.

## YuNet runtime artifact preparation

The verified runtime cache helper is implemented in `src/human_models.py`. `YuNetFaceDetector` in `src/face_preservation.py` consumes verified weights, converts source-scale linear RGB to sRGB uint8 BGR, clips inferred boxes to image bounds and records acquisition or inference failure while preserving manual boxes. YuNet is the configuration default and its pinned metadata is recorded in the run manifest even when acquisition fails. Acquisition and inference failures are recorded in stage/report provenance and preserve manual observations. The notebook parameter cell can explicitly select optional Haar or disable inference. `yunet_model_metadata()` supplies the expected manifest model entry, including provenance even if acquisition fails.

- Artifact: `face_detection_yunet_2023mar.onnx`.
- Pinned opencv_zoo revision: `48b10fa505aeb51fbd127fc60a5c660040e823b7`.
- Exact source URL: https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- Independently downloaded outside the repository and verified: 232589 bytes; SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Model-directory license, inspected at the same revision: https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/LICENSE
- MIT License, copyright 2020 Shiqi Yu. Commercial use, modification, distribution, sublicensing and sale are permitted provided the copyright and permission notice accompany copies or substantial portions. Provided without warranty.
- Inference only; no training. Runtime weights go into the existing ignored `weights/` directory. Both cached and downloaded bytes must match the pinned size and SHA-256 before use. Failed acquisition and corrupt caches raise precise errors for the detector integration to record before continuing with manual boxes. Partial downloads are never published.

Local verification note: this Mac's Homebrew Python lacked a usable default CA bundle. Runtime download succeeded with `SSL_CERT_FILE=/etc/ssl/cert.pem`; certificate verification was retained. Without a valid trust store the helper reports the TLS failure. OpenCV Zoo also documents that the March 2023 model has fixed input dimensions, so OpenCV 5 integration must verify input sizing rather than assume arbitrary sizes work.

Adapter verification: real verified weights successfully inferred on a 193 x 127 blank source image using the installed OpenCV build with source-sized `FaceDetectorYN` input. This checks runtime input sizing only, not detection accuracy or OpenCV 5 compatibility; repeat after dependency restoration. Offline adapter tests use fake inference and local artifacts.

### Notebook verification on OpenCV 5

On 2026-10-22, Notebook 01 completed in synthetic quick mode with
`FACE_DETECTOR = "yunet"` on `opencv-python-headless==5.0.0.93` (runtime
`5.0.0`). The real runtime artifact was downloaded over the network and verified
against the pinned size and SHA-256. Notebook execution, weights, inputs, exports
and reports were kept outside the repository under `/tmp/exp50mp-iteration5/`;
the existing cache helper was redirected to that scratch directory for this run.
TLS verification used `SSL_CERT_FILE=/etc/ssl/cert.pem`.

The report's per-tier Landmark status column showed
`not available: no landmark found in source crop` for face 0 (tier 1), face 1
(tier 2) and face 2 (tier 3). Each tier retained one manual synthetic observation;
all three landmark counts were zero and displacement fields were null. These
drawn fixtures do not establish landmark accuracy on photographs. Numeric
per-face/per-tier displacement and the normalized failure threshold are covered
separately by offline tests with injected five-point estimators. The complete
suite passed (428 tests), with Ruff lint and formatting checks clean.
