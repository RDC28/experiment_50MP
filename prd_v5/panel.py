"""Executable subset of the notebook's PRD field catalogue."""

from prd_v5.fields import resolve_camera
from prd_v5.hypothesis import CameraField, CameraHypothesis

CAMERA_FIELDS = {
    "CFA pattern": "cfa_pattern",
    "ADC bit depth": "adc_bits",
    "Black level": "black_level",
    "Full-well capacity": "full_well_e",
    "ISO": "iso",
    "Shutter time": "shutter_s",
    "Aperture": "aperture",
    "Random seed": "seed",
}


def resolve_camera_panel(
    base: CameraHypothesis, requests: list[dict], modes: dict, *, exposure_prior: dict | None = None
):
    """Resolve supported fields; reject unsupported edits before pixel work.

    Explicit catalogue values require USER mode and are labelled USER_SET.
    AUTO/FABRICATE without values use the declared synthetic base hypothesis.
    Joint exposure ESTIMATE requires an explicit exposure_prior. Other absent
    estimates fail rather than inventing evidence.
    Dual conversion gain accepts an explicitly accepted USER profile dictionary
    after ISO resolution, with no fabricated calibration or scalar well conflict.
    Read-noise amplitude remains in CAMERA_VALUES; its PRD row combines multiple
    distributions and is deliberately not treated as a scalar override.
    """
    names = [field["name"] for field in requests]
    if len(set(names)) != len(names) or set(modes) != set(names):
        raise ValueError("Panel requires unique fields and exactly one mode per field")
    selected_modes, candidates = {}, {}
    electronics = None
    if exposure_prior is not None:
        from prd_v5.exposure import resolve_exposure

        joint = {"ISO", "Shutter time", "Aperture"}
        fields = {field["name"]: field for field in requests}
        if not joint <= fields.keys() or any(
            modes[name] != "ESTIMATE" or fields[name]["value"] is not None for name in joint
        ):
            raise ValueError(
                "Joint exposure prior requires ISO, Shutter time and Aperture in ESTIMATE mode without values"
            )
        if not isinstance(exposure_prior, dict):
            raise ValueError("exposure_prior must be a dictionary")
        _, records, _ = resolve_exposure(base, **exposure_prior)
        candidates.update(
            {record.name: (record,) for record in records if record.name in {"iso", "shutter_s", "aperture"}}
        )
    registry = base.registry()
    for field in requests:
        name = field["name"]
        mode = modes[name]
        if mode not in {"AUTO", "ESTIMATE", "FABRICATE", "USER"}:
            raise ValueError(f"Invalid panel mode: {name}")
        value = field["value"]
        if name == "Dual conversion gain":
            if value is None and mode == "AUTO":
                continue
            if mode != "USER" or not isinstance(value, dict):
                raise ValueError("Dual conversion gain requires USER mode and a supplied profile")
            electronics = dict(value)
            continue
        target = CAMERA_FIELDS.get(name)
        if target is None:
            if value is not None or mode != field["mode"]:
                raise ValueError(f"Unsupported catalogue override: {name}")
            continue
        selected_modes[target] = mode
        if value is not None:
            if mode != "USER":
                raise ValueError(f"Explicit panel value requires USER mode: {name}")
            candidates[target] = (
                CameraField(
                    target,
                    value,
                    registry[target]["units"],
                    "USER",
                    "USER_SET",
                    None,
                    None,
                    "User-supplied notebook catalogue hypothesis",
                    "CameraHypothesis domain checks",
                ),
            )
    camera, records = resolve_camera(base, selected_modes, candidates)
    if electronics is not None:
        from prd_v5.electronics import DualConversionProfile

        if "full_well_e" in candidates:
            raise ValueError("Dual conversion gain conflicts with explicit Full-well capacity")
        accepted = electronics.pop("accept_experimental", False)
        camera, selected, _ = DualConversionProfile(**electronics).resolve(camera, accept_experimental=accepted)
        replacements = {r.name: r for r in selected if r.name in {"full_well_e", "read_noise_e"}}
        records = tuple(replacements.get(r.name, r) for r in records)
    return camera, records


def resolve_analytic_panel(
    camera: CameraHypothesis,
    isp,
    requests: list[dict],
    modes: dict,
    *,
    exposure_prior: dict | None = None,
    colour_source=None,
    colour_strip_rows: int = 64,
):
    """Resolve camera inputs and AsShotNeutral into shared analytic hypotheses.

    AsShotNeutral is a positive, green-normalized camera neutral, not WB gains.
    USER values invert into gains; absent AUTO/FABRICATE values keep the declared
    ISP hypothesis. Explicit colour_source evidence requires ESTIMATE mode and
    supplies a gray-world rendered-sRGB proxy hypothesis shared by paired runs.
    No calibrated illuminant estimator or camera-profile solver is implied.
    ColorMatrix1/2 accepts one USER D65 XYZ-to-camera matrix for both tags.
    Other colour metadata remains derived from the ISP or unsupported.
    """
    import numpy as np

    from prd_v5.fields import resolve_isp

    names = [field["name"] for field in requests]
    if len(set(names)) != len(names) or set(modes) != set(names):
        raise ValueError("Panel requires unique fields and exactly one mode per field")
    colour_records = ()
    if colour_source is not None:
        from prd_v5.colour import resolve_gray_world

        neutral_fields = [field for field in requests if field["name"] == "AsShotNeutral"]
        if not neutral_fields or modes["AsShotNeutral"] != "ESTIMATE" or neutral_fields[0]["value"] is not None:
            raise ValueError("Colour evidence requires AsShotNeutral in ESTIMATE mode without a value")
        _, colour_records, _ = resolve_gray_world(isp, colour_source, strip_rows=colour_strip_rows)
    profile_modes = {}
    profile_candidates = {}
    for field in requests:
        if field["name"] != "ColorMatrix1/2":
            continue
        mode = modes[field["name"]]
        profile_modes["camera_to_srgb"] = mode
        if field["value"] is not None:
            if mode != "USER":
                raise ValueError("Explicit panel value requires USER mode: ColorMatrix1/2")
            from prd_v5.profile import resolve_d65_profile

            _, records, _ = resolve_d65_profile(
                isp, field["value"], label="Notebook ColorMatrix1/2 single-D65 user hypothesis"
            )
            profile_candidates = {r.name: (r,) for r in records if r.name == "camera_to_srgb"}
    camera_requests = [field for field in requests if field["name"] not in {"AsShotNeutral", "ColorMatrix1/2"}]
    camera, camera_records = resolve_camera_panel(
        camera,
        camera_requests,
        {field["name"]: modes[field["name"]] for field in camera_requests},
        exposure_prior=exposure_prior,
    )
    isp_modes = dict(profile_modes)
    candidates = {record.name: (record,) for record in colour_records}
    candidates.update(profile_candidates)
    for field in requests:
        if field["name"] != "AsShotNeutral":
            continue
        mode = modes[field["name"]]
        isp_modes["white_balance"] = mode
        if field["value"] is not None:
            if mode != "USER":
                raise ValueError("Explicit panel value requires USER mode: AsShotNeutral")
            neutral = np.asarray(field["value"], dtype=np.float64)
            if neutral.shape != (3,) or not np.isfinite(neutral).all() or np.any(neutral <= 0) or neutral[1] != 1:
                raise ValueError("AsShotNeutral must contain three positive finite values with green equal to 1")
            profile = candidates["camera_to_srgb"][0].value if "camera_to_srgb" in candidates else isp.camera_to_srgb
            white = np.linalg.solve(np.asarray(profile, dtype=np.float64), np.ones(3))
            candidates["white_balance"] = (
                CameraField(
                    "white_balance",
                    tuple(white / neutral / white[1]),
                    "ratio",
                    "USER",
                    "USER_SET",
                    None,
                    None,
                    "Camera-space sRGB white divided by user-supplied green-normalized AsShotNeutral",
                    "Positive neutral and AnalyticISP domain checks; no illuminant estimation",
                ),
            )
    isp, isp_records = resolve_isp(isp, isp_modes, candidates)
    return camera, isp, camera_records, isp_records


def resolve_sr_panel(backend: str, checkpoint: dict | None, *, allow_baseline: bool):
    """Preflight SR requests without importing tensors or running inference.

    Missing weights may use the explicitly accepted analytic baseline. Supplied
    invalid weights always fail; they never trigger a silent fallback. Readiness
    is not execution evidence: child manifests record successful model execution.
    """
    from prd_v5.fidelity import SRCheckpoint

    if backend not in {"HAT", "REAL_ESRGAN", "BILINEAR_BASELINE"}:
        raise ValueError("Unsupported notebook SR backend")
    if backend == "BILINEAR_BASELINE" and checkpoint is not None:
        raise ValueError("Baseline SR cannot accept a checkpoint")
    if checkpoint is not None:
        spec = SRCheckpoint(**checkpoint)
        if spec.backend != backend:
            raise ValueError("Requested SR backend differs from checkpoint backend")
        evidence = spec.verify()
        return spec, {
            "requested": backend,
            "effective": None,
            "planned": backend,
            "status": "checkpoint verified; inference not executed",
            "checkpoint": evidence,
        }
    if not allow_baseline:
        raise ValueError("SR checkpoint unavailable; explicit analytic baseline opt-in required")
    return None, {
        "requested": backend,
        "effective": None,
        "planned": "BILINEAR_BASELINE",
        "status": "explicit baseline accepted; inference not executed",
        "skip_reason": None if backend == "BILINEAR_BASELINE" else "No user checkpoint supplied",
    }


def resolve_analytic_panel_from_source(
    camera,
    isp,
    requests,
    modes,
    *,
    colour_source_path=None,
    assume_unprofiled_srgb=False,
    max_decode_bytes=256 * 1024 * 1024,
    strip_rows=64,
    exposure_prior=None,
):
    """Resolve one optional decoded colour proxy, releasing scratch on all exits.

    The same immutable result is shared across paired runs. A null path preserves
    panel defaults. Decoder admission precedes disk allocation; source hashes and
    colour policy are retained in the estimated white-balance selection evidence.
    """
    from dataclasses import replace
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import numpy as np

    from prd_v5.decode import decode_source
    from prd_v5.inputs import InputInspector

    if colour_source_path is None:
        return resolve_analytic_panel(camera, isp, requests, modes, exposure_prior=exposure_prior)
    record = InputInspector().inspect(colour_source_path, assume_unprofiled_srgb=assume_unprofiled_srgb)
    if type(max_decode_bytes) is not int or max_decode_bytes <= 0:
        raise ValueError("max_decode_bytes must be a positive integer")
    if record.width * record.height * 8 > max_decode_bytes:
        raise ValueError("Source decoder memory budget exceeded before scratch allocation")
    with TemporaryDirectory(prefix="v5-colour-") as scratch:
        pixels = np.memmap(
            Path(scratch) / "colour.f32", mode="w+", dtype=np.float32, shape=(record.height, record.width, 3)
        )
        try:
            decoded = decode_source(
                colour_source_path,
                pixels,
                assume_unprofiled_srgb=assume_unprofiled_srgb,
                max_decode_bytes=max_decode_bytes,
                strip_rows=strip_rows,
            )
            selected_camera, selected_isp, camera_records, isp_records = resolve_analytic_panel(
                camera,
                isp,
                requests,
                modes,
                exposure_prior=exposure_prior,
                colour_source=pixels,
                colour_strip_rows=strip_rows,
            )
            isp_records = tuple(
                replace(item, posterior={**item.posterior, "decoded_colour_source": decoded})
                if item.name == "white_balance"
                else item
                for item in isp_records
            )
            return selected_camera, selected_isp, camera_records, isp_records
        finally:
            pixels._mmap.close()
