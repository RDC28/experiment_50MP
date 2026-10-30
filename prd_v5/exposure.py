"""Joint exposure hypotheses connected to the executable v5 camera contract."""

from dataclasses import asdict

from prd_v5.fields import resolve_camera
from prd_v5.hypothesis import CameraField, CameraHypothesis
from synthraw.camera_profile import CameraProfileManager
from synthraw.exposure import ExposurePosteriorEstimator


def resolve_exposure(
    base: CameraHypothesis,
    *,
    scene_ev100: float,
    candidate_count: int = 8,
    electronic_shutter: bool = False,
    expanded_iso: bool = False,
) -> tuple[CameraHypothesis, tuple[CameraField, ...], dict]:
    """Select a joint heuristic exposure under an explicit Sony-like envelope.

    No brightness meter, blur estimator or calibrated posterior is implied.
    Relative weights normalize only over retained alternatives. ISO changes
    executable electronics; shutter/aperture remain metadata hypotheses because
    absolute scene radiance and lens throughput are unknown. ISP exposure gain
    is deliberately independent and unchanged. No frame arrays are allocated.
    """
    if not isinstance(base, CameraHypothesis):
        raise ValueError("base must be a CameraHypothesis")
    posterior = ExposurePosteriorEstimator().estimate(
        CameraProfileManager().build("SONY_A1II_LIKE"),
        scene_ev100=scene_ev100,
        candidate_count=candidate_count,
        electronic_shutter=electronic_shutter,
        expanded_iso=expanded_iso,
    )
    joint = [asdict(candidate) for candidate in posterior.candidates]
    selected = joint[0]
    names = {"iso": "iso", "shutter_s": "exposure_time", "aperture": "f_number"}
    registry = base.registry()
    candidates = {
        name: (
            CameraField(
                name,
                selected[key],
                registry[name]["units"],
                "ESTIMATE",
                "ESTIMATED",
                None,
                {"joint_candidates": joint, "scene_ev100": scene_ev100},
                "Heuristic joint EV100 ranking under explicit scene-scale and Sony-like profile priors",
                "Uncalibrated relative weights; not recovered capture settings or image-derived evidence",
            ),
        )
        for name, key in names.items()
    }
    camera, records = resolve_camera(base, dict.fromkeys(names, "ESTIMATE"), candidates)
    report = {
        "status": "effective",
        "method": "heuristic_joint_ev100",
        "profile": "SONY_A1II_LIKE",
        "scene_ev100": scene_ev100,
        "electronic_shutter": electronic_shutter,
        "expanded_iso": expanded_iso,
        "joint_candidates": joint,
        "selected": selected,
        "limitations": (
            "Explicit EV prior, not image metering; weights normalize over retained candidates only. "
            "ISO trades collected electrons for analogue gain, lowering SNR at equal raw signal; "
            "shutter and aperture do not rescale unknown radiance. "
            "ISP exposure_gain is independent. No calibrated capture-exposure recovery is claimed."
        ),
    }
    return camera, records, report
