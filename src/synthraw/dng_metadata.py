"""Core CFA field contract and dependencies from PRD_v3's DNG specification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose, log2
from numbers import Real
from typing import Any

import numpy as np

from synthraw.config import ExperimentConfig
from synthraw.geometry import GeometryPlan
from synthraw.provenance import FieldRecord, Provenance

_EXPOSURE_TYPES = {
    "ExposureTime": "RATIONAL",
    "FNumber": "RATIONAL",
    "PhotographicSensitivity": "LONG",
    "ExposureBiasValue": "SRATIONAL",
}

_LENS_TYPES = {
    "LensInfo": "RATIONAL[4]",
    "FocalLength": "RATIONAL",
    "FocalLengthIn35mmFilm": "SHORT",
    "LensMake": "ASCII",
    "LensModel": "ASCII",
}

CFA_PATTERNS = {
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}
DISCLOSURE = (
    "Synthetic sensor-space reconstruction derived from a rendered RGB source. No physical camera capture occurred."
)


@dataclass
class DNGFieldRecord(FieldRecord):
    """A provenance record with its container type, including element count."""

    dng_type: str = ""

    def validate(self) -> None:
        super().validate()
        if not self.dng_type:
            raise ValueError("DNG field requires a container type")


class DNGMetadataBuilder:
    """Build core RAW fields; profile defaults are explicitly fabricated priors.

    Exposure inputs retain upstream provenance; APEX values are deterministic.
    Optional calibrated fields are separate from this core schema. No binary
    MakerNotes or inferred capture time/location are accepted.
    """

    def build(
        self,
        geometry: GeometryPlan,
        config: ExperimentConfig | None = None,
        *,
        cfa_phase: str = "RGGB",
        black_level: float = 512,
        white_level: int | None = None,
        color_matrix1: Any = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        color_matrix2: Any = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        as_shot_neutral: tuple[float, float, float] = (1, 1, 1),
        noise_profile: tuple[float, ...] = (0.001, 0.000001),
        software_version: str = "0.1.0",
        exposure: Mapping[str, FieldRecord] | None = None,
        lens: Mapping[str, FieldRecord] | None = None,
    ) -> dict[str, DNGFieldRecord]:
        """Construct allocation-free metadata for a rectangular Bayer sensor.

        NoiseProfile must come from the same noise model applied to the pixels;
        the defaults only describe a trivial synthetic milestone fixture.
        """
        config = config or ExperimentConfig()
        if cfa_phase not in CFA_PATTERNS:
            raise ValueError("Unsupported CFA phase")
        if not software_version.strip() or not software_version.isascii():
            raise ValueError("Software version must be nonempty ASCII")
        records: dict[str, DNGFieldRecord] = {}

        def add(name: str, value: Any, dtype: str, provenance: str = "D") -> None:
            records[name] = DNGFieldRecord(
                field=name,
                value=value,
                provenance=Provenance(provenance),
                method="PRD_v3 core DNG contract" if provenance == "D" else "synthetic fixture profile prior",
                dng_type=dtype,
            )

        for name, value, dtype in (
            ("DNGVersion", (1, 7, 1, 0), "BYTE[4]"),
            # NoiseProfile requires DNG 1.3; no newer rendering features are emitted.
            ("DNGBackwardVersion", (1, 3, 0, 0), "BYTE[4]"),
            ("ImageWidth", geometry.width, "LONG"),
            ("ImageLength", geometry.height, "LONG"),
            ("BitsPerSample", 16, "SHORT"),
            ("SamplesPerPixel", 1, "SHORT"),
            ("PhotometricInterpretation", 32803, "SHORT"),
            ("Compression", 1, "SHORT"),
            ("CFARepeatPatternDim", (2, 2), "SHORT[2]"),
            ("CFAPlaneColor", (0, 1, 2), "BYTE[3]"),
            ("CFALayout", 1, "SHORT"),
            ("BlackLevelRepeatDim", (1, 1), "SHORT[2]"),
            ("WhiteLevel", (1 << config.ADC_BITS) - 1 if white_level is None else white_level, "LONG"),
            ("NoiseProfile", tuple(noise_profile), f"DOUBLE[{len(noise_profile)}]"),
            ("NoiseReductionApplied", 0, "RATIONAL"),
            ("DefaultCropOrigin", geometry.default_crop_origin, "LONG[2]"),
            ("DefaultCropSize", geometry.default_crop_size, "LONG[2]"),
            ("ActiveArea", geometry.active_area, "LONG[4]"),
            ("Orientation", 1, "SHORT"),
            ("Make", "Synthetic", "ASCII"),
            ("Software", f"SyntheticCameraRawColab {software_version}", "ASCII"),
            ("ImageDescription", DISCLOSURE, "ASCII"),
        ):
            add(name, value, dtype)
        model, unique = {
            "SONY_A1II_LIKE": ("Sony Alpha 1 II-like Synthetic Sensor", "Synthetic Sony Alpha 1 II-like 50MP v1"),
            "CINEMA_50MP_VIRTUAL": ("Virtual Cinema 50MP Synthetic Sensor", "Synthetic Virtual Cinema 50MP v1"),
            "VIRTUAL_16BIT_SENSOR": ("VIRTUAL_16BIT_SENSOR", "Synthetic VIRTUAL_16BIT_SENSOR v1"),
        }[config.TARGET_PROFILE]
        add("Model", model, "ASCII")
        add("UniqueCameraModel", unique, "ASCII")
        add("CFAPattern", CFA_PATTERNS[cfa_phase], "BYTE[4]", "F")
        add("BlackLevel", black_level, "RATIONAL", "F")
        for index, matrix in enumerate((color_matrix1, color_matrix2), 1):
            array = np.asarray(matrix, dtype=float)
            if array.shape != (3, 3):
                raise ValueError("ColorMatrix must be 3x3")
            add(f"ColorMatrix{index}", tuple(array.ravel().tolist()), "SRATIONAL[9]", "F")
            add(f"CalibrationIlluminant{index}", (17, 21)[index - 1], "SHORT", "F")
            add(f"CameraCalibration{index}", (1, 0, 0, 0, 1, 0, 0, 0, 1), "SRATIONAL[9]", "F")
        add("AnalogBalance", (1, 1, 1), "RATIONAL[3]", "F")
        add("AsShotNeutral", as_shot_neutral, "RATIONAL[3]", "F")
        for name, value, dtype in (
            ("BaselineExposure", 0, "SRATIONAL"),
            ("BaselineNoise", 1, "RATIONAL"),
            ("BaselineSharpness", 1, "RATIONAL"),
            ("LinearResponseLimit", 1, "RATIONAL"),
            ("AntiAliasStrength", 1, "RATIONAL"),
        ):
            add(name, value, dtype, "F")
        for name, record in (exposure or {}).items():
            if name not in _EXPOSURE_TYPES or name != record.field:
                raise ValueError(f"Unsupported exposure input: {name}")
            record.validate()
            records[name] = DNGFieldRecord(**asdict(record), dng_type=_EXPOSURE_TYPES[name])
        for source, derived, factor, dtype in (
            ("ExposureTime", "ShutterSpeedValue", -1, "SRATIONAL"),
            ("FNumber", "ApertureValue", 2, "RATIONAL"),
        ):
            if source in records:
                number = records[source].value
                if isinstance(number, bool) or not isinstance(number, Real) or not np.isfinite(number) or number <= 0:
                    raise ValueError(f"{source} must be a positive finite scalar")
                add(derived, factor * log2(number), dtype)
                records[derived].method = f"APEX derived from {source}"
                records[derived].dependencies = [source]
        for name, record in (lens or {}).items():
            if name not in _LENS_TYPES or name != record.field:
                raise ValueError(f"Unsupported lens input: {name}")
            record.validate()
            records[name] = DNGFieldRecord(**asdict(record), dng_type=_LENS_TYPES[name])
        self.validate(records, config)
        return records

    def validate(self, records: dict[str, DNGFieldRecord], config: ExperimentConfig) -> None:
        """Recheck mutable records before export; reject unsafe identity additions."""
        for name, record in records.items():
            record.validate()
            if name != record.field:
                raise ValueError("Field name does not match record")
            if "makernote" in name.lower():
                raise ValueError("MakerNotes are prohibited")
            if ("date" in name.lower() or name.upper().startswith("GPS")) and not record.user_override:
                raise ValueError("Date/GPS requires explicit user input")
            if "serial" in name.lower() and (
                not isinstance(record.value, str) or not record.value.startswith("SYNTHETIC-")
            ):
                raise ValueError("Serial must be an explicitly synthetic token")
            if record.dng_type == "ASCII" and (not isinstance(record.value, str) or not record.value.isascii()):
                raise ValueError("ASCII field contains invalid text")

        def value(name: str) -> Any:
            return records[name].value

        if value("Make") != "Synthetic" or value("ImageDescription") != DISCLOSURE:
            raise ValueError("Synthetic identity and disclosure are mandatory")
        for name, expected in (
            ("BitsPerSample", 16),
            ("SamplesPerPixel", 1),
            ("PhotometricInterpretation", 32803),
            ("Compression", 1),
            ("NoiseReductionApplied", 0),
        ):
            if type(value(name)) is not int or value(name) != expected:
                raise ValueError(f"Invalid {name}")
        for name, dtype in _EXPOSURE_TYPES.items():
            if name not in records:
                continue
            record = records[name]
            number = record.value
            if record.dng_type != dtype:
                raise ValueError(f"Invalid container type for {name}")
            if record.provenance not in {Provenance.INFERRED, Provenance.FABRICATED}:
                raise ValueError(f"{name} requires inferred or fabricated provenance")
            if isinstance(number, bool) or not isinstance(number, Real) or not np.isfinite(number):
                raise ValueError(f"{name} must be a finite scalar")
            if name == "ExposureTime" and (
                number <= 0 or (config.TARGET_PROFILE == "SONY_A1II_LIKE" and number < 1 / 32000)
            ):
                raise ValueError("ExposureTime outside profile range")
            if name == "FNumber" and number < 1:
                raise ValueError("FNumber must be at least 1 for nonnegative EXIF ApertureValue")
            if name == "PhotographicSensitivity" and (
                type(number) is not int
                or not 1 <= number <= 2**32 - 1
                or (config.TARGET_PROFILE == "SONY_A1II_LIKE" and not 50 <= number <= 102400)
            ):
                raise ValueError("PhotographicSensitivity outside profile range")
            if name == "ExposureBiasValue" and not -5 <= number <= 5:
                raise ValueError("ExposureBiasValue outside design range")
        for source, derived, factor, dtype in (
            ("ExposureTime", "ShutterSpeedValue", -1, "SRATIONAL"),
            ("FNumber", "ApertureValue", 2, "RATIONAL"),
        ):
            if source not in records and derived not in records:
                continue
            if source not in records or derived not in records:
                raise ValueError(f"{derived} requires {source} and its derived value")
            record = records[derived]
            number = record.value
            if (
                record.dng_type != dtype
                or record.provenance != Provenance.DETERMINISTIC
                or record.dependencies != [source]
                or isinstance(number, bool)
                or not isinstance(number, Real)
                or not np.isfinite(number)
                or not isclose(number, factor * log2(value(source)), rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError(f"Invalid derived {derived}")
        for name, dtype in _LENS_TYPES.items():
            if name not in records:
                continue
            record = records[name]
            allowed = {Provenance.INFERRED, Provenance.FABRICATED}
            if name == "LensInfo":
                allowed.add(Provenance.MEASURED)
            elif name == "FocalLength":
                allowed = {Provenance.INFERRED}
            elif name == "FocalLengthIn35mmFilm":
                allowed = {Provenance.DETERMINISTIC, Provenance.INFERRED}
            if record.dng_type != dtype or record.provenance not in allowed:
                raise ValueError(f"Invalid type or provenance for {name}")
            if dtype == "ASCII":
                if not record.value.strip() or "\x00" in record.value:
                    raise ValueError(f"{name} requires nonempty ASCII text without NUL")
            elif name == "LensInfo":
                numbers = np.asarray(record.value)
                if (
                    numbers.shape != (4,)
                    or numbers.dtype.kind not in "iuf"
                    or not np.isfinite(numbers).all()
                    or (numbers <= 0).any()
                    or numbers[0] > numbers[1]
                ):
                    raise ValueError("LensInfo requires ordered focal limits and positive maximum apertures")
            elif name == "FocalLengthIn35mmFilm":
                if type(record.value) is not int or not 1 <= record.value <= 65535:
                    raise ValueError("35mm equivalent focal length must fit a positive SHORT")
            elif (
                isinstance(record.value, bool)
                or not isinstance(record.value, Real)
                or not np.isfinite(record.value)
                or record.value <= 0
            ):
                raise ValueError("FocalLength requires a positive finite scalar")
        if "LensInfo" in records and "FocalLength" in records:
            minimum, maximum, _, _ = value("LensInfo")
            if not minimum <= value("FocalLength") <= maximum:
                raise ValueError("FocalLength outside selected lens range")
        white, black = value("WhiteLevel"), value("BlackLevel")
        if type(white) is not int or not 0 < white <= (1 << config.ADC_BITS) - 1:
            raise ValueError("WhiteLevel exceeds ADC range")
        if not np.isfinite(black) or not 0 <= black < white:
            raise ValueError("BlackLevel must be below WhiteLevel")
        if tuple(value("CFAPattern")) not in CFA_PATTERNS.values():
            raise ValueError("Invalid Bayer phase")
        for index in (1, 2):
            for prefix in ("ColorMatrix", "CameraCalibration"):
                matrix = np.asarray(value(f"{prefix}{index}"), dtype=float)
                if matrix.shape != (9,) or not np.isfinite(matrix).all():
                    raise ValueError("Matrix must contain nine finite values")
                if abs(np.linalg.det(matrix.reshape(3, 3))) < 1e-12:
                    raise ValueError("Matrix must be invertible")
        for name in ("AnalogBalance", "AsShotNeutral"):
            array = np.asarray(value(name), dtype=float)
            if array.shape != (3,) or not np.isfinite(array).all() or (array <= 0).any():
                raise ValueError(f"{name} requires three positive values")
        noise = np.asarray(value("NoiseProfile"), dtype=float)
        if noise.shape not in ((2,), (6,)) or not np.isfinite(noise).all():
            raise ValueError("NoiseProfile requires one or three finite S/O pairs")
        if (noise[::2] <= 0).any() or (noise[1::2] < 0).any():
            raise ValueError("NoiseProfile requires S > 0 and O >= 0")
        for name in ("BaselineNoise", "BaselineSharpness"):
            if not np.isfinite(value(name)) or value(name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, low, high in (
            ("BaselineExposure", -5, 5),
            ("LinearResponseLimit", 0, 1),
            ("AntiAliasStrength", 0, 1),
        ):
            if not np.isfinite(value(name)) or not low <= value(name) <= high:
                raise ValueError(f"{name} outside design range")
        width, height = value("ImageWidth"), value("ImageLength")
        if any(type(side) is not int or side <= 0 or side % 2 for side in (width, height)):
            raise ValueError("Bayer dimensions must be positive even integers")
        active = value("ActiveArea")
        origin, size = value("DefaultCropOrigin"), value("DefaultCropSize")
        if len(active) != 4 or len(origin) != 2 or len(size) != 2:
            raise ValueError("Invalid geometry vector length")
        if any(type(item) is not int for item in (*active, *origin, *size)):
            raise ValueError("Geometry must use integer coordinates")
        top, left, bottom, right = active
        x, y = origin
        cw, ch = size
        # DefaultCropOrigin is relative to ActiveArea, not to the full sensor.
        if not (0 <= top < bottom <= height and 0 <= left < right <= width):
            raise ValueError("ActiveArea outside sensor")
        if not (x >= 0 and y >= 0 and cw > 0 and ch > 0 and x + cw <= right - left and y + ch <= bottom - top):
            raise ValueError("Default crop outside ActiveArea")
