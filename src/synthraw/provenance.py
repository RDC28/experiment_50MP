"""Field provenance and JSON output following PRD_v3's metadata contract."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SCIENTIFIC_BOUNDARY = (
    "The system generates a plausible physical explanation of the rendered image. "
    "It does not recover the physical exposure that actually happened."
)
SYNTHETIC_CAPTURE_DISCLOSURE = (
    "Synthetic sensor reconstruction of a hypothetical exposure, derived from a rendered RGB source. "
    "No physical camera capture occurred; no photons represented by these files were recorded by a Sony camera."
)
FABRICATED_DISCLAIMER = "Fabricated simulation value; not measured and carries no confidence."
PROVENANCE_CLASSES = {
    "D": "Deterministic: computed exactly from inputs or configuration",
    "I": "Inferred: estimated from image evidence with stated confidence",
    "F": "Fabricated: plausible synthetic prior or realisation, never measured",
    "M": "Measured: derived from user-supplied reference RAW or calibration data",
}
DIGITAL_SOURCE_TYPES = {
    "algorithmic": "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
    "trained": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
}


class Provenance(str, Enum):
    """The four scientific provenance classes defined by PRD_v3."""

    DETERMINISTIC = "D"
    INFERRED = "I"
    FABRICATED = "F"
    MEASURED = "M"


@dataclass
class FieldRecord:
    """A metadata field; fabricated values must never claim confidence."""

    field: str
    value: Any
    provenance: Provenance
    method: str
    unit: str | None = None
    confidence: float | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    user_override: bool = False
    disclaimer: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate again before writing, including records edited after creation."""
        self.provenance = Provenance(self.provenance)
        if not self.field.strip() or not self.method.strip():
            raise ValueError("field and method must be nonempty")
        if self.confidence is not None:
            if self.provenance == Provenance.FABRICATED:
                raise ValueError("Fabricated fields must have confidence None")
            if isinstance(self.confidence, bool) or not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete metadata.json field schema."""
        self.validate()
        result = asdict(self)
        result["provenance"] = self.provenance.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldRecord:
        """Load a JSON field record with the same validation as new records."""
        return cls(**data)


class ProvenanceWriter:
    """Serialize field records without allowing nonstandard JSON numeric values."""

    def write(self, records: list[FieldRecord], path: str | Path) -> Path:
        """Write a JSON array of complete field records to the requested path."""
        payload = json.dumps([record.to_dict() for record in records], indent=2, allow_nan=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
        return destination

    def write_metadata(self, groups: Mapping[str, Iterable[FieldRecord]], path: str | Path) -> Path:
        """Write metadata.json: every stage's field records under the PRD scientific boundary.

        Fabricated fields keep confidence null and always carry a disclaimer.
        """
        payload = {
            "scientific_boundary": SCIENTIFIC_BOUNDARY,
            "disclosure": SYNTHETIC_CAPTURE_DISCLOSURE,
            "provenance_classes": PROVENANCE_CLASSES,
            "groups": {name: [_field_dict(record) for record in records] for name, records in groups.items()},
        }
        return _write_json(payload, path)

    def write_provenance(self, payload: Mapping[str, Any], path: str | Path) -> Path:
        """Write the mandatory provenance.json; the boundary statements cannot be overridden."""
        reserved = {"scientific_boundary", "disclosure", "synthetic_capture"}
        if reserved & payload.keys():
            raise ValueError(f"Provenance payload must not override {sorted(reserved)}")
        return _write_json(
            {
                "scientific_boundary": SCIENTIFIC_BOUNDARY,
                "disclosure": SYNTHETIC_CAPTURE_DISCLOSURE,
                "synthetic_capture": True,
                **payload,
            },
            path,
        )


def _field_dict(record: FieldRecord) -> dict[str, Any]:
    result = record.to_dict()
    if record.provenance == Provenance.FABRICATED and not result.get("disclaimer"):
        result["disclaimer"] = FABRICATED_DISCLAIMER
    return result


def _write_json(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False, default=str)
    destination.write_text(text + "\n", encoding="utf-8")
    return destination


def c2pa_fallback(status: Mapping[str, Any]) -> str | None:
    """Run-report line for C2PA that was requested but not written; None when signed or disabled."""
    if status.get("status") != "C2PA unavailable":
        return None
    return f"C2PA unavailable: {status.get('reason')}"


def write_c2pa(
    targets: Iterable[str | Path],
    *,
    enabled: bool,
    signer: Any = None,
    learned_models: bool = False,
    version: str = "0.1.0",
) -> dict[str, Any]:
    """Optionally embed a C2PA digitalSourceType declaration (PRD provenance section).

    Requires an importable c2pa-python and a caller-configured signer. Any absence or
    signing failure is recorded as "C2PA unavailable"; JSON provenance stays mandatory.
    Each target is signed to a sibling temporary file and atomically replaced.
    """
    if not enabled:
        return {"status": "C2PA disabled", "reason": "WRITE_C2PA is False", "signed": []}
    try:
        import c2pa
    except ImportError as error:
        return {"status": "C2PA unavailable", "reason": f"c2pa-python not importable: {error}", "signed": []}
    if signer is None:
        return {"status": "C2PA unavailable", "reason": "No C2PA signer configured", "signed": []}
    source_type = DIGITAL_SOURCE_TYPES["trained" if learned_models else "algorithmic"]
    mime_types = {".png": "image/png", ".tif": "image/tiff", ".tiff": "image/tiff", ".dng": "image/x-adobe-dng"}
    signed = []
    try:
        for target in map(Path, targets):
            manifest = {
                "claim_generator_info": [{"name": "synthraw", "version": version}],
                "title": target.name,
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": source_type,
                                    "description": SYNTHETIC_CAPTURE_DISCLOSURE,
                                }
                            ]
                        },
                    }
                ],
            }
            staged = target.with_name(f".{target.name}.c2pa")
            try:
                builder = c2pa.Builder(json.dumps(manifest))
                with target.open("rb") as source, staged.open("wb") as destination:
                    builder.sign(signer, mime_types[target.suffix.lower()], source, destination)
                staged.replace(target)
            finally:
                staged.unlink(missing_ok=True)
            signed.append(str(target))
    except Exception as error:
        return {"status": "C2PA unavailable", "reason": f"C2PA signing failed: {error}", "signed": signed}
    return {"status": "C2PA signed", "digitalSourceType": source_type, "signed": signed}
