"""Field provenance and JSON output following PRD_v3's metadata contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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
