"""Strip-based uncompressed CFA container writer for PRD_v3's RAW milestone."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile

from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import DNGFieldRecord, DNGMetadataBuilder

# tifffile owns these structural fields and writes them from the array/options.
_STRUCTURAL = frozenset(
    {"ImageWidth", "ImageLength", "BitsPerSample", "SamplesPerPixel", "PhotometricInterpretation", "Compression"}
)
_TYPES = {"BYTE": "B", "SHORT": "H", "LONG": "I", "DOUBLE": "d", "RATIONAL": "2I", "SRATIONAL": "2i"}


def _tag(record: DNGFieldRecord) -> tuple:
    """Encode a typed field without changing its provenance record."""
    code = tifffile.TIFF.TAGS[record.field]
    kind = record.dng_type.split("[")[0]
    if kind == "ASCII":
        return code, "s", 0, record.value, False
    values = np.asarray(record.value).reshape(-1).tolist()
    if kind in {"RATIONAL", "SRATIONAL"}:
        encoded = []
        limit = 2**31 - 1 if kind == "SRATIONAL" else 2**32 - 1
        for value in values:
            fraction = Fraction(float(value)).limit_denominator(1_000_000)
            if not -limit <= fraction.numerator <= limit or (kind == "RATIONAL" and fraction.numerator < 0):
                raise ValueError(f"{record.field} cannot be represented as {kind}")
            encoded.extend((fraction.numerator, fraction.denominator))
        return code, _TYPES[kind], len(values), tuple(encoded), False
    return code, _TYPES[kind], len(values), tuple(values), False


class DNGWriter:
    """Write core Bayer DNGs without duplicating the full mosaic in memory.

    Structural success is not a LibRaw compatibility claim. The independent
    LibRaw gate must also pass before an export is reported as successful.
    """

    def write(
        self,
        path: str | Path,
        mosaic: np.ndarray,
        records: dict[str, DNGFieldRecord],
        config: ExperimentConfig | None = None,
        *,
        rows_per_strip: int = 64,
    ) -> Path:
        """Stream bounded row strips, then verify tags and mapped pixels."""
        config = config or ExperimentConfig()
        DNGMetadataBuilder().validate(records, config)
        shape = (records["ImageLength"].value, records["ImageWidth"].value)
        if mosaic.dtype != np.uint16 or mosaic.shape != shape:
            raise ValueError("Mosaic must be uint16 with the metadata sensor dimensions")
        if type(rows_per_strip) is not int or rows_per_strip <= 0:
            raise ValueError("rows_per_strip must be a positive integer")
        rows_per_strip = min(rows_per_strip, shape[0])
        # Validate stripwise: a full-size boolean temporary would defeat streaming.
        for start in range(0, shape[0], rows_per_strip):
            if np.any(mosaic[start : start + rows_per_strip] > records["WhiteLevel"].value):
                raise ValueError("Mosaic exceeds WhiteLevel")
        if mosaic.nbytes >= 2**32 - 2**20:
            raise ValueError("Mosaic exceeds classic TIFF offset capacity")
        tags = [_tag(record) for name, record in records.items() if name not in _STRUCTURAL]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def strips():
            for start in range(0, shape[0], rows_per_strip):
                yield np.ascontiguousarray(mosaic[start : start + rows_per_strip], dtype="<u2").tobytes()

        with tifffile.TiffWriter(path, byteorder="<", bigtiff=False) as writer:
            writer.write(
                strips(),
                shape=shape,
                dtype="<u2",
                photometric=32803,
                compression=None,
                rowsperstrip=rows_per_strip,
                metadata=None,
                software=False,
                description=None,
                extratags=tags,
            )
        self.validate_structure(path, mosaic, records, rows_per_strip=rows_per_strip)
        return path

    def validate_structure(
        self,
        path: str | Path,
        mosaic: np.ndarray,
        records: dict[str, DNGFieldRecord],
        *,
        rows_per_strip: int = 64,
    ) -> None:
        """Check every emitted field and compare pixels in bounded mapped slices."""
        if type(rows_per_strip) is not int or rows_per_strip <= 0:
            raise ValueError("rows_per_strip must be a positive integer")
        with tifffile.TiffFile(path) as container:
            if container.is_bigtiff or container.byteorder != "<" or len(container.pages) != 1:
                raise ValueError("Unexpected DNG container layout")
            page = container.pages[0]
            for name, record in records.items():
                tag = page.tags.get(name)
                if tag is None:
                    raise ValueError(f"Missing DNG tag: {name}")
                actual = tag.value
                if record.dng_type == "ASCII":
                    matches = actual == record.value
                else:
                    actual = np.asarray(tuple(actual) if isinstance(actual, bytes) else actual).reshape(-1)
                    if record.dng_type.split("[")[0] in {"RATIONAL", "SRATIONAL"}:
                        pairs = actual.reshape(-1, 2)
                        actual = pairs[:, 0] / pairs[:, 1]
                    expected = np.asarray(record.value).reshape(-1)
                    matches = actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=1e-6)
                if not matches:
                    raise ValueError(f"DNG tag mismatch: {name}")
        mapped = tifffile.memmap(path, mode="r")
        try:
            if mapped.shape != mosaic.shape or mapped.dtype != np.dtype("<u2"):
                raise ValueError("DNG pixel layout mismatch")
            for start in range(0, mosaic.shape[0], rows_per_strip):
                if not np.array_equal(mapped[start : start + rows_per_strip], mosaic[start : start + rows_per_strip]):
                    raise ValueError("DNG pixels differ")
        finally:
            mapped._mmap.close()
