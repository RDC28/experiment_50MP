"""CFA container writer for PRD_v3's RAW milestone: uncompressed strips or lossless JPEG tiles."""

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
# Compression 7 tiles, as the DNG SDK writes them. Each tile is one lossless JPEG (SOF3, predictor 1).
LOSSLESS_TILE = 256


def lossless_tile_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Tile sides: multiples of 16 (TIFF), at most LOSSLESS_TILE and never larger than the image.

    LibRaw misplaces rows of a lossless tile that is wider than the whole image, so small images get
    smaller tiles instead of one padded 256 x 256 tile.
    """
    if min(shape) < 16:
        raise ValueError("Lossless DNG tiles need an image of at least 16 x 16 photosites")
    return tuple(min(LOSSLESS_TILE, side // 16 * 16) for side in shape)


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


def _tiles(mosaic: np.ndarray, tile: tuple[int, int]):
    """Yield (row, column, tile) for full tiles; edge tiles are padded by repeating their last row/column."""
    for top in range(0, mosaic.shape[0], tile[0]):
        for left in range(0, mosaic.shape[1], tile[1]):
            block = np.ascontiguousarray(mosaic[top : top + tile[0], left : left + tile[1]], dtype=np.uint16)
            pad = ((0, tile[0] - block.shape[0]), (0, tile[1] - block.shape[1]))
            yield top, left, np.pad(block, pad, mode="edge") if any(after for _, after in pad) else block


class DNGWriter:
    """Write core Bayer DNGs without duplicating the full mosaic in memory.

    ``Compression`` in the records selects the layout: 1 streams uncompressed row strips, 7 streams
    lossless JPEG tiles. Structural success is not a LibRaw compatibility claim. The independent
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
        """Stream bounded strips or tiles, then verify tags and every decoded pixel."""
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
        lossless = records["Compression"].value == 7
        if lossless:
            import imagecodecs

            tile = lossless_tile_shape(shape)

            def segments():
                for _, _, block in _tiles(mosaic, tile):
                    yield imagecodecs.ljpeg_encode(block, bitspersample=16)

            # tifffile would otherwise assume 12-bit JPEG samples.
            layout = {"compression": 7, "tile": tile, "bitspersample": 16}
        else:

            def segments():
                for start in range(0, shape[0], rows_per_strip):
                    yield np.ascontiguousarray(mosaic[start : start + rows_per_strip], dtype="<u2").tobytes()

            layout = {"compression": None, "rowsperstrip": rows_per_strip}

        with tifffile.TiffWriter(path, byteorder="<", bigtiff=False) as writer:
            writer.write(
                segments(),
                shape=shape,
                dtype="<u2",
                photometric=32803,
                metadata=None,
                software=False,
                description=None,
                extratags=tags,
                **layout,
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
        """Check every emitted field and compare pixels in bounded strips or decoded tiles."""
        if type(rows_per_strip) is not int or rows_per_strip <= 0:
            raise ValueError("rows_per_strip must be a positive integer")
        lossless = records["Compression"].value == 7
        with tifffile.TiffFile(path) as container:
            if container.is_bigtiff or container.byteorder != "<" or len(container.pages) != 1:
                raise ValueError("Unexpected DNG container layout")
            page = container.pages[0]
            for name, record in records.items():
                tag = page.tags.get(tifffile.TIFF.TAGS[name])
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
            if lossless:
                self._compare_tiles(container, page, mosaic)
                return
        mapped = tifffile.memmap(path, mode="r")
        try:
            if mapped.shape != mosaic.shape or mapped.dtype != np.dtype("<u2"):
                raise ValueError("DNG pixel layout mismatch")
            for start in range(0, mosaic.shape[0], rows_per_strip):
                if not np.array_equal(mapped[start : start + rows_per_strip], mosaic[start : start + rows_per_strip]):
                    raise ValueError("DNG pixels differ")
        finally:
            mapped._mmap.close()

    @staticmethod
    def _compare_tiles(container: tifffile.TiffFile, page: tifffile.TiffPage, mosaic: np.ndarray) -> None:
        """Decode each stored lossless JPEG tile on its own and require the exact source pixels."""
        import imagecodecs

        tile = lossless_tile_shape(mosaic.shape)
        if page.shape != mosaic.shape or not page.is_tiled or (page.tilelength, page.tilewidth) != tile:
            raise ValueError("DNG pixel layout mismatch")
        offsets = list(zip(page.dataoffsets, page.databytecounts, strict=True))
        expected = list(_tiles(mosaic, tile))
        if len(offsets) != len(expected):
            raise ValueError("DNG tile count mismatch")
        handle = container.filehandle
        for (offset, count), (_, _, block) in zip(offsets, expected, strict=True):
            handle.seek(offset)
            try:
                decoded = imagecodecs.ljpeg_decode(handle.read(count))
            except imagecodecs.LjpegError as error:
                raise ValueError(f"DNG pixels differ: tile at offset {offset} does not decode: {error}") from error
            if decoded.dtype != np.uint16 or not np.array_equal(decoded, block):
                raise ValueError("DNG pixels differ")
