"""Independent RAW development and master export for PRD_v3 TIFF requirements."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import tifffile

from synthraw.libraw_validator import HardValidationError

ADOBE_GAMMA = 563 / 256


def adobe_rgb_icc() -> bytes:
    """Build an ICC v2 Adobe RGB matrix/TRC profile with a D50 connection space."""

    def xyz(values: tuple[float, float, float]) -> bytes:
        return b"XYZ \0\0\0\0" + struct.pack(">3i", *(round(v * 65536) for v in values))

    label = b"Synthetic Adobe RGB (1998)\0"
    description = b"desc\0\0\0\0" + struct.pack(">I", len(label)) + label + bytes(78)
    curve = b"curv\0\0\0\0" + struct.pack(">IH", 1, 563)
    # Bradford-adapted Adobe RGB primaries, expressed in the ICC D50 PCS.
    tags = {
        b"desc": description,
        b"cprt": b"text\0\0\0\0Public domain synthetic profile\0",
        b"wtpt": xyz((0.9642, 1.0, 0.8249)),
        b"rXYZ": xyz((0.6097559, 0.3111242, 0.0194811)),
        b"gXYZ": xyz((0.2052401, 0.6256560, 0.0608902)),
        b"bXYZ": xyz((0.1492240, 0.0632197, 0.7448387)),
        b"rTRC": curve,
        b"gTRC": curve,
        b"bTRC": curve,
    }
    header = bytearray(128)
    header[8:12] = bytes((2, 0x10, 0, 0))
    header[12:24] = b"mntrRGB XYZ "
    header[24:36] = struct.pack(">6H", 2000, 1, 1, 0, 0, 0)
    header[36:40] = b"acsp"
    header[68:80] = struct.pack(">3i", 63190, 65536, 54061)
    offset = 132 + 12 * len(tags)
    table = bytearray(struct.pack(">I", len(tags)))
    payload = bytearray()
    for signature, data in tags.items():
        table.extend(signature + struct.pack(">II", offset + len(payload), len(data)))
        payload.extend(data)
        payload.extend(bytes((-len(data)) % 4))
    struct.pack_into(">I", header, 0, offset + len(payload))
    return bytes(header + table + payload)


class TIFFWriter:
    """Develop a synthetic DNG via LibRaw into a tagged, stripped 16-bit master."""

    def write(self, path: str | Path, dng_path: str | Path, *, rows_per_strip: int = 64) -> Path:
        """Use camera WB and Adobe RGB with an explicitly matched pure gamma TRC."""
        if isinstance(rows_per_strip, bool) or not isinstance(rows_per_strip, int) or rows_per_strip < 1:
            raise ValueError("rows_per_strip must be a positive integer")
        path = Path(path)
        if path.resolve() == Path(dng_path).resolve():
            raise ValueError("TIFF destination must differ from the source DNG")
        try:
            import rawpy

            with rawpy.imread(str(dng_path)) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    output_color=rawpy.ColorSpace.Adobe,
                    gamma=(ADOBE_GAMMA, 0),
                    use_camera_wb=True,
                    no_auto_bright=True,
                    user_flip=0,
                )
        except Exception as exc:
            raise HardValidationError(f"Independent RAW development failed: {exc}") from exc
        if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise HardValidationError("RAW development must produce uint16 RGB")
        profile = adobe_rgb_icc()
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            path,
            data=(rgb[start : start + rows_per_strip] for start in range(0, rgb.shape[0], rows_per_strip)),
            shape=rgb.shape,
            dtype=rgb.dtype,
            photometric="rgb",
            rowsperstrip=rows_per_strip,
            metadata=None,
            iccprofile=profile,
            bigtiff=rgb.nbytes >= 2**32 - 2**25,
        )
        self.validate(path, rgb, profile, rows_per_strip=rows_per_strip)
        return path

    @staticmethod
    def validate(path: Path, rgb: np.ndarray, profile: bytes, *, rows_per_strip: int = 64) -> None:
        """Check the master identity and pixels with bounded memory-mapped strips."""
        with tifffile.TiffFile(path) as container:
            page = container.pages[0]
            if page.shape != rgb.shape or page.dtype != np.uint16 or page.photometric != 2:
                raise HardValidationError("Developed TIFF geometry or RGB precision differs")
            if page.tags[34675].value != profile:
                raise HardValidationError("Developed TIFF ICC profile differs")
        mapped = tifffile.memmap(path, mode="r")
        try:
            for start in range(0, rgb.shape[0], rows_per_strip):
                if not np.array_equal(mapped[start : start + rows_per_strip], rgb[start : start + rows_per_strip]):
                    raise HardValidationError("Developed TIFF pixels differ")
        finally:
            mapped._mmap.close()


def _linear_rgb(rgb: np.ndarray) -> np.ndarray:
    """Require finite scene-linear RGB without clipping HDR or negative samples."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError("Expected nonempty H x W x 3 scene-linear RGB")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("Scene-linear RGB must contain finite floating-point samples")
    with np.errstate(over="ignore"):
        result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Scene-linear RGB must be representable as float32")
    return result


class EXRWriter:
    """Preserve unclamped linear sRGB/D65 HDR radiance per PRD_v3 master export."""

    def write(self, path: str | Path, radiance: np.ndarray) -> Path:
        """Write lossless float32 RGB with explicit primary chromaticities."""
        import OpenEXR

        rgb = _linear_rgb(radiance)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage,
            "chromaticities": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
            "comments": "Synthetic scene-linear reconstruction; linear sRGB / D65",
        }
        with OpenEXR.File(header, {name: np.ascontiguousarray(rgb[..., i]) for i, name in enumerate("RGB")}) as exr:
            exr.write(str(path))
        with OpenEXR.File(str(path), separate_channels=True) as reopened:
            for i, name in enumerate("RGB"):
                if not np.array_equal(reopened.channels()[name].pixels, rgb[..., i]):
                    raise HardValidationError("Scene-linear EXR pixels differ")
        return path


class PreviewWriter:
    """Create a bounded display preview from linear sRGB, separate from RAW development."""

    def write(self, path: str | Path, radiance: np.ndarray, *, max_side: int = 1600) -> Path:
        """Downsample before applying a clipped standard sRGB display transfer."""
        from PIL import Image, ImageCms

        if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 1:
            raise ValueError("max_side must be a positive integer")
        rgb = _linear_rgb(radiance)
        height, width = rgb.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if size != (width, height):
            rgb = np.stack(
                [np.asarray(Image.fromarray(rgb[..., i]).resize(size, Image.Resampling.BOX)) for i in range(3)],
                axis=-1,
            )
        bounded = np.clip(rgb, 0, 1)
        encoded = np.where(bounded <= 0.0031308, 12.92 * bounded, 1.055 * bounded ** (1 / 2.4) - 0.055)
        pixels = np.rint(encoded * 255).astype(np.uint8)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        Image.fromarray(pixels).save(path, format="PNG", icc_profile=profile)
        with Image.open(path) as reopened:
            if reopened.mode != "RGB" or not np.array_equal(np.asarray(reopened), pixels):
                raise HardValidationError("Preview PNG pixels differ")
        return path
