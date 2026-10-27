"""Independent LibRaw hard gate required by PRD_v3's RAW milestone."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthraw.dng_metadata import DNGFieldRecord


class HardValidationError(RuntimeError):
    """An exported RAW failed independent decoding or metadata validation."""


class LibRawValidator:
    """Reopen through lazy-loaded rawpy and reject any RAW contract mismatch."""

    def validate(
        self,
        path: str | Path,
        mosaic: np.ndarray,
        records: dict[str, DNGFieldRecord],
        *,
        develop: bool = False,
    ) -> None:
        """Compare decoded metadata and bounded pixel strips; optionally develop RGB."""
        try:
            import rawpy

            def require(condition: bool, message: str) -> None:
                if not condition:
                    raise HardValidationError(message)

            with rawpy.imread(str(path)) as raw:
                expected_shape = (records["ImageLength"].value, records["ImageWidth"].value)
                require(raw.raw_image.shape == mosaic.shape == expected_shape, "RAW dimensions differ")
                require(
                    (raw.sizes.raw_height, raw.sizes.raw_width) == expected_shape,
                    "RAW sensor dimensions differ",
                )
                # LibRaw distinguishes green channels as RGBG indices 1 and 3.
                pattern = np.array([raw.color_desc[int(i)] for i in raw.raw_pattern.ravel()])
                expected = np.array([b"RGB"[i] for i in records["CFAPattern"].value])
                require(np.array_equal(pattern, expected), "CFA phase differs")
                require(
                    np.allclose(raw.black_level_per_channel, records["BlackLevel"].value, rtol=0, atol=1),
                    "Black level differs",
                )
                require(raw.white_level == records["WhiteLevel"].value, "White level differs")
                wb = np.asarray(raw.camera_whitebalance[:3])
                neutral = np.asarray(records["AsShotNeutral"].value)
                require(np.allclose(wb / wb[1], neutral[1] / neutral, atol=1e-5), "White balance differs")
                # LibRaw normalizes camera-to-RGB rows before inversion. Its
                # daylight DNG conversion uses ColorMatrix2 (D65 in this schema).
                rgb_xyz = np.array(
                    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
                )
                camera_rgb = np.asarray(records["ColorMatrix2"].value).reshape(3, 3) @ rgb_xyz
                camera_rgb /= camera_rgb.sum(axis=1, keepdims=True)
                expected_matrix = np.linalg.inv(camera_rgb)
                require(
                    np.allclose(raw.color_matrix[:, :3], expected_matrix, rtol=1e-4, atol=1e-4),
                    "Colour matrix differs",
                )
                for start in range(0, mosaic.shape[0], 64):
                    require(
                        np.array_equal(raw.raw_image[start : start + 64], mosaic[start : start + 64]),
                        "RAW pixels differ",
                    )
                if develop:
                    rgb = raw.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True)
                    require(
                        rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint16 and np.isfinite(rgb).all(),
                        "RAW development failed",
                    )
        except HardValidationError:
            raise
        except Exception as exc:
            raise HardValidationError(f"Independent LibRaw validation failed: {exc}") from exc
