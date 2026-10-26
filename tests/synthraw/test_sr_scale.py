"""Arithmetic-only coverage of PRD_v3's 2K/4K SR scale selection."""

import pytest

from synthraw.super_resolution import SuperResolutionEngine


@pytest.mark.parametrize(
    "source,target,scale,intermediate",
    [
        ((2048, 868), (10872, 4608), 4, (8192, 3472)),
        ((4096, 1736), (10872, 4608), 2, (8192, 3472)),
        ((868, 2048), (4608, 10872), 4, (3472, 8192)),
        ((1736, 4096), (4608, 10872), 2, (3472, 8192)),
        ((4096, 2048), (12288, 6144), 4, (16384, 8192)),
        ((4096, 2048), (12287, 6143), 2, (8192, 4096)),
        ((4096, 1024), (8192, 4096), 4, (16384, 4096)),
        ((2048, 1024), (512, 256), 4, (8192, 4096)),
        ((2049, 1024), (512, 256), 2, (4098, 2048)),
    ],
)
def test_scale_selection(source, target, scale, intermediate):
    result = SuperResolutionEngine.plan_scale(
        source_width=source[0], source_height=source[1], content_width=target[0], content_height=target[1]
    )
    assert result.scale == scale
    assert result.intermediate_size == intermediate
    assert result.content_size == target


@pytest.mark.parametrize("field", ["source_width", "source_height", "content_width", "content_height"])
@pytest.mark.parametrize("invalid", [0, -1, True, 2.5, None])
def test_invalid_dimensions(field, invalid):
    dimensions = dict(source_width=4096, source_height=1736, content_width=10872, content_height=4608)
    dimensions[field] = invalid
    with pytest.raises(ValueError, match="positive integers"):
        SuperResolutionEngine.plan_scale(**dimensions)
