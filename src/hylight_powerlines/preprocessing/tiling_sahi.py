"""SAHI tiling adapter.

This module isolates the optional SAHI dependency behind a small API.
"""

from typing import Any, cast

from PIL import Image


def slice_with_sahi(
    img: Image.Image,
    *,
    slice_w: int,
    slice_h: int,
    overlap_ratio: float,
) -> list[tuple[Image.Image, int, int]]:
    """Slice `img` into overlapping tiles using SAHI.

    Returns:
        List of (tile_image_rgb, ox, oy) where (ox, oy) is the tile origin in the
        full-image coordinate frame.

    Raises:
        RuntimeError: If SAHI (and its array dependency) is not installed.
    """
    try:
        import numpy as np
        from sahi.slicing import slice_image
    except ImportError as e:
        raise RuntimeError(
            "SAHI slicing requires extra dependencies.\n"
            "Install with:\n"
            "  uv add sahi numpy\n"
            "Or (pip-style):\n"
            "  uv pip install sahi numpy"
        ) from e

    # SAHI's typing is not consistent across versions. Keep the boundary typed as `Any`.
    arr: Any = np.array(img)
    sliced = slice_image(
        image=arr,
        slice_width=slice_w,
        slice_height=slice_h,
        overlap_width_ratio=overlap_ratio,
        overlap_height_ratio=overlap_ratio,
    )

    images = cast(list[Any], sliced.images)
    starting_pixels = cast(list[tuple[int, int]], sliced.starting_pixels)
    tiles: list[tuple[Image.Image, int, int]] = []
    for tile_arr, (ox, oy) in zip(images, starting_pixels, strict=False):
        tiles.append((Image.fromarray(tile_arr).convert("RGB"), int(ox), int(oy)))
    return tiles

