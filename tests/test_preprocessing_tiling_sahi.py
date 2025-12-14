import sys
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

from hylight_powerlines.preprocessing.tiling_sahi import slice_with_sahi


class _FakeSliced:
    def __init__(self) -> None:
        self.images = [np.zeros((2, 3, 3), dtype=np.uint8)]
        self.starting_pixels = [(5, 7)]


def test_slice_with_sahi_uses_sahi_slicing_module(monkeypatch) -> None:
    def fake_slice_image(
        *,
        image: Any,
        slice_width: int,
        slice_height: int,
        overlap_width_ratio: float,
        overlap_height_ratio: float,
    ) -> _FakeSliced:
        _ = image
        assert slice_width == 4
        assert slice_height == 5
        assert overlap_width_ratio == 0.25
        assert overlap_height_ratio == 0.25
        return _FakeSliced()

    sahi_mod = ModuleType("sahi")
    slicing_mod = ModuleType("sahi.slicing")
    slicing_mod.slice_image = fake_slice_image  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sahi", sahi_mod)
    monkeypatch.setitem(sys.modules, "sahi.slicing", slicing_mod)

    img = Image.new("RGB", (10, 10), color=(0, 0, 0))
    tiles = slice_with_sahi(img, slice_w=4, slice_h=5, overlap_ratio=0.25)
    assert len(tiles) == 1
    tile_img, ox, oy = tiles[0]
    assert (ox, oy) == (5, 7)
    assert tile_img.size == (3, 2)
