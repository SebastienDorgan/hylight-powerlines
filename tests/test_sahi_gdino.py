from pathlib import Path

import pytest
from PIL import Image

from hylight_powerlines.pipelines.sahi_gdino import (
    SahiGdinoParams,
    SahiGdinoRun,
    SahiTiling,
    detect_multiresolution,
    detect_on_full_image,
    detect_on_slices,
    run_sahi_gdino,
    run_sahi_gdino_batch,
)
from hylight_powerlines.vision.labels import combined_prompt, normalize_label
from hylight_powerlines.vision.types import Box


class _FakeGdino:
    def __init__(self, boxes_per_call: list[list[Box]]):
        self._boxes_per_call = boxes_per_call
        self.calls: list[tuple[str, float, float]] = []

    def detect(
        self,
        img: Image.Image,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> list[Box]:
        _ = img
        self.calls.append((prompt, box_threshold, text_threshold))
        if not self._boxes_per_call:
            return []
        return self._boxes_per_call.pop(0)


def test_combined_prompt() -> None:
    assert combined_prompt(["tower", "insulator"]) == "tower . insulator"


@pytest.mark.parametrize(
    ("raw", "targets", "expected"),
    [
        ("tower", ["tower", "insulator"], "tower"),
        ("Tower", ["tower", "insulator"], "tower"),
        ("tower/pylon", ["tower", "insulator"], "tower"),
        ("tower plate", ["tower", "tower_plate"], "tower_plate"),
        ("", ["tower"], None),
        ("unknown", ["tower"], None),
    ],
)
def test_normalize_label(raw: str, targets: list[str], expected: str | None) -> None:
    assert normalize_label(raw, targets) == expected


def test_detect_on_slices_offsets_and_nms_without_sahi_dependency(tmp_path: Path) -> None:
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))

    # Two tiles; second tile is offset by +10px in x/y.
    def tile_provider(_img: Image.Image, _tiling: SahiTiling):
        tile = Image.new("RGB", (60, 60), color=(0, 0, 0))
        return [(tile, 0, 0), (tile, 10, 10)]

    # Boxes are in tile coordinates; after offset, these overlap heavily.
    gdino = _FakeGdino(
        boxes_per_call=[
            [Box(label="tower/pylon", x1=10, y1=10, x2=40, y2=40, score=0.6, source="gdino")],
            [Box(label="tower", x1=0, y1=0, x2=30, y2=30, score=0.9, source="gdino")],
        ]
    )

    out = detect_on_slices(
        img,
        targets=["tower"],
        gdino=gdino,
        tiling=SahiTiling(slice_w=60, slice_h=60, overlap_ratio=0.2),
        gdino_params=SahiGdinoParams(
            box_thr=0.2,
            text_thr=0.2,
            max_dets_per_tile=30,
            min_score=0.0,
            min_area_frac=0.0,
            max_area_frac=1.0,
            max_aspect=100.0,
            min_support=1,
        ),
        nms_iou=0.5,
        tile_provider=tile_provider,
    )

    # NMS keeps the higher score one (from the second tile).
    assert len(out) == 1
    assert out[0].label == "tower"
    assert out[0].score == pytest.approx(0.9)

    # Combined prompt used for both calls.
    assert gdino.calls[0][0].startswith("tower")
    assert gdino.calls[1][0].startswith("tower")


def test_detect_on_full_image_scales_boxes() -> None:
    img = Image.new("RGB", (200, 100), color=(0, 0, 0))

    class _ScalingGdino:
        def __init__(self):
            self.seen_sizes: list[tuple[int, int]] = []

        def detect(
            self,
            img: Image.Image,
            prompt: str,
            *,
            box_threshold: float,
            text_threshold: float,
        ) -> list[Box]:
            _ = prompt, box_threshold, text_threshold
            self.seen_sizes.append(img.size)
            # Box is in resized coords: (10..60, 5..45)
            return [Box(label="tower", x1=10, y1=5, x2=60, y2=45, score=0.9, source="gdino")]

    gdino = _ScalingGdino()
    params = SahiGdinoParams(
        box_thr=0.2,
        text_thr=0.2,
        global_max_side=100,  # forces resize from (200,100) -> (100,50)
        max_dets_per_tile=30,
        min_score=0.0,
        min_area_frac=0.0,
        max_area_frac=1.0,
        max_aspect=100.0,
        min_support=1,
    )
    out = detect_on_full_image(
        img,
        targets=["tower"],
        gdino=gdino,
        gdino_params=params,
        global_max_side=100,
    )
    assert gdino.seen_sizes == [(100, 50)]
    # scaled back by sx=2, sy=2
    assert len(out) == 1
    assert (out[0].x1, out[0].y1, out[0].x2, out[0].y2) == (20.0, 10.0, 120.0, 90.0)


def test_detect_multiresolution_auto_global_max_side_uses_full_image() -> None:
    img = Image.new("RGB", (200, 100), color=(0, 0, 0))

    class _SizeTrackingGdino:
        def __init__(self):
            self.seen_sizes: list[tuple[int, int]] = []

        def detect(
            self,
            img: Image.Image,
            prompt: str,
            *,
            box_threshold: float,
            text_threshold: float,
        ) -> list[Box]:
            _ = prompt, box_threshold, text_threshold
            self.seen_sizes.append(img.size)
            return []

    def tile_provider(_img: Image.Image, _tiling: SahiTiling):
        tile = Image.new("RGB", (60, 60), color=(0, 0, 0))
        return [(tile, 0, 0)]

    gdino = _SizeTrackingGdino()
    detect_multiresolution(
        img,
        targets=["tower"],
        gdino=gdino,
        tilings=[SahiTiling(slice_w=60, slice_h=60, overlap_ratio=0.2, require_support=False)],
        gdino_params=SahiGdinoParams(global_max_side=0, min_support=1),
        nms_iou=0.5,
        tile_provider=tile_provider,
    )

    # The first call is the global pass; with auto global_max_side it should not
    # downscale below the original image size.
    assert gdino.seen_sizes[0] == (200, 100)


def test_detect_multiresolution_no_anchor_low_scores_returns_empty() -> None:
    img = Image.new("RGB", (200, 100), color=(0, 0, 0))

    class _Gdino:
        def __init__(self):
            self.calls = 0

        def detect(
            self,
            img: Image.Image,
            prompt: str,
            *,
            box_threshold: float,
            text_threshold: float,
        ) -> list[Box]:
            _ = img, prompt, box_threshold, text_threshold
            self.calls += 1
            if self.calls == 1:  # global pass
                return [
                    Box(
                        label="insulator",
                        x1=10,
                        y1=10,
                        x2=40,
                        y2=40,
                        score=0.54,
                        source="gdino",
                    )
                ]
            return []

    def tile_provider(_img: Image.Image, _tiling: SahiTiling):
        tile = Image.new("RGB", (60, 60), color=(0, 0, 0))
        return [(tile, 0, 0)]

    global_boxes, tiled, merged = detect_multiresolution(
        img,
        targets=["tower", "insulator"],
        gdino=_Gdino(),
        tilings=[SahiTiling(slice_w=60, slice_h=60, overlap_ratio=0.2, require_support=False)],
        gdino_params=SahiGdinoParams(
            min_score=0.0,
            min_area_frac=0.0,
            max_area_frac=1.0,
            max_aspect=100.0,
            parts_max_area_frac=1.0,
            parts_max_aspect=100.0,
            min_support=1,
            anchor_min_score=0.99,  # force "no anchor"
            no_anchor_min_keep_score=0.55,
        ),
        nms_iou=0.5,
        tile_provider=tile_provider,
    )
    assert global_boxes
    assert tiled == []
    assert merged == []


def test_detect_multiresolution_clutter_requires_cross_group_support() -> None:
    img = Image.new("RGB", (200, 100), color=(0, 0, 0))

    class _Gdino:
        def __init__(self):
            self.calls = 0

        def detect(
            self,
            img: Image.Image,
            prompt: str,
            *,
            box_threshold: float,
            text_threshold: float,
        ) -> list[Box]:
            _ = img, prompt, box_threshold, text_threshold
            self.calls += 1
            if self.calls == 1:  # global pass: none
                return []
            if self.calls == 2:  # tiling1: lots of hallucinations
                return [
                    Box(
                        label="insulator",
                        x1=10 + i,
                        y1=10,
                        x2=20 + i,
                        y2=20,
                        score=0.8,
                        source="gdino",
                    )
                    for i in range(10)
                ]
            return []  # tiling2: none

    def tile_provider(_img: Image.Image, _tiling: SahiTiling):
        tile = Image.new("RGB", (60, 60), color=(0, 0, 0))
        return [(tile, 0, 0)]

    _, _, merged = detect_multiresolution(
        img,
        targets=["tower", "insulator"],
        gdino=_Gdino(),
        tilings=[
            SahiTiling(slice_w=60, slice_h=60, overlap_ratio=0.2, require_support=False),
            SahiTiling(slice_w=80, slice_h=80, overlap_ratio=0.2, require_support=False),
        ],
        gdino_params=SahiGdinoParams(
            min_score=0.0,
            min_area_frac=0.0,
            max_area_frac=1.0,
            max_aspect=100.0,
            parts_max_area_frac=1.0,
            parts_max_aspect=100.0,
            min_support=1,
            anchor_min_score=0.99,  # force "no anchor"
            clutter_max_boxes=1,  # trigger clutter mode
            clutter_rescue_score=0.95,
            clutter_support_iou=0.2,
            max_boxes_per_image_no_anchor=50,
        ),
        nms_iou=0.5,
        tile_provider=tile_provider,
    )
    assert merged == []


def test_detect_multiresolution_clutter_keeps_cross_supported_boxes() -> None:
    img = Image.new("RGB", (200, 100), color=(0, 0, 0))

    class _Gdino:
        def __init__(self):
            self.calls = 0

        def detect(
            self,
            img: Image.Image,
            prompt: str,
            *,
            box_threshold: float,
            text_threshold: float,
        ) -> list[Box]:
            _ = img, prompt, box_threshold, text_threshold
            self.calls += 1
            if self.calls == 1:  # global pass: none
                return []
            if self.calls == 2:  # tiling1
                return [
                    Box(
                        label="insulator",
                        x1=10,
                        y1=10,
                        x2=40,
                        y2=40,
                        score=0.8,
                        source="gdino",
                    )
                ]
            if self.calls == 3:  # tiling2 sees same object
                return [
                    Box(
                        label="insulator",
                        x1=12,
                        y1=12,
                        x2=42,
                        y2=42,
                        score=0.75,
                        source="gdino",
                    )
                ]
            return []

    def tile_provider(_img: Image.Image, _tiling: SahiTiling):
        tile = Image.new("RGB", (60, 60), color=(0, 0, 0))
        return [(tile, 0, 0)]

    _, _, merged = detect_multiresolution(
        img,
        targets=["tower", "insulator"],
        gdino=_Gdino(),
        tilings=[
            SahiTiling(slice_w=60, slice_h=60, overlap_ratio=0.2, require_support=False),
            SahiTiling(slice_w=80, slice_h=80, overlap_ratio=0.2, require_support=False),
        ],
        gdino_params=SahiGdinoParams(
            min_score=0.0,
            min_area_frac=0.0,
            max_area_frac=1.0,
            max_aspect=100.0,
            parts_max_area_frac=1.0,
            parts_max_aspect=100.0,
            min_support=1,
            anchor_min_score=0.99,  # force "no anchor"
            clutter_max_boxes=1,  # trigger clutter mode
            clutter_rescue_score=0.95,
            clutter_support_iou=0.2,
            max_boxes_per_image_no_anchor=50,
        ),
        nms_iou=0.5,
        tile_provider=tile_provider,
    )
    assert len(merged) == 1
    assert merged[0].label == "insulator"


def test_run_sahi_gdino_writes_outputs_and_reuses_final_json(tmp_path: Path, monkeypatch) -> None:
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 24), color=(0, 0, 0)).save(img_path)

    # Avoid SAHI dependency by patching tiling at the pipeline module boundary.
    def fake_slice_with_sahi(
        _img: Image.Image,
        *,
        slice_w: int,
        slice_h: int,
        overlap_ratio: float,
    ):
        _ = slice_w, slice_h, overlap_ratio
        tile = Image.new("RGB", (16, 12), color=(0, 0, 0))
        return [(tile, 0, 0)]

    import hylight_powerlines.pipelines.sahi_gdino as mod

    monkeypatch.setattr(mod, "slice_with_sahi", fake_slice_with_sahi)

    gdino = _FakeGdino(
        boxes_per_call=[
            [Box(label="tower/pylon", x1=1, y1=2, x2=10, y2=12, score=0.9, source="gdino")]
        ]
    )

    outdir = tmp_path / "out"
    cfg = SahiGdinoRun(
        image_path=img_path,
        outdir=outdir,
        targets=["tower"],
        tilings=(SahiTiling(slice_w=16, slice_h=12, overlap_ratio=0.2),),
        gdino_params=SahiGdinoParams(
            box_thr=0.2,
            text_thr=0.2,
            max_dets_per_tile=30,
            min_score=0.0,
            min_area_frac=0.0,
            max_area_frac=1.0,
            max_aspect=100.0,
            min_support=1,
        ),
        nms_iou=0.5,
        overwrite=True,
        verbose=False,
    )

    payload1 = run_sahi_gdino(cfg, gdino=gdino)
    assert (outdir / "02_gdino_boxes.jpg").is_file()
    assert (outdir / "final.json").is_file()
    assert (outdir / "img.txt").is_file()
    assert payload1["final_boxes"][0]["label"] == "tower"

    # Second run without overwrite should hit the "reuse final.json" fast path.
    cfg2 = SahiGdinoRun(
        image_path=img_path,
        outdir=outdir,
        targets=["tower"],
        tilings=(SahiTiling(slice_w=16, slice_h=12, overlap_ratio=0.2),),
        overwrite=False,
        verbose=False,
    )
    payload2 = run_sahi_gdino(cfg2, gdino=gdino)
    assert payload2 == payload1


def test_run_sahi_gdino_batch_records_failures(tmp_path: Path, monkeypatch) -> None:
    good = tmp_path / "ok.jpg"
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(good)
    bad = tmp_path / "missing.jpg"

    # Use a GDINO that produces no detections; we only care about error handling.
    gdino = _FakeGdino(boxes_per_call=[[]])

    import hylight_powerlines.pipelines.sahi_gdino as mod

    def fake_slice_with_sahi(
        _img: Image.Image,
        *,
        slice_w: int,
        slice_h: int,
        overlap_ratio: float,
    ):
        _ = slice_w, slice_h, overlap_ratio
        tile = Image.new("RGB", (8, 8), color=(0, 0, 0))
        return [(tile, 0, 0)]

    monkeypatch.setattr(mod, "slice_with_sahi", fake_slice_with_sahi)

    summary, failures = run_sahi_gdino_batch(
        images=[good, bad],
        out_root=tmp_path / "out_root",
        targets=["tower"],
        gdino=gdino,
        tilings=[SahiTiling(slice_w=8, slice_h=8, overlap_ratio=0.2)],
        gdino_params=SahiGdinoParams(
            box_thr=0.2,
            text_thr=0.2,
            max_dets_per_tile=30,
            min_score=0.0,
            min_area_frac=0.0,
            max_area_frac=1.0,
            max_aspect=100.0,
            min_support=1,
        ),
        nms_iou=0.5,
        overwrite=True,
        verbose=False,
    )

    assert failures == 1
    assert (tmp_path / "out_root" / "summary.yaml").is_file()
    assert len(summary) == 2
    assert any("error" in item for item in summary)
