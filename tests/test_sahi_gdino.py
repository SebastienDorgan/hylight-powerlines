from pathlib import Path

import pytest
from PIL import Image

from hylight_powerlines.pipelines.sahi_gdino import (
    SahiGdinoParams,
    SahiGdinoRun,
    SahiTiling,
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
        gdino_params=SahiGdinoParams(box_thr=0.2, text_thr=0.2),
        nms_iou=0.5,
        tile_provider=tile_provider,
    )

    # NMS keeps the higher score one (from the second tile).
    assert len(out) == 1
    assert out[0].label == "tower"
    assert out[0].score == pytest.approx(0.9)

    # Combined prompt used for both calls.
    assert gdino.calls[0][0] == "tower"
    assert gdino.calls[1][0] == "tower"


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
        tiling=SahiTiling(slice_w=16, slice_h=12, overlap_ratio=0.2),
        gdino_params=SahiGdinoParams(box_thr=0.2, text_thr=0.2),
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
        tiling=SahiTiling(slice_w=8, slice_h=8, overlap_ratio=0.2),
        gdino_params=SahiGdinoParams(box_thr=0.2, text_thr=0.2),
        nms_iou=0.5,
        overwrite=True,
        verbose=False,
    )

    assert failures == 1
    assert (tmp_path / "out_root" / "summary.yaml").is_file()
    assert len(summary) == 2
    assert any("error" in item for item in summary)
