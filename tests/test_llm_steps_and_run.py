from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hylight_powerlines.llm.models import Box, PipelineConfig, PreAnalysis, Proposal
from hylight_powerlines.llm.pipeline import run_pipeline
from hylight_powerlines.llm.steps import refine_with_gdino, refine_with_sam2


class _FakeGdino:
    def __init__(self, boxes: list[Box]):
        self._boxes = boxes
        self.calls: list[tuple[str, float, float]] = []

    def detect(
        self,
        img: Image.Image,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> list[Box]:
        self.calls.append((prompt, box_threshold, text_threshold))
        # Boxes are expressed in the crop coordinate frame already.
        return list(self._boxes)


class _FakeSam2:
    def __init__(self, mask: np.ndarray, score: float = 0.9):
        self.mask = mask
        self.score = score
        self.calls: list[Box] = []

    def segment_from_box(self, img: Image.Image, box: Box) -> tuple[np.ndarray, float]:
        self.calls.append(box)
        return self.mask, self.score


def test_refine_with_gdino_selects_best_match_and_keeps_vlm_if_missing() -> None:
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    prop = Proposal(
        label="tower",
        box=Box(label="tower", x1=10, y1=10, x2=30, y2=30, score=0.5, source="vlm"),
        prompt_variants=["tower", "pylon"],
    )
    pre = PreAnalysis(image_w=100, image_h=100, proposals=[prop])

    # In ROI crop coordinates; the crop offset should be (0,0) because expand_box clips to bounds.
    gdino = _FakeGdino(
        [
            Box(label="tower", x1=8, y1=8, x2=28, y2=28, score=0.2, source="gdino"),
            Box(label="tower", x1=10, y1=10, x2=31, y2=31, score=0.9, source="gdino"),
        ]
    )
    refined = refine_with_gdino(
        img=img,
        pre=pre,
        gdino=gdino,
        roi_scale=2.0,
        box_thr=0.2,
        text_thr=0.2,
        nms_iou_thr=0.5,
        keep_vlm_if_missing=False,
        debug_dir=None,
    )
    assert len(refined) == 1
    assert refined[0].source == "gdino"
    assert refined[0].score == pytest.approx(0.9)
    assert gdino.calls[0][0] == "tower . pylon"

    gdino_missing = _FakeGdino([])
    refined2 = refine_with_gdino(
        img=img,
        pre=pre,
        gdino=gdino_missing,
        roi_scale=2.0,
        box_thr=0.2,
        text_thr=0.2,
        nms_iou_thr=0.5,
        keep_vlm_if_missing=True,
        debug_dir=None,
    )
    assert len(refined2) == 1
    assert refined2[0].source == "vlm"


def test_refine_with_sam2_filters_small_masks() -> None:
    img = Image.new("RGB", (20, 20), color=(0, 0, 0))
    det = Box(label="tower", x1=0, y1=0, x2=10, y2=10, score=0.9, source="gdino")

    small_mask = np.zeros((20, 20), dtype=np.uint8)
    small_mask[0:2, 0:2] = 255
    sam2_small = _FakeSam2(small_mask, score=0.9)
    out_boxes, out_masks = refine_with_sam2(img=img, dets=[det], sam2=sam2_small, min_mask_area=10)
    assert out_boxes == []
    assert out_masks == []

    big_mask = np.zeros((20, 20), dtype=np.uint8)
    big_mask[3:10, 4:12] = 255
    sam2_big = _FakeSam2(big_mask, score=0.4)
    out_boxes2, out_masks2 = refine_with_sam2(img=img, dets=[det], sam2=sam2_big, min_mask_area=10)
    assert len(out_boxes2) == 1
    assert out_boxes2[0].source == "sam2"
    assert out_boxes2[0].score == pytest.approx(0.4)
    assert out_masks2[0].shape == (20, 20)


def test_run_pipeline_with_pre_json_and_injected_backends(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 32), color=(0, 0, 0)).save(img_path)

    pre_json = tmp_path / "pre.json"
    pre_json.write_text(
        json.dumps(
            {
                "image_w": 64,
                "image_h": 32,
                "proposals": [
                    {
                        "label": "tower",
                        "box": {"x1": 5, "y1": 5, "x2": 20, "y2": 20},
                        "confidence": 0.9,
                        "prompt_variants": ["tower"],
                        "notes": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    outdir = tmp_path / "out"
    det = Box(label="tower", x1=6, y1=6, x2=19, y2=19, score=0.8, source="gdino")
    gdino = _FakeGdino([det])

    mask = np.zeros((32, 64), dtype=np.uint8)
    mask[6:19, 6:19] = 255
    sam2 = _FakeSam2(mask, score=0.7)

    cfg = PipelineConfig(
        image_path=img_path,
        outdir=outdir,
        targets=["tower"],
        vlm_model="fake",
        pre_json=pre_json,
        use_sam2=True,
        sam2_config="cfg.yaml",
        sam2_ckpt="ckpt.pt",
    )

    res = run_pipeline(
        cfg,
        gdino_factory=lambda _mid, _dev: gdino,
        sam2_factory=lambda _cfg, _ckpt, _dev: sam2,
    )
    assert res.image == img_path
    assert len(res.boxes) == 1

    assert (outdir / "preanalysis.json").is_file()
    assert (outdir / "01_vlm_boxes.jpg").is_file()
    assert (outdir / "02_gdino_boxes.jpg").is_file()
    assert (outdir / "03_sam2_boxes.jpg").is_file()
    assert (outdir / "final.json").is_file()
    assert (outdir / f"{img_path.stem}.txt").is_file()


def test_run_pipeline_requires_sam2_paths_when_enabled(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(img_path)

    cfg = PipelineConfig(
        image_path=img_path,
        outdir=tmp_path / "out",
        targets=["tower"],
        vlm_model="fake",
        pre_json=None,
        use_sam2=True,
        sam2_config="",
        sam2_ckpt="",
    )

    with pytest.raises(RuntimeError, match="--use_sam2 requires"):
        run_pipeline(
            cfg,
            pre_analyzer=lambda **_: PreAnalysis(10, 10, []),
            gdino_factory=lambda *_: _FakeGdino([]),
        )
