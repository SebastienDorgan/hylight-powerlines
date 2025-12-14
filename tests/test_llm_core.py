from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from hylight_powerlines.llm.geometry import expand_box, iou, nms, tight_box_from_mask
from hylight_powerlines.llm.image import crop_with_box, img_to_b64_jpeg
from hylight_powerlines.llm.models import Box
from hylight_powerlines.llm.vis import draw_boxes
from hylight_powerlines.llm.vlm import _extract_json, pre_analyze
from hylight_powerlines.llm.yolo_export import export_yolo


def test_box_area_and_clip() -> None:
    b = Box(label="x", x1=10, y1=20, x2=30, y2=50, score=0.9, source="t")
    assert b.area() == 20 * 30

    b2 = Box(label="x", x1=50, y1=40, x2=-5, y2=-10, score=0.0, source="t").clip(32, 32)
    assert 0 <= b2.x1 <= b2.x2 <= 31
    assert 0 <= b2.y1 <= b2.y2 <= 31


def test_geometry_iou_nms_expand_and_tight_box() -> None:
    a = Box(label="a", x1=0, y1=0, x2=10, y2=10, score=0.9, source="t")
    b = Box(label="b", x1=5, y1=5, x2=15, y2=15, score=0.8, source="t")
    assert iou(a, b) == pytest.approx(25 / (100 + 100 - 25))

    kept = nms([a, b], iou_thr=0.1)
    assert kept == [a]

    ex = expand_box(a, w=20, h=20, scale=2.0)
    assert ex.x1 == 0.0
    assert ex.y1 == 0.0
    assert ex.x2 == 15.0
    assert ex.y2 == 15.0

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    assert tight_box_from_mask(mask) == (3, 2, 7, 5)


def test_image_to_base64_and_crop() -> None:
    img = Image.new("RGB", (20, 10), color=(255, 0, 0))
    b64 = img_to_b64_jpeg(img)
    # basic sanity: decodable base64
    assert base64.b64decode(b64)

    box = Box(label="x", x1=2, y1=3, x2=12, y2=8, score=1.0, source="t")
    crop, (ox, oy) = crop_with_box(img, box)
    assert crop.size == (10, 5)
    assert (ox, oy) == (2, 3)


def test_extract_json_from_messy_text() -> None:
    payload = {"a": 1, "b": {"c": 2}}
    s = f"prefix {json.dumps(payload)} trailing"
    assert json.loads(_extract_json(s)) == payload


def test_pre_analyze_with_injected_completion() -> None:
    img = Image.new("RGB", (100, 50), color=(0, 0, 0))

    def fake_responses(**_: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "label": "tower",
                                            "box": {"x1": 0.01, "y1": 0.04, "x2": 0.30, "y2": 0.80},
                                            "confidence": 0.7,
                                            "prompt_variants": ["tower", "pylon"],
                                            "notes": "",
                                        },
                                        # ignored: not in targets
                                        {
                                            "label": "not_a_target",
                                            "box": {"x1": 0.0, "y1": 0.0, "x2": 0.01, "y2": 0.01},
                                            "confidence": 1.0,
                                            "prompt_variants": ["x"],
                                            "notes": "",
                                        },
                                    ],
                                }
                            ),
                        }
                    ],
                }
            ],
        }

    import hylight_powerlines.llm.vlm as vlm

    old = vlm.litellm.responses
    vlm.litellm.responses = fake_responses  # type: ignore[assignment]
    try:
        pre = pre_analyze(
            img=img,
            targets=["tower"],
            model="openai/fake",
        )
    finally:
        vlm.litellm.responses = old  # type: ignore[assignment]
    assert pre.image_w == 100
    assert pre.image_h == 50
    assert [p.label for p in pre.proposals] == ["tower"]
    assert pre.proposals[0].prompt_variants == ["pylon", "tower"]  # sorted/deduped
    b = pre.proposals[0].box
    assert (b.x1, b.y1, b.x2, b.y2) == (1.0, 2.0, 30.0, 40.0)


def test_pre_analyze_accepts_pydantic_completion_model() -> None:
    from pydantic import BaseModel

    img = Image.new("RGB", (100, 50), color=(0, 0, 0))

    class _Resp(BaseModel):
        status: str
        output: list[dict[str, Any]]

    def fake_completion(**_: Any) -> _Resp:
        return _Resp(
            status="completed",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "label": "tower",
                                            "box": {"x1": 0.01, "y1": 0.04, "x2": 0.30, "y2": 0.80},
                                            "confidence": 0.7,
                                            "prompt_variants": ["tower", "pylon"],
                                            "notes": "",
                                        }
                                    ],
                                }
                            ),
                        }
                    ],
                }
            ],
        )

    import hylight_powerlines.llm.vlm as vlm

    old = vlm.litellm.responses
    vlm.litellm.responses = fake_completion  # type: ignore[assignment]
    try:
        pre = pre_analyze(
            img=img,
            targets=["tower"],
            model="openai/fake",
        )
    finally:
        vlm.litellm.responses = old  # type: ignore[assignment]
    assert [p.label for p in pre.proposals] == ["tower"]
    b = pre.proposals[0].box
    assert (b.x1, b.y1, b.x2, b.y2) == (1.0, 2.0, 30.0, 40.0)


def test_pre_analyze_scales_normalized_boxes() -> None:
    img = Image.new("RGB", (100, 50), color=(0, 0, 0))

    def fake_responses(**_: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "proposals": [
                                        {
                                            "label": "tower",
                                            "box": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
                                            "confidence": 0.7,
                                            "prompt_variants": ["tower"],
                                            "notes": "",
                                        }
                                    ],
                                }
                            ),
                        }
                    ],
                }
            ],
        }

    import hylight_powerlines.llm.vlm as vlm

    old = vlm.litellm.responses
    vlm.litellm.responses = fake_responses  # type: ignore[assignment]
    try:
        pre = pre_analyze(img=img, targets=["tower"], model="openai/fake")
    finally:
        vlm.litellm.responses = old  # type: ignore[assignment]
    b = pre.proposals[0].box
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 10.0, 30.0, 20.0)


def test_export_yolo_writes_normalized_labels(tmp_path: Path) -> None:
    out = tmp_path / "a.txt"
    boxes = [
        Box(label="tower", x1=0, y1=0, x2=10, y2=10, score=1.0, source="x"),
        Box(label="unknown", x1=0, y1=0, x2=1, y2=1, score=1.0, source="x"),
    ]
    export_yolo(boxes=boxes, image_w=20, image_h=10, class_names=["tower"], out_txt=out)
    assert out.read_text(encoding="utf-8").strip() == "0 0.250000 0.500000 0.500000 1.000000"


def test_draw_boxes_modifies_pixels(tmp_path: Path) -> None:
    img = Image.new("RGB", (500, 400), color=(0, 0, 0))
    out_path = tmp_path / "vis.jpg"
    draw_boxes(
        img,
        [Box(label="tower", x1=10, y1=20, x2=200, y2=300, score=0.9, source="gdino")],
        out_path,
    )
    assert out_path.is_file()
    orig = np.array(img)
    vis = np.array(Image.open(out_path).convert("RGB"))
    assert int(np.abs(orig.astype(np.int16) - vis.astype(np.int16)).sum()) > 0
