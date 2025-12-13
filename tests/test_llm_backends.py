from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
from PIL import Image

from hylight_powerlines.llm.gdino import GroundingDinoHF
from hylight_powerlines.llm.models import Box
from hylight_powerlines.llm.sam2 import Sam2Segmenter


class _FakeTorchCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    cuda = _FakeTorchCuda()

    @staticmethod
    @contextmanager
    def inference_mode():
        yield

    @staticmethod
    def tensor(data, device: str | None = None):
        return {"data": data, "device": device}


class _ToDevice:
    def __init__(self, value: Any):
        self.value = value

    def to(self, device: str) -> _ToDevice:
        _ = device
        return self


class _Arr:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self) -> _Arr:
        return self

    def cpu(self) -> _Arr:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeProcessor:
    def __call__(self, *, images: Image.Image, text: str, return_tensors: str) -> dict[str, Any]:
        _ = images
        _ = text
        _ = return_tensors
        return {"input_ids": _ToDevice([1, 2, 3]), "pixel_values": _ToDevice("pv")}

    def post_process_grounded_object_detection(
        self,
        *,
        outputs: Any,
        input_ids: Any,
        box_threshold: float,
        text_threshold: float,
        target_sizes: Any,
    ) -> list[dict[str, Any]]:
        _ = outputs, input_ids, box_threshold, text_threshold, target_sizes
        return [
            {
                "boxes": _Arr(np.array([[1.0, 2.0, 9.0, 10.0]], dtype=np.float32)),
                "scores": _Arr(np.array([0.75], dtype=np.float32)),
                "labels": ["tower"],
            }
        ]


class _FakeProcessorThreshold:
    def __call__(self, *, images: Image.Image, text: str, return_tensors: str) -> dict[str, Any]:
        _ = images
        _ = text
        _ = return_tensors
        return {"input_ids": _ToDevice([1, 2, 3]), "pixel_values": _ToDevice("pv")}

    def post_process_grounded_object_detection(
        self,
        *,
        outputs: Any,
        input_ids: Any,
        threshold: float,
        text_threshold: float,
        target_sizes: Any,
    ) -> list[dict[str, Any]]:
        _ = outputs, input_ids, threshold, text_threshold, target_sizes
        return [
            {
                "boxes": _Arr(np.array([[1.0, 2.0, 9.0, 10.0]], dtype=np.float32)),
                "scores": _Arr(np.array([0.75], dtype=np.float32)),
                "labels": ["tower"],
            }
        ]


class _FakeModel:
    def to(self, device: str) -> None:
        _ = device

    def eval(self) -> None:
        return

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {"ok": True}


def test_grounding_dino_hf_detect_with_injected_components() -> None:
    gdino = GroundingDinoHF(
        model_id="unused",
        device="auto",
        torch_module=_FakeTorch,
        processor=_FakeProcessor(),
        model=_FakeModel(),
    )
    img = Image.new("RGB", (20, 20), color=(0, 0, 0))
    boxes = gdino.detect(img=img, prompt="tower", box_threshold=0.2, text_threshold=0.2)
    assert boxes == [
        Box(
            label="tower",
            x1=1.0,
            y1=2.0,
            x2=9.0,
            y2=10.0,
            score=0.75,
            source="gdino",
            prompt="tower",
        )
    ]


def test_grounding_dino_hf_detect_with_threshold_keyword() -> None:
    gdino = GroundingDinoHF(
        model_id="unused",
        device="auto",
        torch_module=_FakeTorch,
        processor=_FakeProcessorThreshold(),
        model=_FakeModel(),
    )
    img = Image.new("RGB", (20, 20), color=(0, 0, 0))
    boxes = gdino.detect(img=img, prompt="tower", box_threshold=0.2, text_threshold=0.2)
    assert boxes == [
        Box(
            label="tower",
            x1=1.0,
            y1=2.0,
            x2=9.0,
            y2=10.0,
            score=0.75,
            source="gdino",
            prompt="tower",
        )
    ]


class _FakeSamPredictor:
    def __init__(self, masks: np.ndarray, scores: np.ndarray):
        self._masks = masks
        self._scores = scores
        self.images: list[np.ndarray] = []

    def set_image(self, arr: np.ndarray) -> None:
        self.images.append(arr)

    def predict(
        self, *, box: np.ndarray, multimask_output: bool
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        _ = box, multimask_output
        return self._masks, self._scores, None


def test_sam2_segmenter_with_injected_predictor() -> None:
    masks = np.zeros((1, 5, 6), dtype=np.uint8)
    masks[0, 1:3, 2:5] = 1
    scores = np.array([0.5], dtype=np.float32)

    seg = Sam2Segmenter(
        sam2_config="unused",
        sam2_ckpt="unused",
        device="auto",
        torch_module=_FakeTorch,
        predictor=_FakeSamPredictor(masks=masks, scores=scores),
    )
    img = Image.new("RGB", (6, 5), color=(0, 0, 0))
    m, sc = seg.segment_from_box(
        img=img,
        box=Box(label="tower", x1=0, y1=0, x2=5, y2=4, score=1.0, source="x"),
    )
    assert sc == 0.5
    assert m.dtype == np.uint8
    assert int(m.max()) == 255
