from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from hylight_powerlines.yolo import YoloFineTuner, YoloPredictor


class _FakeYolo:
    def __init__(self, name: str, *, train_outputs: dict[str, Any] | None = None):
        self.name = name
        self.train_outputs = train_outputs or {}
        self.train_calls: list[dict[str, Any]] = []
        self.val_calls: list[dict[str, Any]] = []
        self.predict_calls: list[dict[str, Any]] = []

    def train(self, **kwargs: Any) -> dict[str, Any]:
        self.train_calls.append(dict(kwargs))
        run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        best = run_dir / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"weights")
        return self.train_outputs

    def val(self, **kwargs: Any) -> dict[str, Any]:
        self.val_calls.append(dict(kwargs))
        return {"ok": True}

    def predict(self, **kwargs: Any) -> Iterable[int]:
        self.predict_calls.append(dict(kwargs))
        if kwargs.get("stream", False):
            return iter([1, 2, 3])
        return [1, 2, 3]


def test_yolo_finetuner_train_writes_best_ckpt_and_maps_auto_batch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds"
    dataset_root.mkdir(parents=True)
    (dataset_root / "data.yaml").write_text("names: ['a']\n", encoding="utf-8")

    created: list[_FakeYolo] = []

    def factory(name: str) -> _FakeYolo:
        m = _FakeYolo(name)
        created.append(m)
        return m

    tuner = YoloFineTuner(
        dataset_root=dataset_root,
        model_name="yolov8s.pt",
        epochs=1,
        img_size=64,
        batch="auto",
        device="cpu",
        project=str(tmp_path / "runs"),
        run_name="r1",
        model_factory=factory,
    )
    best = tuner.train()
    assert best.is_file()
    assert best.read_bytes() == b"weights"
    assert created[0].train_calls[0]["batch"] == 0


def test_yolo_finetuner_rejects_unknown_batch_string(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds"
    dataset_root.mkdir(parents=True)
    (dataset_root / "data.yaml").write_text("names: ['a']\n", encoding="utf-8")

    tuner = YoloFineTuner(
        dataset_root=dataset_root,
        batch="AUTO",  # invalid
        model_factory=lambda _: _FakeYolo("m"),
    )
    with pytest.raises(ValueError):
        tuner.train()


def test_yolo_predictor_predict_args_and_validation(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")

    model = _FakeYolo("w")
    predictor = YoloPredictor(
        weights=weights,
        device="cpu",
        img_size=320,
        conf=0.3,
        model_factory=lambda _: model,
    )

    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"x")

    out = predictor.predict_on_folder(images_dir, save=False, save_txt=True, stream=False)
    assert list(out) == [1, 2, 3]
    assert model.predict_calls[0]["conf"] == 0.3
    assert model.predict_calls[0]["imgsz"] == 320
    assert model.predict_calls[0]["save_txt"] is True

    img = tmp_path / "one.jpg"
    img.write_bytes(b"x")
    res = predictor.predict_on_image(img)
    assert list(res) == [1, 2, 3]
