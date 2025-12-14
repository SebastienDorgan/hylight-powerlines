"""HuggingFace Grounding DINO wrapper."""

from collections.abc import Callable
from typing import Any

from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from hylight_powerlines.vision.types import Box


class GroundingDinoHF:
    """Thin wrapper around HF Grounding DINO for object detection."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        *,
        torch_module: Any | None = None,
        processor: Any | None = None,
        model: Any | None = None,
        processor_factory: Callable[[str], Any] = AutoProcessor.from_pretrained,
        model_factory: Callable[[str], Any] = GroundingDinoForObjectDetection.from_pretrained,
    ) -> None:
        """Initialize the detector.

        Args:
            model_id: HuggingFace model id.
            device: "auto", "cpu", or "cuda".
            torch_module: Optional torch-like module for dependency injection.
            processor: Optional pre-built processor.
            model: Optional pre-built model.
            processor_factory: Factory to build the processor from `model_id`.
            model_factory: Factory to build the model from `model_id`.
        """
        if torch_module is None:
            import torch as torch_module  # local import to keep module import lightweight

        if device == "auto":
            device = "cuda" if torch_module.cuda.is_available() else "cpu"

        self.torch = torch_module
        self.device = device
        self.processor: Any = processor if processor is not None else processor_factory(model_id)
        self.model = model if model is not None else model_factory(model_id)
        self.model.to(device)
        self.model.eval()

    def detect(
        self,
        img: Image.Image,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> list[Box]:
        """Detect boxes for `prompt` on `img`.

        Args:
            img: Input image.
            prompt: Text prompt (typically a set of phrases joined by separators).
            box_threshold: Box confidence threshold.
            text_threshold: Text relevance threshold.

        Returns:
            Detected boxes in absolute pixel coordinates of `img`.
        """
        w, h = img.size
        inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([[h, w]], device=self.device)
        try:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs["input_ids"],
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )
            except TypeError as e:
                # Newer `transformers` uses `threshold` instead of `box_threshold`.
                # Keep compatibility across versions.
                if "box_threshold" not in str(e):
                    raise
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs["input_ids"],
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )
        except AttributeError as e:  # pragma: no cover
            raise RuntimeError(
                "Transformers version missing 'post_process_grounded_object_detection'."
            ) from e

        out: list[Box] = []
        r0 = results[0]
        boxes = r0["boxes"].detach().cpu().numpy()
        scores = r0["scores"].detach().cpu().numpy()
        labels = r0["labels"]

        for (x1, y1, x2, y2), sc, lab in zip(boxes, scores, labels, strict=False):
            out.append(
                Box(
                    label=str(lab),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(sc),
                    source="gdino",
                    prompt=prompt,
                ).clip(w, h)
            )
        return out
