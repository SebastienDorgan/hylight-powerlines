"""SAM2 wrapper used to refine boxes into masks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .models import Box


class Sam2Segmenter:
    """Segment Anything Model v2 (SAM2) wrapper for box→mask refinement."""

    def __init__(
        self,
        sam2_config: str,
        sam2_ckpt: str,
        device: str = "auto",
        *,
        torch_module: Any | None = None,
        predictor: Any | None = None,
        sam_builder: Callable[[str, str], Any] | None = None,
        predictor_factory: Callable[[Any], Any] = SAM2ImagePredictor,
    ) -> None:
        """Initialize the segmenter.

        Args:
            sam2_config: SAM2 config path.
            sam2_ckpt: SAM2 checkpoint path.
            device: "auto", "cpu", or "cuda".
            torch_module: Optional torch-like module for dependency injection.
            predictor: Optional pre-built predictor (for tests).
            sam_builder: Optional builder for a SAM2 model.
            predictor_factory: Factory that builds a predictor from the model.
        """
        if torch_module is None:
            import torch as torch_module  # local import to keep module import lightweight

        if device == "auto":
            device = "cuda" if torch_module.cuda.is_available() else "cpu"

        self.torch = torch_module
        self.device = device

        if predictor is not None:
            self.predictor = predictor
            return

        if sam_builder is None:

            def sam_builder(cfg: str, ckpt: str) -> Any:
                return build_sam2(cfg, ckpt, device=device)

        sam2_model = sam_builder(sam2_config, sam2_ckpt)
        self.predictor = predictor_factory(sam2_model)

    def segment_from_box(self, img: Image.Image, box: Box) -> tuple[np.ndarray, float]:
        """Predict a binary mask for `box` on `img`.

        Returns:
            A tuple of (mask_uint8_0_or_255, score).
        """
        arr = np.array(img)
        self.predictor.set_image(arr)
        box_np = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            box=box_np[None, :],
            multimask_output=False,
        )
        m = masks[0].astype(np.uint8) * 255
        sc = float(scores[0])
        return m, sc
