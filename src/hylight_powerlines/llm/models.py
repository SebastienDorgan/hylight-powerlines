"""Data models for the LLM → Grounding DINO → SAM2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box with metadata.

    Attributes:
        label: Normalized class label used throughout the pipeline.
        x1, y1, x2, y2: Absolute pixel coordinates in the source image frame.
        score: Confidence score in [0, 1].
        source: Stage that produced the box (e.g., "vlm", "gdino", "sam2").
        prompt: Optional prompt text used to obtain this box.
    """

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str
    prompt: str | None = None

    def area(self) -> float:
        """Return the box area in pixels squared."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def clip(self, w: int, h: int) -> Box:
        """Clip coordinates to image bounds and normalize (x1<=x2, y1<=y2)."""
        x1 = float(max(0.0, min(self.x1, w - 1)))
        y1 = float(max(0.0, min(self.y1, h - 1)))
        x2 = float(max(0.0, min(self.x2, w - 1)))
        y2 = float(max(0.0, min(self.y2, h - 1)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return Box(
            label=self.label,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=self.score,
            source=self.source,
            prompt=self.prompt,
        )


@dataclass(frozen=True)
class Proposal:
    """A VLM proposal: a coarse box plus prompt variants."""

    label: str
    box: Box
    prompt_variants: list[str]
    notes: str = ""


@dataclass(frozen=True)
class PreAnalysis:
    """VLM pre-analysis output: proposals in pixel coordinates."""

    image_w: int
    image_h: int
    proposals: list[Proposal]


@dataclass(frozen=True)
class PipelineResult:
    """Final pipeline outputs."""

    image: Path
    image_w: int
    image_h: int
    targets: list[str]
    boxes: list[Box]
    masks: list[np.ndarray]


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""

    image_path: Path
    outdir: Path
    targets: list[str]

    vlm_model: str
    reasoning_effort: str = "medium"
    vlm_temperature: float = 0.0
    vlm_timeout_s: float = 60.0
    vlm_max_tokens: int = 2000

    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    gdino_device: str = "auto"
    gdino_box_thr: float = 0.20
    gdino_text_thr: float = 0.20
    roi_scale: float = 2.5
    nms_iou: float = 0.5
    keep_vlm_if_missing: bool = False
    save_debug: Path | None = None

    use_sam2: bool = False
    sam2_config: str = ""
    sam2_ckpt: str = ""
    sam2_device: str = "auto"
    min_mask_area: int = 50

    pre_json: Path | None = None
    verbose: bool = False
