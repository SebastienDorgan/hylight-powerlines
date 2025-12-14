"""Geometry helpers (IoU, NMS, ROI expansion) for detection boxes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import Box

if TYPE_CHECKING:
    import numpy as np


def iou(a: Box, b: Box) -> float:
    """Compute intersection-over-union (IoU) between two boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.area() + b.area() - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes: list[Box], iou_thr: float = 0.5) -> list[Box]:
    """Non-maximum suppression (NMS) by IoU, keeping highest-score boxes."""
    boxes_sorted = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: list[Box] = []
    for b in boxes_sorted:
        if all(iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


def expand_box(b: Box, w: int, h: int, scale: float) -> Box:
    """Expand a box around its center by `scale`, clipped to image bounds."""
    cx = 0.5 * (b.x1 + b.x2)
    cy = 0.5 * (b.y1 + b.y2)
    bw = (b.x2 - b.x1) * scale
    bh = (b.y2 - b.y1) * scale
    return Box(
        label=b.label,
        x1=cx - 0.5 * bw,
        y1=cy - 0.5 * bh,
        x2=cx + 0.5 * bw,
        y2=cy + 0.5 * bh,
        score=b.score,
        source=b.source,
        prompt=b.prompt,
    ).clip(w, h)


def tight_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Compute the tightest bounding box around non-zero mask pixels."""
    import numpy as np

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max() + 1)
    y2 = int(ys.max() + 1)
    return x1, y1, x2, y2
