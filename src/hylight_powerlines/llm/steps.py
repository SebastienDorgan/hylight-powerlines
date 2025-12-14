"""Pipeline refinement steps (VLM proposals → GDINO → optional SAM2)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .geometry import expand_box, iou, nms, tight_box_from_mask
from .image import crop_with_box, ensure_dir
from .models import Box, PreAnalysis


class SupportsGdino(Protocol):
    """Protocol for a Grounding DINO-like detector."""

    def detect(
        self,
        img: Image.Image,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> list[Box]:
        """Detect boxes for a text prompt on an image."""
        ...


class SupportsSam2(Protocol):
    """Protocol for a SAM2-like segmenter."""

    def segment_from_box(self, img: Image.Image, box: Box) -> tuple[np.ndarray, float]:
        """Segment a mask from a box prompt."""
        ...


def _best_match_in_roi(
    dets: list[Box],
    proposal_label: str,
    proposal_box_roi: Box,
    prefer_iou: bool = True,
) -> Box | None:
    if not dets:
        return None
    filtered = [d for d in dets if proposal_label.lower() in d.label.lower()] or dets
    if prefer_iou:
        filtered = sorted(filtered, key=lambda d: (iou(d, proposal_box_roi), d.score), reverse=True)
    else:
        filtered = sorted(filtered, key=lambda d: d.score, reverse=True)
    return filtered[0]


def refine_with_gdino(
    img: Image.Image,
    pre: PreAnalysis,
    gdino: SupportsGdino,
    *,
    roi_scale: float,
    box_thr: float,
    text_thr: float,
    nms_iou_thr: float,
    keep_vlm_if_missing: bool,
    debug_dir: Path | None,
) -> list[Box]:
    """Refine VLM proposals using Grounding DINO in expanded ROIs."""
    w, h = img.size
    refined: list[Box] = []
    for prop in pre.proposals:
        roi = expand_box(prop.box, w, h, roi_scale)
        crop, (ox, oy) = crop_with_box(img, roi)
        prompt = " . ".join(prop.prompt_variants)
        dets_local = gdino.detect(
            img=crop,
            prompt=prompt,
            box_threshold=box_thr,
            text_threshold=text_thr,
        )

        if debug_dir is not None:
            ensure_dir(debug_dir)
            stem = f"roi_{prop.label}_{ox}_{oy}"
            crop.save(debug_dir / f"{stem}.jpg")

        dets_global = [
            Box(
                label=prop.label,
                x1=d.x1 + ox,
                y1=d.y1 + oy,
                x2=d.x2 + ox,
                y2=d.y2 + oy,
                score=d.score,
                source="gdino",
                prompt=prompt,
            ).clip(w, h)
            for d in dets_local
        ]

        best = _best_match_in_roi(dets_global, prop.label, prop.box, prefer_iou=True)
        if best is not None:
            refined.append(best)
        elif keep_vlm_if_missing:
            refined.append(prop.box)

    return nms(refined, iou_thr=nms_iou_thr)


def refine_with_sam2(
    img: Image.Image,
    dets: list[Box],
    sam2: SupportsSam2,
    *,
    min_mask_area: int,
) -> tuple[list[Box], list[np.ndarray]]:
    """Refine detection boxes using SAM2 box-to-mask segmentation.

    Args:
        img: Source image.
        dets: Input boxes in pixel coordinates.
        sam2: SAM2 segmenter adapter.
        min_mask_area: Minimum foreground pixels to keep a detection.

    Returns:
        A tuple of (refined_boxes, masks).
    """
    w, h = img.size
    out_boxes: list[Box] = []
    out_masks: list[np.ndarray] = []
    for d in dets:
        mask, sc = sam2.segment_from_box(img, d)
        tb = tight_box_from_mask(mask)
        if tb is None:
            continue
        x1, y1, x2, y2 = tb
        area = int((mask > 0).sum())
        if area < min_mask_area:
            continue
        out_boxes.append(
            Box(
                label=d.label,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=min(d.score, sc),
                source="sam2",
                prompt=d.prompt,
            ).clip(w, h)
        )
        out_masks.append(mask)
    return out_boxes, out_masks
