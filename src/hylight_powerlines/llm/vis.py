"""Visualization helpers for the LLM pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import Box


def draw_boxes(img: Image.Image, boxes: list[Box], out_path: Path) -> None:
    """Draw labeled bounding boxes on an image and save to disk."""
    vis = img.copy()
    dr = ImageDraw.Draw(vis)
    w, h = vis.size
    # Make boxes readable even on very large aerial images.
    thickness = max(3, round(min(w, h) / 200))
    font_size = max(12, round(min(w, h) / 80))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:  # pragma: no cover
        font = ImageFont.load_default()
    color_by_source = {
        "vlm": (66, 135, 245),  # blue
        "gdino": (60, 179, 113),  # green
        "sam2": (220, 20, 60),  # crimson
    }
    for b in boxes:
        color = color_by_source.get(b.source, (255, 165, 0))
        x1, y1, x2, y2 = (round(b.x1), round(b.y1), round(b.x2), round(b.y2))
        dr.rectangle([x1, y1, x2, y2], width=thickness, outline=color)
        txt = f"{b.label} {b.score:.2f} {b.source}"
        tx, ty = x1 + thickness, y1 + thickness
        bbox = dr.textbbox((tx, ty), txt, font=font)
        dr.rectangle(bbox, fill=(0, 0, 0))
        dr.text((tx, ty), txt, fill=(255, 255, 255), font=font)
    vis.save(out_path)


def save_masks(masks: list[np.ndarray], outdir: Path, stem: str) -> None:
    """Save binary masks as PNG images under `outdir`."""
    from PIL import Image as PILImage

    for i, m in enumerate(masks):
        PILImage.fromarray(m).save(outdir / f"{stem}.mask{i:02d}.png")
