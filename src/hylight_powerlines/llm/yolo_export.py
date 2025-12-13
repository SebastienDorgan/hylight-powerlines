"""YOLO label export helpers."""

from __future__ import annotations

from pathlib import Path

from .models import Box


def export_yolo(
    boxes: list[Box],
    image_w: int,
    image_h: int,
    class_names: list[str],
    out_txt: Path,
) -> None:
    """Write YOLO detection labels to a single .txt file for the image.

    The format is: class cx cy w h (normalized to [0, 1]).
    """
    name_to_id = {n: i for i, n in enumerate(class_names)}
    lines: list[str] = []
    for b in boxes:
        if b.label not in name_to_id:
            continue
        cls = name_to_id[b.label]
        cx = ((b.x1 + b.x2) / 2.0) / image_w
        cy = ((b.y1 + b.y2) / 2.0) / image_h
        bw = (b.x2 - b.x1) / image_w
        bh = (b.y2 - b.y1) / image_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
