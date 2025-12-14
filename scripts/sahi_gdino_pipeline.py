#!/usr/bin/env python3
"""Batch runner: SAHI tiling → Grounding DINO detection (single combined prompt).

Core logic lives in `hylight_powerlines.pipelines.sahi_gdino`. This script stays small
on purpose (test-only workflow).
"""

import argparse
import os
import sys
from pathlib import Path

from hylight_powerlines.detectors.gdino_hf import GroundingDinoHF
from hylight_powerlines.pipelines.sahi_gdino import (
    SahiGdinoParams,
    SahiTiling,
    run_sahi_gdino_batch,
)


def _iter_jpgs(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg"}
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, default="assets/images")
    ap.add_argument("--out_root", type=str, default="outputs/sahi_gdino")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    targets = ["tower", "insulator", "damper", "spacer", "tower_plate"]

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"--images_dir is not a directory: {images_dir}")

    out_root = Path(args.out_root).expanduser().resolve()

    images = _iter_jpgs(images_dir)
    if not images:
        raise SystemExit(f"No JPG images found under: {images_dir}")

    tiling = SahiTiling(
        slice_w=int(os.environ.get("SLICE_W", "1024")),
        slice_h=int(os.environ.get("SLICE_H", "1024")),
        overlap_ratio=float(os.environ.get("OVERLAP_RATIO", "0.20")),
    )
    gdino_params = SahiGdinoParams(
        box_thr=float(os.environ.get("GDINO_BOX_THR", "0.20")),
        text_thr=float(os.environ.get("GDINO_TEXT_THR", "0.20")),
    )
    nms_iou = float(os.environ.get("NMS_IOU", "0.50"))

    gdino_model = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
    gdino_device = os.environ.get("GDINO_DEVICE", "auto")
    gdino = GroundingDinoHF(model_id=gdino_model, device=gdino_device)

    _, failures = run_sahi_gdino_batch(
        images=images,
        out_root=out_root,
        targets=targets,
        gdino=gdino,
        tiling=tiling,
        gdino_params=gdino_params,
        nms_iou=nms_iou,
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
