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

DEFAULT_TARGETS: list[str] = ["tower", "insulator", "damper", "spacer", "tower_plate"]

# Conservative defaults for open-vocabulary detection in aerial imagery:
# - reduce hallucinations in empty/clutter scenes
# - keep "obvious" detections without micro-tuning env vars
DEFAULT_GDINO_BOX_THR = 0.35
DEFAULT_GDINO_TEXT_THR = 0.25
DEFAULT_MIN_SCORE = 0.30

# Anchor (tower/pole) confirmation
DEFAULT_ANCHOR_MIN_SCORE = 0.45
DEFAULT_ANCHOR_MIN_GROUPS = 2
DEFAULT_ANCHOR_SUPPORT_IOU = 0.30
DEFAULT_ANCHOR_RESCUE_SCORE = 0.85

# No-anchor scenes (aggressive suppression)
DEFAULT_NO_ANCHOR_MIN_KEEP_SCORE = 0.60
DEFAULT_MAX_BOXES_PER_IMAGE_NO_ANCHOR = 15

# Clutter filtering
DEFAULT_CLUTTER_MAX_BOXES = 30
DEFAULT_CLUTTER_SUPPORT_IOU = 0.20
DEFAULT_CLUTTER_RESCUE_SCORE = 0.92

# Small parts geometry + context gate
DEFAULT_PARTS_MAX_AREA_FRAC = 0.005
DEFAULT_PARTS_MAX_ASPECT = 10.0
DEFAULT_PARTS_ANCHOR_MAX_DIST_FRAC = 0.15
DEFAULT_PARTS_NO_ANCHOR_MIN_KEEP_SCORE = 0.85
DEFAULT_PARTS_FAR_RESCUE_SCORE = 0.92


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

    targets = list(DEFAULT_TARGETS)

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"--images_dir is not a directory: {images_dir}")

    out_root = Path(args.out_root).expanduser().resolve()

    images = _iter_jpgs(images_dir)
    if not images:
        raise SystemExit(f"No JPG images found under: {images_dir}")

    tilings = [
        SahiTiling(
            slice_w=int(os.environ.get("SLICE_W_LARGE", "2048")),
            slice_h=int(os.environ.get("SLICE_H_LARGE", "2048")),
            overlap_ratio=float(os.environ.get("OVERLAP_RATIO_LARGE", "0.10")),
            require_support=False,
        ),
        SahiTiling(
            slice_w=int(os.environ.get("SLICE_W", "1024")),
            slice_h=int(os.environ.get("SLICE_H", "1024")),
            overlap_ratio=float(os.environ.get("OVERLAP_RATIO", "0.20")),
            require_support=True,
        ),
    ]
    gdino_params = SahiGdinoParams(
        box_thr=float(os.environ.get("GDINO_BOX_THR", str(DEFAULT_GDINO_BOX_THR))),
        text_thr=float(os.environ.get("GDINO_TEXT_THR", str(DEFAULT_GDINO_TEXT_THR))),
        min_score=float(os.environ.get("MIN_SCORE", str(DEFAULT_MIN_SCORE))),
        anchor_min_score=float(os.environ.get("ANCHOR_MIN_SCORE", str(DEFAULT_ANCHOR_MIN_SCORE))),
        anchor_min_groups=int(os.environ.get("ANCHOR_MIN_GROUPS", str(DEFAULT_ANCHOR_MIN_GROUPS))),
        anchor_support_iou=float(
            os.environ.get("ANCHOR_SUPPORT_IOU", str(DEFAULT_ANCHOR_SUPPORT_IOU))
        ),
        anchor_rescue_score=float(
            os.environ.get("ANCHOR_RESCUE_SCORE", str(DEFAULT_ANCHOR_RESCUE_SCORE))
        ),
        no_anchor_min_keep_score=float(
            os.environ.get("NO_ANCHOR_MIN_KEEP_SCORE", str(DEFAULT_NO_ANCHOR_MIN_KEEP_SCORE))
        ),
        clutter_max_boxes=int(os.environ.get("CLUTTER_MAX_BOXES", str(DEFAULT_CLUTTER_MAX_BOXES))),
        clutter_support_iou=float(
            os.environ.get("CLUTTER_SUPPORT_IOU", str(DEFAULT_CLUTTER_SUPPORT_IOU))
        ),
        clutter_rescue_score=float(
            os.environ.get("CLUTTER_RESCUE_SCORE", str(DEFAULT_CLUTTER_RESCUE_SCORE))
        ),
        max_boxes_per_image_no_anchor=int(
            os.environ.get(
                "MAX_BOXES_PER_IMAGE_NO_ANCHOR", str(DEFAULT_MAX_BOXES_PER_IMAGE_NO_ANCHOR)
            )
        ),
        parts_max_area_frac=float(
            os.environ.get("PARTS_MAX_AREA_FRAC", str(DEFAULT_PARTS_MAX_AREA_FRAC))
        ),
        parts_max_aspect=float(os.environ.get("PARTS_MAX_ASPECT", str(DEFAULT_PARTS_MAX_ASPECT))),
        parts_anchor_max_dist_frac=float(
            os.environ.get("PARTS_ANCHOR_MAX_DIST_FRAC", str(DEFAULT_PARTS_ANCHOR_MAX_DIST_FRAC))
        ),
        parts_no_anchor_min_keep_score=float(
            os.environ.get(
                "PARTS_NO_ANCHOR_MIN_KEEP_SCORE",
                str(DEFAULT_PARTS_NO_ANCHOR_MIN_KEEP_SCORE),
            )
        ),
        parts_far_rescue_score=float(
            os.environ.get("PARTS_FAR_RESCUE_SCORE", str(DEFAULT_PARTS_FAR_RESCUE_SCORE))
        ),
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
        tilings=tilings,
        gdino_params=gdino_params,
        nms_iou=nms_iou,
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
