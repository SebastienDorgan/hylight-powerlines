"""SAHI tiling → Grounding DINO detection pipeline (single combined prompt).

This pipeline is designed for large aerial images where tiling improves recall.
"""

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from hylight_powerlines.pipelines.steps import SupportsGdino
from hylight_powerlines.preprocessing.tiling_sahi import slice_with_sahi
from hylight_powerlines.vision.export_yolo import export_yolo
from hylight_powerlines.vision.geometry import nms
from hylight_powerlines.vision.image import ensure_dir
from hylight_powerlines.vision.labels import combined_prompt, normalize_label
from hylight_powerlines.vision.types import Box
from hylight_powerlines.vision.vis import draw_boxes

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SahiTiling:
    """SAHI tiling parameters."""

    slice_w: int = 1024
    slice_h: int = 1024
    overlap_ratio: float = 0.20


@dataclass(frozen=True, slots=True, kw_only=True)
class SahiGdinoParams:
    """Grounding DINO inference parameters."""

    box_thr: float = 0.20
    text_thr: float = 0.20


@dataclass(frozen=True, slots=True, kw_only=True)
class SahiGdinoRun:
    """Run configuration for a SAHI→GDINO pass on a single image."""

    image_path: Path
    outdir: Path
    targets: list[str]
    tiling: SahiTiling = SahiTiling()
    gdino_params: SahiGdinoParams = SahiGdinoParams()
    nms_iou: float = 0.50
    overwrite: bool = False
    verbose: bool = False


TileProvider = Callable[[Image.Image, SahiTiling], list[tuple[Image.Image, int, int]]]


def nms_by_label(boxes: list[Box], *, iou_thr: float) -> list[Box]:
    """Run NMS per class label and return boxes sorted by score."""
    grouped: dict[str, list[Box]] = defaultdict(list)
    for b in boxes:
        grouped[b.label].append(b)

    out: list[Box] = []
    for bs in grouped.values():
        out.extend(nms(bs, iou_thr=iou_thr))
    return sorted(out, key=lambda b: b.score, reverse=True)


def detect_on_slices(
    img: Image.Image,
    *,
    targets: list[str],
    gdino: SupportsGdino,
    tiling: SahiTiling,
    gdino_params: SahiGdinoParams,
    nms_iou: float,
    tile_provider: TileProvider | None = None,
) -> list[Box]:
    """Run SAHI slicing + GDINO detection and return merged boxes in full-image coords."""
    w, h = img.size
    prompt = combined_prompt(targets)

    if tile_provider is None:
        tiles = slice_with_sahi(
            img,
            slice_w=int(tiling.slice_w),
            slice_h=int(tiling.slice_h),
            overlap_ratio=float(tiling.overlap_ratio),
        )
    else:
        tiles = tile_provider(img, tiling)

    dets: list[Box] = []
    for tile, ox, oy in tiles:
        tile_dets = gdino.detect(
            img=tile,
            prompt=prompt,
            box_threshold=float(gdino_params.box_thr),
            text_threshold=float(gdino_params.text_thr),
        )
        for d in tile_dets:
            label = normalize_label(str(d.label), targets)
            if label is None:
                continue
            dets.append(
                Box(
                    label=label,
                    x1=float(d.x1 + ox),
                    y1=float(d.y1 + oy),
                    x2=float(d.x2 + ox),
                    y2=float(d.y2 + oy),
                    score=float(d.score),
                    source="gdino",
                    prompt=prompt,
                ).clip(w, h)
            )

    return nms_by_label(dets, iou_thr=float(nms_iou))


def run_sahi_gdino(
    cfg: SahiGdinoRun,
    *,
    gdino: SupportsGdino,
) -> dict[str, Any]:
    """Run the SAHI→GDINO pipeline for a single image and write outputs.

    Outputs (under `cfg.outdir`):
      - `02_gdino_boxes.jpg`: merged boxes drawn on the full image
      - `<image_stem>.txt`: YOLO label export for the merged boxes
      - `final.json`: JSON payload with boxes and metadata

    Returns:
        The dict that is written to `final.json`.
    """
    if cfg.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ensure_dir(cfg.outdir)
    final_json = cfg.outdir / "final.json"
    if final_json.exists() and not cfg.overwrite:
        return json.loads(final_json.read_text(encoding="utf-8"))

    img = Image.open(cfg.image_path).convert("RGB")
    w, h = img.size

    boxes = detect_on_slices(
        img,
        targets=list(cfg.targets),
        gdino=gdino,
        tiling=cfg.tiling,
        gdino_params=cfg.gdino_params,
        nms_iou=float(cfg.nms_iou),
    )

    draw_boxes(img, boxes, cfg.outdir / "02_gdino_boxes.jpg")
    export_yolo(
        boxes=boxes,
        image_w=w,
        image_h=h,
        class_names=list(cfg.targets),
        out_txt=cfg.outdir / f"{cfg.image_path.stem}.txt",
    )

    payload: dict[str, Any] = {
        "image": str(cfg.image_path),
        "image_w": int(w),
        "image_h": int(h),
        "targets": list(cfg.targets),
        "final_boxes": [
            {
                "label": b.label,
                "x1": float(b.x1),
                "y1": float(b.y1),
                "x2": float(b.x2),
                "y2": float(b.y2),
                "score": float(b.score),
                "source": str(b.source),
                "prompt": b.prompt,
            }
            for b in boxes
        ],
    }
    final_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_sahi_gdino_batch(
    *,
    images: list[Path],
    out_root: Path,
    targets: list[str],
    gdino: SupportsGdino,
    tiling: SahiTiling | None = None,
    gdino_params: SahiGdinoParams | None = None,
    nms_iou: float = 0.50,
    overwrite: bool = False,
    verbose: bool = False,
) -> tuple[list[dict[str, object]], int]:
    """Run SAHI→GDINO over `images` and write `summary.yaml` in `out_root`.

    Returns:
        (summary_images, failures)
    """
    ensure_dir(out_root)
    tiling = tiling or SahiTiling()
    gdino_params = gdino_params or SahiGdinoParams()
    summary: list[dict[str, object]] = []
    failures = 0

    for image_path in images:
        per_outdir = out_root / image_path.stem
        try:
            payload = run_sahi_gdino(
                SahiGdinoRun(
                    image_path=image_path,
                    outdir=per_outdir,
                    targets=list(targets),
                    tiling=tiling,
                    gdino_params=gdino_params,
                    nms_iou=float(nms_iou),
                    overwrite=overwrite,
                    verbose=verbose,
                ),
                gdino=gdino,
            )
            summary.append(
                {
                    "image": str(image_path),
                    "outdir": str(per_outdir),
                    "targets": list(targets),
                    "image_w": int(payload.get("image_w", 0) or 0),
                    "image_h": int(payload.get("image_h", 0) or 0),
                    "detections": list(payload.get("final_boxes", []) or []),
                }
            )
        except Exception as e:
            failures += 1
            LOG.exception("SAHI→GDINO failed for image=%s", image_path)
            summary.append(
                {
                    "image": str(image_path),
                    "outdir": str(per_outdir),
                    "targets": list(targets),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    dumped = yaml.safe_dump({"images": summary}, sort_keys=False)
    if dumped is None:
        dumped = ""
    if isinstance(dumped, bytes):
        dumped = dumped.decode("utf-8")
    (out_root / "summary.yaml").write_text(dumped, encoding="utf-8")
    return summary, failures
