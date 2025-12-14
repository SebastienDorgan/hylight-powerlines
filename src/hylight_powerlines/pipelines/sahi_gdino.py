"""SAHI tiling → Grounding DINO detection pipeline (single combined prompt).

This pipeline is designed for large aerial images where tiling improves recall.
"""

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from hylight_powerlines.pipelines.steps import SupportsGdino
from hylight_powerlines.preprocessing.tiling_sahi import slice_with_sahi
from hylight_powerlines.vision.export_yolo import export_yolo
from hylight_powerlines.vision.geometry import iou, nms
from hylight_powerlines.vision.image import ensure_dir
from hylight_powerlines.vision.labels import combined_prompt, normalize_label
from hylight_powerlines.vision.types import Box
from hylight_powerlines.vision.vis import draw_boxes

LOG = logging.getLogger(__name__)

_GDINO_QUERY_SUFFIX = (
    " . overhead aerial photo"
    " . overhead power line infrastructure"
    " . utility pole . power pole . electric pylon . transmission tower"
    " . overhead power lines . conductor wires"
    " . insulator string . stockbridge damper . conductor spacer"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SahiTiling:
    """SAHI tiling parameters."""

    slice_w: int = 1024
    slice_h: int = 1024
    overlap_ratio: float = 0.20
    require_support: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class SahiGdinoParams:
    """Grounding DINO inference parameters."""

    box_thr: float = 0.35
    text_thr: float = 0.25
    max_dets_per_tile: int = 30
    min_score: float = 0.30
    min_area_frac: float = 0.00001
    max_area_frac: float = 0.60
    max_aspect: float = 20.0
    # More conservative limits for small parts (helps reject big boxes on roads/cars).
    parts_max_area_frac: float = 0.01
    parts_max_aspect: float = 12.0
    # Towers/utility poles can be tall/skinny.
    tower_max_area_frac: float = 0.60
    tower_max_aspect: float = 80.0
    support_iou: float = 0.30
    min_support: int = 2
    # If <=0, auto-select based on image size and tilings.
    global_max_side: int = 0
    # Image-level guardrails for empty / cluttered scenes.
    anchor_label: str = "tower"
    anchor_min_score: float = 0.45
    anchor_min_groups: int = 2
    anchor_support_iou: float = 0.30
    anchor_rescue_score: float = 0.85
    no_anchor_min_keep_score: float = 0.55
    clutter_max_boxes: int = 40
    clutter_support_iou: float = 0.20
    clutter_rescue_score: float = 0.90
    # Context gate: keep small-part detections only near an anchored structure.
    parts_anchor_max_dist_frac: float = 0.25
    parts_no_anchor_min_keep_score: float = 0.75
    parts_far_rescue_score: float = 0.90
    max_boxes_per_label: int = 50
    max_boxes_per_image: int = 200
    max_boxes_per_image_no_anchor: int = 25


@dataclass(frozen=True, slots=True, kw_only=True)
class SahiGdinoRun:
    """Run configuration for a SAHI→GDINO pass on a single image."""

    image_path: Path
    outdir: Path
    targets: list[str]
    tilings: tuple[SahiTiling, ...] = (
        SahiTiling(slice_w=2048, slice_h=2048, overlap_ratio=0.10, require_support=False),
        SahiTiling(),
    )
    gdino_params: SahiGdinoParams = SahiGdinoParams()
    nms_iou: float = 0.50
    overwrite: bool = False
    verbose: bool = False


TileProvider = Callable[[Image.Image, SahiTiling], list[tuple[Image.Image, int, int]]]


def gdino_query(targets: list[str]) -> str:
    """Build the Grounding DINO text query used for all tiles."""
    return combined_prompt(targets) + _GDINO_QUERY_SUFFIX


def _auto_global_max_side(img: Image.Image, *, tilings: list[SahiTiling]) -> int:
    """Choose the global-pass resize target from image/tiling geometry.

    The global pass exists to catch large objects that may be split across tiles,
    so default to the full image size (no extra downscale) and only rely on
    tilings to avoid pathological configs (e.g., tiles larger than the image).
    """
    max_dim = max(img.size)
    max_tile = max((max(int(t.slice_w), int(t.slice_h)) for t in tilings), default=0)
    return max(max_dim, max_tile)


def _resize_for_global(img: Image.Image, *, max_side: int) -> tuple[Image.Image, float, float]:
    """Resize `img` so that max(w,h) == max_side (if larger), returning scale factors.

    Returns:
        (resized_img, sx, sy) where sx = orig_w/resized_w and sy = orig_h/resized_h.
    """
    w, h = img.size
    if max_side <= 0:
        return img, 1.0, 1.0
    if max(w, h) <= max_side:
        return img, 1.0, 1.0
    if w >= h:
        new_w = max_side
        new_h = max(1, round(h * (max_side / w)))
    else:
        new_h = max_side
        new_w = max(1, round(w * (max_side / h)))
    resized = img.resize((int(new_w), int(new_h)))
    sx = w / float(new_w)
    sy = h / float(new_h)
    return resized, sx, sy


def detect_on_full_image(
    img: Image.Image,
    *,
    targets: list[str],
    gdino: SupportsGdino,
    gdino_params: SahiGdinoParams,
    global_max_side: int,
) -> list[Box]:
    """Run GDINO once on a (possibly downscaled) full image to catch large objects."""
    w, h = img.size
    resized, sx, sy = _resize_for_global(img, max_side=int(global_max_side))
    prompt = gdino_query(targets)
    dets = gdino.detect(
        img=resized,
        prompt=prompt,
        box_threshold=float(gdino_params.box_thr),
        text_threshold=float(gdino_params.text_thr),
    )
    # Keep a small top-K even in global pass.
    dets = sorted(dets, key=lambda b: b.score, reverse=True)[: int(gdino_params.max_dets_per_tile)]

    out: list[Box] = []
    for d in dets:
        label = normalize_label(str(d.label), targets)
        if label is None:
            continue
        out.append(
            Box(
                label=label,
                x1=float(d.x1 * sx),
                y1=float(d.y1 * sy),
                x2=float(d.x2 * sx),
                y2=float(d.y2 * sy),
                score=float(d.score),
                source="gdino_global",
                prompt=prompt,
            ).clip(w, h)
        )
    return _filter_boxes_with_params(out, image_w=w, image_h=h, params=gdino_params)


def _filter_boxes(
    boxes: list[Box],
    *,
    image_w: int,
    image_h: int,
    min_score: float,
    min_area_frac: float,
    max_area_frac: float,
    max_aspect: float,
) -> list[Box]:
    """Cheap post-filters to reduce obvious false positives."""
    if not boxes:
        return []
    img_area = float(image_w * image_h)
    out: list[Box] = []
    for b in boxes:
        if b.score < min_score:
            continue
        bw = max(0.0, b.x2 - b.x1)
        bh = max(0.0, b.y2 - b.y1)
        if bw <= 0.0 or bh <= 0.0:
            continue
        area_frac = (bw * bh) / img_area if img_area > 0 else 0.0
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        aspect = max(bw / bh, bh / bw)
        if aspect > max_aspect:
            continue
        out.append(b)
    return out


_PART_LABELS = {"insulator", "damper", "spacer", "tower_plate"}


def _intersection_area(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _area(b: Box) -> float:
    return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)


def _ioa(a: Box, b: Box) -> float:
    """Intersection over min-area (useful when boxes nest/are partial views)."""
    denom = min(_area(a), _area(b))
    if denom <= 0.0:
        return 0.0
    return _intersection_area(a, b) / denom


def _union_box(boxes: list[Box], *, source: str) -> Box:
    """Union a cluster of boxes into a single box (keeping max score/prompt)."""
    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    best = max(boxes, key=lambda b: b.score)
    return Box(
        label=best.label,
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        score=float(best.score),
        source=source,
        prompt=best.prompt,
    )


def _merge_anchor_boxes(
    *,
    global_boxes: list[Box],
    tiled_by_tiling: list[list[Box]],
    params: SahiGdinoParams,
    nms_iou: float,
) -> list[Box]:
    """Consolidate tower/pole detections using global+tile agreement.

    The tiling pass often returns many partial/duplicate boxes along a single pole.
    This merges them into a small set of anchor boxes that are either:
      - very confident (anchor_rescue_score), or
      - supported across multiple passes (anchor_min_groups).
    """
    label = str(params.anchor_label)
    groups = [list(global_boxes), *[list(bs) for bs in tiled_by_tiling]]

    group_towers: list[list[Box]] = []
    for g in groups:
        ts = [b for b in g if b.label == label]
        # Mild NMS within group to reduce trivially duplicated tower boxes.
        ts = nms(ts, iou_thr=min(float(nms_iou), 0.30))
        group_towers.append(ts)

    all_towers = [b for ts in group_towers for b in ts]
    if not all_towers:
        return []

    clusters: list[list[Box]] = []
    thr = float(params.anchor_support_iou)
    for b in sorted(all_towers, key=lambda bb: bb.score, reverse=True):
        placed = False
        for cl in clusters:
            # Use a permissive match: IOA catches partial segments, IoU catches near-equal boxes.
            if any(iou(b, c) >= thr or _ioa(b, c) >= 0.50 for c in cl):
                cl.append(b)
                placed = True
                break
        if not placed:
            clusters.append([b])

    fused: list[Box] = []
    min_groups = max(1, int(params.anchor_min_groups))
    rescue = float(params.anchor_rescue_score)
    for cl in clusters:
        u = _union_box(cl, source="gdino_anchor_fused")
        support = 0
        for ts in group_towers:
            if any(iou(u, t) >= thr or _ioa(u, t) >= 0.30 for t in ts):
                support += 1
        if u.score >= rescue or support >= min_groups:
            fused.append(u)

    # Final NMS to avoid adjacent fused duplicates.
    return nms(fused, iou_thr=min(float(nms_iou), 0.40))


def _label_max_area_frac(label: str, *, params: SahiGdinoParams) -> float:
    if label == str(params.anchor_label):
        return float(params.tower_max_area_frac)
    if label in _PART_LABELS:
        return float(params.parts_max_area_frac)
    return float(params.max_area_frac)


def _label_max_aspect(label: str, *, params: SahiGdinoParams) -> float:
    if label == str(params.anchor_label):
        return float(params.tower_max_aspect)
    if label in _PART_LABELS:
        return float(params.parts_max_aspect)
    return float(params.max_aspect)


def _filter_boxes_with_params(
    boxes: list[Box],
    *,
    image_w: int,
    image_h: int,
    params: SahiGdinoParams,
) -> list[Box]:
    """Apply score/area/aspect filters with label-aware limits."""
    if not boxes:
        return []
    img_area = float(image_w * image_h)
    out: list[Box] = []
    for b in boxes:
        if b.score < float(params.min_score):
            continue
        bw = max(0.0, b.x2 - b.x1)
        bh = max(0.0, b.y2 - b.y1)
        if bw <= 0.0 or bh <= 0.0:
            continue
        area_frac = (bw * bh) / img_area if img_area > 0 else 0.0
        max_area = _label_max_area_frac(b.label, params=params)
        if area_frac < float(params.min_area_frac) or area_frac > max_area:
            continue
        aspect = max(bw / bh, bh / bw)
        if aspect > _label_max_aspect(b.label, params=params):
            continue
        out.append(b)
    return out


def _support_filter(boxes: list[Box], *, min_support: int, iou_thr: float) -> list[Box]:
    """Keep only boxes that are supported by other overlapping detections.

    With tiling+overlap, true objects tend to appear in multiple tiles. This
    filter drops boxes that appear only once (often spurious).
    """
    if min_support <= 1 or len(boxes) <= 1:
        return boxes

    grouped: dict[str, list[Box]] = defaultdict(list)
    for b in boxes:
        grouped[b.label].append(b)

    out: list[Box] = []
    for bs in grouped.values():
        if len(bs) < min_support:
            continue
        support = [1] * len(bs)  # include self
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                if iou(bs[i], bs[j]) >= iou_thr:
                    support[i] += 1
                    support[j] += 1
        for b, sc in zip(bs, support, strict=False):
            if sc >= min_support:
                out.append(b)
    return out


def nms_by_label(boxes: list[Box], *, iou_thr: float) -> list[Box]:
    """Run NMS per class label and return boxes sorted by score."""
    grouped: dict[str, list[Box]] = defaultdict(list)
    for b in boxes:
        grouped[b.label].append(b)

    out: list[Box] = []
    for bs in grouped.values():
        out.extend(nms(bs, iou_thr=iou_thr))
    return sorted(out, key=lambda b: b.score, reverse=True)


def _cap_boxes(
    boxes: list[Box],
    *,
    max_per_label: int,
    max_total: int,
) -> list[Box]:
    if not boxes:
        return []
    max_per_label = max(1, int(max_per_label))
    max_total = max(1, int(max_total))
    out: list[Box] = []
    grouped: dict[str, list[Box]] = defaultdict(list)
    for b in boxes:
        grouped[b.label].append(b)
    for bs in grouped.values():
        bs_sorted = sorted(bs, key=lambda b: b.score, reverse=True)[:max_per_label]
        out.extend(bs_sorted)
    return sorted(out, key=lambda b: b.score, reverse=True)[:max_total]


def _has_anchor(
    boxes: list[Box],
    *,
    label: str,
    min_score: float,
) -> bool:
    return any(b.label == label and b.score >= min_score for b in boxes)


def _supported_by_groups(
    b: Box,
    *,
    groups: list[list[Box]],
    iou_thr: float,
) -> int:
    """Count how many groups contain an overlapping box of the same label."""
    count = 0
    for g in groups:
        if any(bb.label == b.label and iou(bb, b) >= iou_thr for bb in g):
            count += 1
    return count


def _anchor_confirmed(
    merged: list[Box],
    *,
    groups: list[list[Box]],
    params: SahiGdinoParams,
) -> bool:
    """Return True if the anchor label is confirmed across multiple passes."""
    label = str(params.anchor_label)
    candidates = [
        b for b in merged if b.label == label and b.score >= float(params.anchor_min_score)
    ]
    if not candidates:
        return False

    min_groups = max(1, int(params.anchor_min_groups))
    rescue = float(params.anchor_rescue_score)
    iou_thr = float(params.anchor_support_iou)

    for b in candidates:
        if b.score >= rescue:
            return True
        support = _supported_by_groups(b, groups=groups, iou_thr=iou_thr)
        if support >= min_groups:
            return True
    return False


def _dist_point_to_rect(px: float, py: float, r: Box) -> float:
    """Distance from a point to a rectangle (0 if the point is inside)."""
    dx = 0.0
    if px < r.x1:
        dx = r.x1 - px
    elif px > r.x2:
        dx = px - r.x2

    dy = 0.0
    if py < r.y1:
        dy = r.y1 - py
    elif py > r.y2:
        dy = py - r.y2

    return hypot(dx, dy)


def _filter_parts_by_anchor_distance(
    boxes: list[Box],
    *,
    anchor_boxes: list[Box],
    image_w: int,
    image_h: int,
    params: SahiGdinoParams,
    require_anchor: bool,
) -> list[Box]:
    """Drop small-part detections that are far from any anchored structure."""
    if not boxes:
        return []

    if not anchor_boxes:
        if not require_anchor:
            return boxes
        return [
            b
            for b in boxes
            if b.label not in _PART_LABELS
            or b.score >= float(params.parts_no_anchor_min_keep_score)
        ]

    max_dim = float(max(image_w, image_h))
    max_dist = float(params.parts_anchor_max_dist_frac) * max_dim
    rescue = float(params.parts_far_rescue_score)

    out: list[Box] = []
    for b in boxes:
        if b.label not in _PART_LABELS:
            out.append(b)
            continue
        if b.score >= rescue:
            out.append(b)
            continue
        cx = (b.x1 + b.x2) / 2.0
        cy = (b.y1 + b.y2) / 2.0
        d = min((_dist_point_to_rect(cx, cy, a) for a in anchor_boxes), default=float("inf"))
        if d <= max_dist:
            out.append(b)
    return out


def _apply_guardrails(
    merged: list[Box],
    *,
    global_boxes: list[Box],
    tiled_by_tiling: list[list[Box]],
    image_w: int,
    image_h: int,
    params: SahiGdinoParams,
) -> list[Box]:
    """Reduce false positives in empty/cluttered images using conservative heuristics."""
    if not merged:
        return []

    merged = _cap_boxes(
        merged,
        max_per_label=int(params.max_boxes_per_label),
        max_total=int(params.max_boxes_per_image),
    )

    groups = [list(global_boxes), *[list(bs) for bs in tiled_by_tiling]]
    anchor_ok = _anchor_confirmed(merged, groups=groups, params=params)

    merged = _filter_boxes_with_params(merged, image_w=image_w, image_h=image_h, params=params)
    anchor_label = str(params.anchor_label)
    anchor_boxes = (
        [b for b in merged if b.label == anchor_label and b.score >= float(params.anchor_min_score)]
        if anchor_ok
        else []
    )
    merged = _filter_parts_by_anchor_distance(
        merged,
        anchor_boxes=anchor_boxes,
        image_w=image_w,
        image_h=image_h,
        params=params,
        require_anchor=not anchor_ok,
    )
    if anchor_ok:
        return merged

    max_score = max((b.score for b in merged), default=0.0)
    if max_score < float(params.no_anchor_min_keep_score):
        return []

    if len(merged) > int(params.clutter_max_boxes):
        kept: list[Box] = []
        for b in merged:
            if b.score >= float(params.clutter_rescue_score):
                kept.append(b)
                continue
            support = _supported_by_groups(
                b,
                groups=groups,
                iou_thr=float(params.clutter_support_iou),
            )
            if support >= 2:
                kept.append(b)
        merged = kept

    return _cap_boxes(
        merged,
        max_per_label=int(params.max_boxes_per_label),
        max_total=int(params.max_boxes_per_image_no_anchor),
    )


def detect_on_slices(
    img: Image.Image,
    *,
    targets: list[str],
    gdino: SupportsGdino,
    tiling: SahiTiling,
    gdino_params: SahiGdinoParams,
    nms_iou: float,
    tile_provider: TileProvider | None = None,
    apply_support: bool = True,
    source: str = "gdino",
) -> list[Box]:
    """Run SAHI slicing + GDINO detection and return merged boxes in full-image coords."""
    w, h = img.size
    prompt = gdino_query(targets)

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
        tile_dets = sorted(tile_dets, key=lambda b: b.score, reverse=True)[
            : int(gdino_params.max_dets_per_tile)
        ]
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
                    source=source,
                    prompt=prompt,
                ).clip(w, h)
            )

    dets = _filter_boxes_with_params(dets, image_w=w, image_h=h, params=gdino_params)
    if apply_support:
        dets = _support_filter(
            dets,
            min_support=int(gdino_params.min_support),
            iou_thr=float(gdino_params.support_iou),
        )
    return nms_by_label(dets, iou_thr=float(nms_iou))


def detect_multiresolution(
    img: Image.Image,
    *,
    targets: list[str],
    gdino: SupportsGdino,
    tilings: list[SahiTiling],
    gdino_params: SahiGdinoParams,
    nms_iou: float,
    tile_provider: TileProvider | None = None,
) -> tuple[list[Box], list[Box], list[Box]]:
    """Run a global pass + multiple tiling passes and merge."""
    w, h = img.size
    global_max_side = int(gdino_params.global_max_side)
    if global_max_side <= 0:
        global_max_side = _auto_global_max_side(img, tilings=tilings)

    global_boxes = detect_on_full_image(
        img,
        targets=list(targets),
        gdino=gdino,
        gdino_params=gdino_params,
        global_max_side=global_max_side,
    )

    tiled_by_tiling = [
        detect_on_slices(
            img,
            targets=list(targets),
            gdino=gdino,
            tiling=t,
            gdino_params=gdino_params,
            nms_iou=float(nms_iou),
            tile_provider=tile_provider,
            apply_support=bool(t.require_support),
            source=f"gdino_{int(t.slice_w)}x{int(t.slice_h)}",
        )
        for t in tilings
    ]

    tiled_all: list[Box] = []
    for bs in tiled_by_tiling:
        tiled_all.extend(bs)

    merged_towers = _merge_anchor_boxes(
        global_boxes=global_boxes,
        tiled_by_tiling=tiled_by_tiling,
        params=gdino_params,
        nms_iou=float(nms_iou),
    )
    non_towers = [
        b for b in (global_boxes + tiled_all) if b.label != str(gdino_params.anchor_label)
    ]
    merged = nms_by_label(merged_towers + non_towers, iou_thr=float(nms_iou))
    merged = _apply_guardrails(
        merged,
        global_boxes=global_boxes,
        tiled_by_tiling=tiled_by_tiling,
        image_w=int(w),
        image_h=int(h),
        params=gdino_params,
    )
    return global_boxes, tiled_all, merged


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

    tilings = list(cfg.tilings)
    global_boxes, tiled_boxes, boxes = detect_multiresolution(
        img,
        targets=list(cfg.targets),
        gdino=gdino,
        tilings=tilings,
        gdino_params=cfg.gdino_params,
        nms_iou=float(cfg.nms_iou),
    )

    if global_boxes:
        draw_boxes(img, global_boxes, cfg.outdir / "01_global_boxes.jpg")
    if tiled_boxes:
        draw_boxes(img, tiled_boxes, cfg.outdir / "02_tiled_boxes.jpg")
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
    tilings: list[SahiTiling] | None = None,
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
    tilings = tilings or [
        SahiTiling(slice_w=2048, slice_h=2048, overlap_ratio=0.10, require_support=False),
        SahiTiling(),
    ]
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
                    tilings=tuple(tilings),
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
