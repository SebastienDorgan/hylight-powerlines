#!/usr/bin/env python3
"""Batch runner for the hylight_powerlines.llm pipeline.

This script is intended to be "zero-config" for the common case:
- Scans JPG/JPEG images under `assets/images/`.
- Detects: tower/pylon, insulator, damper, spacer, tower/pylon plate.
- Writes per-image outputs under `outputs/llm_pipeline/<image_stem>/`.
- Produces:
  - `outputs/llm_pipeline/summary.yaml`: recap of detections per image

Advanced tuning is via environment variables:
- `VLM_MODEL` (default: "openai/gpt-5.2"; must include provider prefix for LiteLLM)
- `VLM_MAX_TOKENS` (default: 12000)
- `GDINO_MODEL` (default: "IDEA-Research/grounding-dino-tiny")
- `GDINO_BOX_THR`, `GDINO_TEXT_THR` (defaults: 0.20)
- `ROI_SCALE` (default: 2.5), `NMS_IOU` (default: 0.5)
- SAM2 refinement:
  - If `USE_SAM2` is unset, SAM2 is enabled automatically when both `SAM2_CONFIG`
    and `SAM2_CKPT` are set.
  - Set `USE_SAM2=0` to force-disable even if config/ckpt are set.
  - Set `USE_SAM2=1` to force-enable (requires `SAM2_CONFIG` and `SAM2_CKPT`).
- `SAM2_DEVICE` (default: "auto")
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import yaml
from PIL import Image

from hylight_powerlines.llm.pipeline import PipelineConfig, run_pipeline


def _iter_jpgs(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg"}
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


class FinalBox(TypedDict):
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str
    prompt: NotRequired[str | None]


def _yaml_dump(data: object) -> str:
    dumped = yaml.safe_dump(data, sort_keys=False)
    if dumped is None:
        return ""
    if isinstance(dumped, bytes):
        return dumped.decode("utf-8")
    return dumped


def _env_bool(name: str) -> bool | None:
    """Parse an optional bool env var.

    Returns:
        True/False if present, else None.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"Invalid {name}={raw!r}; expected 0/1/true/false.")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, default="assets/images")
    ap.add_argument("--out_root", type=str, default="outputs/llm_pipeline")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    targets = ["tower", "insulator", "damper", "spacer", "tower_plate"]

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir).expanduser().resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"--images_dir is not a directory: {images_dir}")

    vlm_model = os.environ.get("VLM_MODEL", "openai/gpt-5.2")
    if "/" not in vlm_model:
        raise SystemExit(
            "LiteLLM requires a provider-prefixed model name.\n"
            f"Got VLM_MODEL={vlm_model!r}.\n"
            "Examples:\n"
            "  export VLM_MODEL='openai/gpt-5.2'\n"
            "  export OPENAI_API_KEY='...'\n"
        )
    vlm_max_tokens = int(os.environ.get("VLM_MAX_TOKENS", "12000"))
    gdino_model = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
    gdino_box_thr = float(os.environ.get("GDINO_BOX_THR", "0.20"))
    gdino_text_thr = float(os.environ.get("GDINO_TEXT_THR", "0.20"))
    roi_scale = float(os.environ.get("ROI_SCALE", "2.5"))
    nms_iou = float(os.environ.get("NMS_IOU", "0.5"))

    sam2_config = os.environ.get("SAM2_CONFIG", "")
    sam2_ckpt = os.environ.get("SAM2_CKPT", "")
    sam2_device = os.environ.get("SAM2_DEVICE", "auto")
    use_sam2_override = _env_bool("USE_SAM2")
    use_sam2 = (
        use_sam2_override if use_sam2_override is not None else bool(sam2_config and sam2_ckpt)
    )
    if use_sam2 and (not sam2_config or not sam2_ckpt):
        raise SystemExit(
            "SAM2 is enabled but SAM2_CONFIG and/or SAM2_CKPT is not set.\n"
            "Set:\n"
            "  export SAM2_CONFIG=/path/to/cfg.yaml\n"
            "  export SAM2_CKPT=/path/to/model.pt\n"
            "Or disable:\n"
            "  export USE_SAM2=0\n"
        )

    image_paths = _iter_jpgs(images_dir)
    if not image_paths:
        raise SystemExit(f"No JPG images found under: {images_dir}")

    summary: list[dict[str, object]] = []
    failures = 0
    for image_path in image_paths:
        per_outdir = out_root / image_path.stem
        final_json = per_outdir / "final.json"
        print(f"Analysing {image_path}")

        try:
            if not final_json.exists() or args.overwrite:
                run_pipeline(
                    PipelineConfig(
                        image_path=image_path,
                        outdir=per_outdir,
                        targets=targets,
                        vlm_model=vlm_model,
                        reasoning_effort="high",
                        vlm_temperature=1.0,
                        vlm_timeout_s=15 * 60.0,
                        vlm_max_tokens=vlm_max_tokens,
                        gdino_model=gdino_model,
                        gdino_device="auto",
                        gdino_box_thr=gdino_box_thr,
                        gdino_text_thr=gdino_text_thr,
                        roi_scale=roi_scale,
                        nms_iou=nms_iou,
                        keep_vlm_if_missing=False,
                        save_debug=None,
                        use_sam2=use_sam2,
                        sam2_config=sam2_config,
                        sam2_ckpt=sam2_ckpt,
                        sam2_device=sam2_device,
                        pre_json=None,
                        verbose=bool(args.verbose),
                    )
                )

            data = json.loads(final_json.read_text(encoding="utf-8"))
            boxes = cast(list[FinalBox], list(data.get("final_boxes", []) or []))

            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            summary.append(
                {
                    "image": str(image_path),
                    "outdir": str(per_outdir),
                    "targets": list(targets),
                    "image_w": w,
                    "image_h": h,
                    "detections": boxes,
                }
            )
        except Exception as e:
            failures += 1
            print(f"[ERROR] {image_path}: {type(e).__name__}: {e}", file=sys.stderr)

    (out_root / "summary.yaml").write_text(
        _yaml_dump({"images": summary}),
        encoding="utf-8",
    )

    if failures:
        print(f"Completed with {failures} failures.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
