"""Orchestrator for the LLM → Grounding DINO → SAM2 pipeline."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hylight_powerlines.detectors.gdino_hf import GroundingDinoHF
from hylight_powerlines.detectors.sam2 import Sam2Segmenter
from hylight_powerlines.detectors.vlm_litellm import pre_analyze
from hylight_powerlines.pipelines.steps import (
    SupportsGdino,
    SupportsSam2,
    refine_with_gdino,
    refine_with_sam2,
)
from hylight_powerlines.vision.export_yolo import export_yolo
from hylight_powerlines.vision.image import ensure_dir, read_image
from hylight_powerlines.vision.types import Box, PreAnalysis, Proposal
from hylight_powerlines.vision.vis import draw_boxes, save_masks


@dataclass
class PipelineConfig:
    """Configuration for the VLM → Grounding DINO → optional SAM2 pipeline."""

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


@dataclass(frozen=True)
class PipelineResult:
    """Final pipeline outputs."""

    image: Path
    image_w: int
    image_h: int
    targets: list[str]
    boxes: list[Box]
    masks: list[np.ndarray]


class _JsonBox(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x1: float
    y1: float
    x2: float
    y2: float


class _JsonProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    box: _JsonBox
    confidence: float = 0.0
    prompt_variants: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("prompt_variants", mode="before")
    @classmethod
    def _coerce_prompt_variants(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(s) for s in v]
        return []


class _PreAnalysisJson(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_w: int | None = None
    image_h: int | None = None
    proposals: list[_JsonProposal] = Field(default_factory=list)

    @field_validator("proposals", mode="before")
    @classmethod
    def _coerce_proposals(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        return []


class _FinalBoxJson(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str
    prompt: str | None = None


class _FinalJson(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image: str
    image_w: int
    image_h: int
    targets: list[str]
    final_boxes: list[_FinalBoxJson]


def run_pipeline(
    cfg: PipelineConfig,
    *,
    pre_analyzer: Callable[..., PreAnalysis] = pre_analyze,
    gdino_factory: Callable[[str, str], SupportsGdino] | None = None,
    sam2_factory: Callable[[str, str, str], SupportsSam2] | None = None,
) -> PipelineResult:
    """Run the end-to-end pipeline with the given configuration."""
    if cfg.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    logger = logging.getLogger(__name__)
    outdir = cfg.outdir
    ensure_dir(outdir)

    img = read_image(cfg.image_path)
    w, h = img.size
    logger.info(
        "Pipeline start: image=%s size=%sx%s targets=%s outdir=%s",
        cfg.image_path,
        w,
        h,
        list(cfg.targets),
        outdir,
    )

    # 1) VLM pre-analysis or load from JSON
    t0 = perf_counter()
    if cfg.pre_json:
        try:
            pre_json = _PreAnalysisJson.model_validate_json(
                Path(cfg.pre_json).read_text(encoding="utf-8")
            )
        except ValidationError as e:
            raise RuntimeError(f"Invalid pre-analysis JSON: {cfg.pre_json}") from e

        props: list[Proposal] = []
        for p in pre_json.proposals:
            b = Box(
                label=str(p.label),
                x1=float(p.box.x1),
                y1=float(p.box.y1),
                x2=float(p.box.x2),
                y2=float(p.box.y2),
                score=float(p.confidence),
                source="vlm",
                prompt=None,
            ).clip(*img.size)
            props.append(
                Proposal(
                    label=b.label,
                    box=b,
                    prompt_variants=[str(s).strip() for s in p.prompt_variants],
                    notes=str(p.notes),
                )
            )
        pre = PreAnalysis(
            image_w=int(pre_json.image_w or w),
            image_h=int(pre_json.image_h or h),
            proposals=props,
        )
        logger.info(
            "Step 1/3 pre-analysis: loaded from %s (%s proposals)",
            cfg.pre_json,
            len(pre.proposals),
        )
    else:
        pre = pre_analyzer(
            img=img,
            targets=list(cfg.targets),
            model=cfg.vlm_model,
            reasoning_effort=str(cfg.reasoning_effort),
            temperature=float(cfg.vlm_temperature),
            max_tokens=int(cfg.vlm_max_tokens),
            timeout_s=float(cfg.vlm_timeout_s),
            verbose=bool(cfg.verbose),
        )
        logger.info(
            "Step 1/3 pre-analysis: model=%s proposals=%s took=%.2fs",
            cfg.vlm_model,
            len(pre.proposals),
            perf_counter() - t0,
        )

    pre_out = _PreAnalysisJson(
        image_w=int(pre.image_w),
        image_h=int(pre.image_h),
        proposals=[
            _JsonProposal(
                label=p.label,
                box=_JsonBox(x1=p.box.x1, y1=p.box.y1, x2=p.box.x2, y2=p.box.y2),
                confidence=float(p.box.score),
                prompt_variants=list(p.prompt_variants),
                notes=str(p.notes),
            )
            for p in pre.proposals
        ],
    )
    (outdir / "preanalysis.json").write_text(pre_out.model_dump_json(indent=2), encoding="utf-8")

    draw_boxes(img, [p.box for p in pre.proposals], outdir / "01_vlm_boxes.jpg")

    # 2) Grounding DINO refine
    t1 = perf_counter()
    if gdino_factory is None:
        def gdino_factory(model_id: str, device: str) -> GroundingDinoHF:
            return GroundingDinoHF(model_id=model_id, device=device)

    gdino: SupportsGdino = gdino_factory(cfg.gdino_model, cfg.gdino_device)
    dets = refine_with_gdino(
        img=img,
        pre=pre,
        gdino=gdino,
        roi_scale=float(cfg.roi_scale),
        box_thr=float(cfg.gdino_box_thr),
        text_thr=float(cfg.gdino_text_thr),
        nms_iou_thr=float(cfg.nms_iou),
        keep_vlm_if_missing=bool(cfg.keep_vlm_if_missing),
        debug_dir=cfg.save_debug,
    )
    logger.info(
        "Step 2/3 Grounding DINO: model=%s boxes=%s took=%.2fs",
        cfg.gdino_model,
        len(dets),
        perf_counter() - t1,
    )
    draw_boxes(img, dets, outdir / "02_gdino_boxes.jpg")

    # 3) SAM2 optional refine
    masks: list[np.ndarray] = []
    final_boxes = dets
    if cfg.use_sam2:
        t2 = perf_counter()
        if not cfg.sam2_config or not cfg.sam2_ckpt:
            raise RuntimeError("--use_sam2 requires sam2_config and sam2_ckpt to be set")
        if sam2_factory is None:
            def sam2_factory(cfg_path: str, ckpt_path: str, device: str) -> Sam2Segmenter:
                return Sam2Segmenter(cfg_path, ckpt_path, device=device)

        sam2: SupportsSam2 = sam2_factory(cfg.sam2_config, cfg.sam2_ckpt, cfg.sam2_device)
        final_boxes, masks = refine_with_sam2(
            img=img,
            dets=dets,
            sam2=sam2,
            min_mask_area=cfg.min_mask_area,
        )
        logger.info(
            "Step 3/3 SAM2: masks=%s boxes=%s took=%.2fs",
            len(masks),
            len(final_boxes),
            perf_counter() - t2,
        )
        draw_boxes(img, final_boxes, outdir / "03_sam2_boxes.jpg")
        save_masks(masks, outdir, stem=cfg.image_path.stem)
    else:
        logger.info("Step 3/3 SAM2: skipped")

    # Export YOLO labels
    export_yolo(
        boxes=final_boxes,
        image_w=w,
        image_h=h,
        class_names=list(cfg.targets),
        out_txt=outdir / f"{cfg.image_path.stem}.txt",
    )

    # Save final JSON
    final_out = _FinalJson(
        image=str(cfg.image_path),
        image_w=int(w),
        image_h=int(h),
        targets=list(cfg.targets),
        final_boxes=[
            _FinalBoxJson(
                label=b.label,
                x1=b.x1,
                y1=b.y1,
                x2=b.x2,
                y2=b.y2,
                score=b.score,
                source=b.source,
                prompt=b.prompt,
            )
            for b in final_boxes
        ],
    )
    (outdir / "final.json").write_text(final_out.model_dump_json(indent=2), encoding="utf-8")

    return PipelineResult(
        image=cfg.image_path,
        image_w=w,
        image_h=h,
        targets=list(cfg.targets),
        boxes=final_boxes,
        masks=masks,
    )
