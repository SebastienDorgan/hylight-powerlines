# AGENTS.md — Guidance for Coding Agents Working on hylight-powerlines

This document gives practical instructions for agents contributing to this repo. It covers tooling, code style, safety, and how to run or modify the main components (YOLO training/inference, Roboflow downloads, dataset merging/curation, and the vision-LLM → Grounding DINO → SAM2 pipeline).

Scope: This file applies to the entire repository.

---

## Overview

- Purpose: tools to download/prepare powerlines datasets, fine‑tune YOLO models, run predictions, and optionally run a VLM→Grounding DINO→SAM2 pipeline for zero‑shot detection.
- Languages: Python only.
- Package layout: app code under `src/hylight_powerlines/`; runnable scripts in `scripts/`.
- Task runner: `just` with `uv` for dependency management.

Key references:
- `README.md` for a high‑level intro and usage examples.
- `strategy.md` for notes and design context.
- `justfile` for reproducible commands.

---

## Tooling & Commands

- Python toolchain and deps are managed with `uv`. Use the provided `just` recipes.
- Common commands:
  - `just sync` — install dependencies
  - `just upgrade` — update lock + sync
  - `just lint [--fix ...]` — ruff checks (accepts pass‑through args)
  - `just format` — ruff format
  - `just typecheck` — light static checks via `pyrefly`
  - `just test` — run tests (none yet; OK to add targeted tests only)
  - `just cqa` — quick code‑quality aggregate

Notes:
- The CI/automation relies on these tasks. Keep their interfaces stable. If you must change, preserve backward compatibility or add a new task.

---

## Python Version, Style, and Conventions

- Target runtime per `pyproject.toml`: `requires-python = ">=3.14"` and ruff `target-version = "py314"`.
  - Use modern typing (`list[str]`, `dict[str, Any]`, `|` unions) and stdlib collections (`collections.abc`).
- Type hints: all new code must be fully annotated (functions, methods, module‑level vars where helpful). Avoid `Any` unless necessary; prefer precise `TypedDict`, `Literal`, and `Protocol` when appropriate.
- Functional core, imperative shell: put pure, side‑effect‑free logic in small functions/classes; keep I/O, networking, and orchestration at the edges (scripts/CLIs). Library modules should be deterministic and testable.
- Docstrings: use Google style. Summarize behavior, Args, Returns, Raises. Keep concise but specific.
- Dataclasses: prefer `dataclasses.dataclass` for structured data over ad‑hoc dicts/Pydantic unless validation/serialization needs justify otherwise. Consider `slots=True`, `frozen=True`, and `kw_only=True` where it improves safety and performance.
- Style & lint: ruff. Run `just lint` and `just format` before submitting changes.
- Logging: prefer `logging.getLogger(__name__)`. Do not use `print` in library code.
- Errors: raise precise exceptions; include actionable messages.
- Public API location: reusable code lives in `src/hylight_powerlines/`. Keep scripts thin and orchestrational under `scripts/`.

Docstrings/Comments:
- Keep docstrings short and task‑oriented. Mention assumptions, inputs, outputs, and failure modes.
- Avoid inline comments unless non‑obvious; prefer clear variable names and small functions.

---

## Project Layout

- `src/hylight_powerlines/`
  - `yolo.py`: wrappers for Ultralytics YOLO training/validation/prediction.
  - `roboflow.py`: robust Roboflow download client (latest version resolution + ZIP streaming).
  - `merge.py`: merge multiple Roboflow YOLO exports into a unified detection dataset (handles segmentation→bbox).
  - `curate.py`: filter/curate merged datasets (drop classes, keep selected sources, rebuild data.yaml).
- `scripts/`
  - `llm_detection_pipeline.py`: Vision‑LLM → Grounding DINO → SAM2 zero‑shot pipeline. Produces debug images, JSON, and YOLO label text.
- `runs/`, `data/`: outputs and datasets (ignored by git). Do not place source code here.

---

## Dependencies & Adding New Ones

- Runtime deps are declared in `pyproject.toml`. Prefer reusing existing libraries. Avoid large, niche, or GPL‑licensed dependencies unless explicitly approved.
- For `scripts/llm_detection_pipeline.py`, additional heavy deps are expected at runtime (user-installed): `numpy`, `torch`, `torchvision`, `transformers`, `accelerate`, `litellm`, and `sam2` (from Meta, optional). Do not add these to the packaged project unless we decide to vendor the pipeline into the library.
- If you must add a new dependency:
  1) explain the rationale in your PR/commit message,
  2) keep versions unconstrained or minimally pinned unless there’s a known issue,
  3) run `just sync` and ensure `uv.lock` updates.

---

## Data, Artifacts, and Secrets

- Never commit datasets, model weights, or large artifacts. Keep them under `data/` or `runs/` which are git‑ignored.
- Secrets: `.env` exists locally but must not be committed. Environment variables used by code include:
  - `ROBOFLOW_API_KEY` — Roboflow export client.
  - `OPENAI_API_KEY` (or other LiteLLM provider keys) — VLM pre‑analysis in `llm_detection_pipeline.py`.
  - `SAM2_CONFIG`, `SAM2_CKPT` — optional defaults for SAM2.

---

## GPU/Heavy Operations

- YOLO fine‑tuning and Grounding DINO/SAM2 inference are GPU‑friendly. The pipeline supports CPU but will be slow.
- The pipeline auto‑selects device (`cuda` if available else `cpu`) for Grounding DINO and SAM2.
- Do not make network calls (Roboflow/LLM/weights downloads) in unit tests. Mock or gate them behind flags.

---

## Making Changes Safely

- Keep APIs stable. If changing signatures, add keyword‑only arguments or new flags rather than breaking positional ones.
- Prefer small, composable functions with clear inputs/outputs.
- Validate file paths early and produce actionable error messages.
- Do not write outside of target output directories; ensure `mkdir -p` and idempotency where possible.
- For dataset transforms, be explicit about formats and include comments describing assumptions (e.g., YOLOv5/8 segmentation format).

---

## Testing & Validation

- There is no formal test suite yet. When adding logic in `src/hylight_powerlines/`, add focused tests under `tests/` that do not require network or GPUs.
- For file‑system work, use temporary directories and small synthetic fixtures.
- For networking (Roboflow), unit test the URL builders and response parsers with sample payloads.

---

## Quick Recipes

YOLO Training:
```
python -c "from pathlib import Path; from hylight_powerlines.yolo.wrappers import YoloFineTuner as T;\
T(Path('data/your_rf_export')).train()"
```

YOLO Inference:
```
python -c "from pathlib import Path; from hylight_powerlines.yolo.wrappers import YoloPredictor as P;\
P(Path('runs/your_run/weights/best.pt')).predict_on_folder('data/images')"
```

Roboflow Download (ZIP only):
```
python -c "from pathlib import Path; from hylight_powerlines.datasets.roboflow import RoboflowDownloader as R;\
R('workspace','project',version=None,export_format='yolov8').download_dataset(Path('data/ds.zip'))"
```

Vision‑LLM → Grounding DINO → SAM2:
```
python scripts/llm_detection_pipeline.py \
  --image path/to/img.jpg \
  --outdir out \
  --targets tower insulator damper \
  --gdino_model IDEA-Research/grounding-dino-tiny \
  --use_sam2 1 --sam2_config cfg.yaml --sam2_ckpt sam2.pt --verbose
```

Tips:
- Use `--pre_json` to skip the VLM call with a saved pre‑analysis.
- Use `--save_debug <dir>` to persist ROI crops.

---

## When in Doubt

- Prefer extending existing modules over creating new top‑level scripts.
- Ask for confirmation before destructive actions (overwriting directories, large moves).
- Keep changes minimal, reversible, and well‑logged.
