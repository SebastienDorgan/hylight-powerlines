# HyLight Powerlines

Automated pipeline to detect power‑line components and defects using Ultralytics YOLO on harmonised public datasets. It includes utilities to download Roboflow datasets, merge heterogeneous labels, train YOLO models, and run inference over folders of images.

## Highlights
- Component and defect detection with YOLO (Ultralytics).
- Roboflow downloader with retry logic for stale export links.
- Dataset merger that remaps labels and converts segmentation polygons to detection boxes.
- Optional VLM → Grounding DINO → SAM2 pipeline for “auto-labeling” and inspection.
- Reproducible environment via `uv` + lockfile, with `just` tasks for linting, type checks, and tests.

## Repository Structure
- `src/hylight_powerlines/`
  - `yolo.py` – thin wrappers for fine‑tuning and inference (`YoloFineTuner`, `YoloPredictor`).
  - `roboflow.py` – `RoboflowDownloader` to fetch exports as ZIPs and resolve latest versions.
  - `merge.py` – merge multiple YOLO exports into a unified dataset (segmentation → bbox).
  - `curate.py` – curate merged datasets (keep sources, drop classes, rebuild `data.yaml`).
  - `llm/` – VLM pre-analysis + Grounding DINO refine + optional SAM2.
- `scripts/`
  - `download_all_training_datasets.py` – fetch selected Roboflow datasets to `data/external/`.
  - `build_detection_dataset.py` – merge component datasets → `data/detection_dataset/`.
  - `curate_detection_dataset.py` – subset + relabel the merged component dataset → `data/detection_dataset_curated/`.
  - `build_defects_dataset.py` – merge defect datasets → `data/defect_detection_dataset/`.
  - `fine_tune_parts_detection.py` – train component detector.
  - `fine_tune_defects_detection.py` – train defect detector.
  - `llm_detection_pipeline.py` – batch “powerline parts” detection over `assets/images/` (writes `outputs/`).
  - `run_parts_detection_network.py` – example YOLO inference runner over `assets/images/`.
  - `download_test_data.py` – pull sample images from Google Drive (service account; optional dependency).
- `data/`
  - `detection_mapping.yaml`, `defect_mapping.yaml` – class maps and per‑dataset remapping.
- `assets/` – images used for experiments (optional; filled by `download_test_data.py`).
- `runs/` – Ultralytics outputs (training and prediction).
- `outputs/` – script outputs (for example LLM pipeline summaries and visualizations).
- `docs/` – PDFs from the original technical test context.
- `justfile` – common tasks (format, lint, typecheck, test, CI bundle).
- `strategy.md` – background and approach for the technical test.

## Requirements
- OS: Linux/macOS/Windows. GPU optional; training benefits from CUDA.
- Python: project targets Python 3.14; use `uv` to manage toolchain and lockfile.
- Model runtimes: Ultralytics / Transformers / SAM2 are PyTorch-based (GPU optional). First run may download model weights.

## Quickstart
- Install `uv` (recommended) and optionally `just` (task runner).
- Set up the environment:
  - `just sync` (or `uv sync`)
  - Run commands via `uv run ...` (or activate `.venv` if you prefer).

## Secrets and Environment
- Create a `.env` (or export vars in your shell). Do not commit real keys.
- Required variables by feature:
  - Roboflow downloads: `ROBOFLOW_API_KEY=<your_key>`
  - LLM pipeline (LiteLLM):
    - Set a provider-prefixed model name: `VLM_MODEL=openai/gpt-5.2` (default used by the batch script)
    - Provide the corresponding API key, e.g. `OPENAI_API_KEY=...`
  - (Optional) Google Drive sample images: `HYLIGHT_FOLDER_ID=<drive_folder_id>`, `GOOGLE_SERVICE_ACCOUNT_JSON=<path_to_service_account_json>`
  - (Optional) SAM2 refinement:
    - `USE_SAM2=1`
    - `SAM2_CONFIG=<path/to/sam2_config.yaml>`
    - `SAM2_CKPT=<path/to/sam2_checkpoint.pt>`
- `.env` is not auto‑loaded by the scripts. Quick ways to load it in POSIX shells:
  - `set -a; source .env; set +a` (exports all KEY=VAL pairs)
  - or export individual variables explicitly.

## Sample Images From Google Drive
`scripts/download_test_data.py` depends on `pydrive2`, which is intentionally not pinned in this repo (it currently constrains `cryptography<44`, flagged by OSV). If you still want to use it, install `pydrive2` in a separate environment and run the script there.

- Prepare a service account JSON and set environment variables:
  - `export HYLIGHT_FOLDER_ID=...`
  - `export GOOGLE_SERVICE_ACCOUNT_JSON=service-account.json`
- Download images to `assets/images/`:
  - `uv run python scripts/download_test_data.py`

## Download Public Training Data (Roboflow)
- Export API key and run the downloader:
  - `export ROBOFLOW_API_KEY=...`
  - `uv run python scripts/download_all_training_datasets.py`
- Datasets are extracted under `data/external/<dataset_name>/`.

## Build Merged Datasets
- Component dataset (6 classes: `tower, insulator, spacer, damper, tower_plate, conductor`):
  - `uv run python scripts/build_detection_dataset.py`
  - Output: `data/detection_dataset/` with YOLO `data.yaml` + images/labels splits.
- Defect dataset (4 classes: `insulator_defect, damper_defect, conductor_defect, foreign_object`):
  - `uv run python scripts/build_defects_dataset.py`
  - Output: `data/defect_detection_dataset/`.

### Curate the Component Dataset (optional)
Use this to keep only selected source subsets and optionally drop classes. It also renumbers class IDs and regenerates `data.yaml`.

- Run:
  - `uv run python scripts/curate_detection_dataset.py`
- Output: `data/detection_dataset_curated/`.
- Configuration:
  - Edit `scripts/curate_detection_dataset.py` to change `keep_sources` and `drop_classes`.
  - As currently configured, it drops `spacer`, `damper`, `conductor`, and `tower_plate` (keeping only `tower` and `insulator`).
  - The curator infers the original source prefix from file names like `<prefix>_<split>_000123.jpg` generated by the merge step.
- Training with curated data:
  - `scripts/fine_tune_parts_detection.py` uses `data/detection_dataset_curated` by default; update `dataset_root` (and run name) if you train on a different class set.

## Train Models
- Component detector:
  - `uv run python scripts/fine_tune_parts_detection.py`
  - Weights and metrics: `runs/powerlines/<run_name>/` (default `yolov8s-6cls`).
- Defect detector:
  - `uv run python scripts/fine_tune_defects_detection.py`
  - Weights and metrics: `runs/defects/<run_name>/` (default `yolov8s-defects-4cls`).

## Inference
- Python API via `YoloPredictor` (use your trained weights):
  ```python
  from hylight_powerlines.yolo import YoloPredictor

  predictor = YoloPredictor(
      weights="runs/powerlines/yolov8s-6cls/weights/best.pt",
      device="0",      # or "cpu"
      img_size=1024,
      conf=0.25,
  )
  predictor.predict_on_folder("assets/images", save=True, save_txt=True)
  ```
- Example script using a COCO‑pretrained model on `assets/images`: `uv run python scratch.py`.

## LLM → Grounding DINO → SAM2 (batch over `assets/images`)
This is useful for quick inspection / auto-labeling when you don’t have a trained YOLO model yet.

- Run (writes outputs under `outputs/llm_pipeline/`):
  - `uv run python scripts/llm_detection_pipeline.py`
- Common CLI flags:
  - `--images_dir assets/images` (default)
  - `--out_root outputs/llm_pipeline` (default)
  - `--overwrite` to regenerate per-image outputs
- Outputs:
  - `outputs/llm_pipeline/summary.yaml` – recap of processed images and their final detections
  - `outputs/llm_pipeline/<image_stem>/preanalysis.json` – VLM proposals (boxes + prompt variants)
  - `outputs/llm_pipeline/<image_stem>/01_vlm_boxes.jpg` – VLM proposals drawn on the full image
  - `outputs/llm_pipeline/<image_stem>/02_gdino_boxes.jpg` – Grounding DINO refined boxes
  - `outputs/llm_pipeline/<image_stem>/<image_stem>.txt` – YOLO label export for the final boxes
  - `outputs/llm_pipeline/<image_stem>/final.json` – final boxes JSON (with prompt/source metadata)
  - If SAM2 is enabled: `outputs/llm_pipeline/<image_stem>/03_sam2_boxes.jpg` + mask images in the same folder
  - Tuning via env vars (defaults shown):
    - `VLM_MODEL=openai/gpt-5.2`, `VLM_MAX_TOKENS=12000`
    - `GDINO_MODEL=IDEA-Research/grounding-dino-tiny`, `GDINO_BOX_THR=0.20`, `GDINO_TEXT_THR=0.20`
    - `ROI_SCALE=2.5`, `NMS_IOU=0.5`
    - `USE_SAM2=0|1`, `SAM2_DEVICE=auto`, `SAM2_CONFIG=...`, `SAM2_CKPT=...`

## Development
- Format: `just format`
- Lint: `just lint`
- Type check: `just typecheck`
- Tests: `just test`
- Coverage: `just test-coverage`
- All‑in‑one (CI‑like): `just ci`

If you don’t have `just`, run the underlying `uv run ...` commands directly.

## Configure Scripts
- Datasets to download are listed in `scripts/download_all_training_datasets.py` (`datasets` list). Edit or extend as needed.
- Training hyper‑parameters live in `scripts/fine_tune_*.py` under the `overrides` dict and class args (epochs, imgsz, device, etc.).
- Label mappings are in `data/detection_mapping.yaml` and `data/defect_mapping.yaml`. Only classes mapped to `null` are dropped during merge.
