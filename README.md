# HyLight Powerlines

Automated pipeline to detect power‑line components and defects using Ultralytics YOLO on harmonised public datasets. It includes utilities to download Roboflow datasets, merge heterogeneous labels, train YOLO models, and run inference over folders of images.

## Highlights
- Component and defect detection with YOLO (Ultralytics).
- Roboflow downloader with retry logic for stale export links.
- Dataset merger that remaps labels and converts segmentation polygons to detection boxes.
- Reproducible environment via `uv` + lockfile, with `just` tasks for linting, type checks, and tests.

## Repository Structure
- `src/hylight_powerlines/`
  - `yolo.py` – thin wrappers for fine‑tuning and inference (`YoloFineTuner`, `YoloPredictor`).
  - `roboflow.py` – `RoboflowDownloader` to fetch exports as ZIPs and resolve latest versions.
  - `datasets.py` – label‑mapping + dataset merging utilities.
- `scripts/`
  - `download_all_training_datasets.py` – fetch selected Roboflow datasets to `data/external/`.
  - `build_detection_dataset.py` – merge component datasets → `data/detection_dataset/`.
  - `build_defects_dataset.py` – merge defect datasets → `data/defect_detection_dataset/`.
  - `fine_tune_parts_detection.py` – train component detector.
  - `fine_tune_defects_detection.py` – train defect detector.
  - `download_test_data.py` – pull sample images from Google Drive (service account).
- `data/`
  - `detection_mapping.yaml`, `defect_mapping.yaml` – class maps and per‑dataset remapping.
- `assets/` – images used for experiments (optional; filled by `download_test_data.py`).
- `runs/` – Ultralytics outputs (training and prediction).
- `justfile` – common tasks (format, lint, typecheck, test, CI bundle).
- `strategy.md` – background and approach for the technical test.

## Requirements
- OS: Linux/macOS/Windows. GPU optional; training benefits from CUDA.
- Python: project targets Python 3.14; use `uv` to manage toolchain and lockfile.

## Quickstart
- Install `uv` (recommended): follow official instructions or run your usual installer.
- Clone and set up the environment:
  - `uv sync`
  - Activate the environment if desired (optional with `uv`): `uv run ...` wraps commands.

## Secrets and Environment
- Create a `.env` (or export vars in your shell). Do not commit real keys.
- Required variables by feature:
  - Roboflow downloads: `ROBOFLOW_API_KEY=<your_key>`
  - Google Drive sample images: `HYLIGHT_FOLDER_ID=<drive_folder_id>`, `GOOGLE_SERVICE_ACCOUNT_JSON=<path_to_service_account_json>`


## Sample Images From Google Drive
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


## Train Models
- Component detector:
  - `uv run python scripts/fine_tune_parts_detection.py`
  - Weights and metrics: `runs/powerlines/<run_name>/` (default `yolov8s-6cls`).
- Defect detector:
  - `uv run python scripts/fine_tune_defects_detection.py`
  - Weights and metrics: `runs/defects/<run_name>/` (default `yolov8s-defects-4cls`).


