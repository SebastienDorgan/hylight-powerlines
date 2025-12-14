"""Dataset curation utilities for merged YOLO detection datasets."""

import logging
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger(__name__)


def _load_class_names(data_yaml: Path) -> list[str]:
    """Load class names from a YOLO data.yaml (supports list or dict)."""
    cfg = yaml.safe_load(data_yaml.read_text())
    names_field = cfg["names"]

    if isinstance(names_field, dict):
        # keys are typically "0", "1", ... or ints
        items = sorted(
            ((int(k), v) for k, v in names_field.items()),
            key=lambda kv: kv[0],
        )
        names = [v for _, v in items]
    elif isinstance(names_field, list):
        names = list(names_field)
    else:
        raise TypeError(f"Unsupported names field type: {type(names_field)!r}")

    return names


def _build_id_map(
    old_names: Sequence[str],
    drop_classes: list[str],
) -> tuple[dict[int, int | None], list[str]]:
    """Build mapping old_id -> new_id (or None) and resulting name list.

    Any class whose name == drop_class is removed (mapped to None).
    Other classes are reindexed to keep ids contiguous.
    """
    id_map: dict[int, int | None] = {}
    new_names: list[str] = []
    new_idx = 0

    for old_id, name in enumerate(old_names):
        if name in drop_classes:
            id_map[old_id] = None
        else:
            id_map[old_id] = new_idx
            new_names.append(name)
            new_idx += 1

    return id_map, new_names


def _dataset_prefix(stem: str, split: str) -> str | None:
    """Extract dataset prefix from filename stem '{prefix}_{split}_...'.

    Example:
        stem='electric_pole_merged_train_000001' and split='train'
        -> 'electric_pole_merged'
    """
    marker = f"_{split}_"
    if marker not in stem:
        return None
    return stem.split(marker, 1)[0]


def _remap_label_file(
    src_label: Path,
    dst_label: Path,
    id_map: dict[int, int | None],
) -> bool:
    """Remap a single YOLO label file using id_map.

    Returns:
        True if there was a label file (even if the result is empty).
        False if the source label file does not exist at all.

    Note:
        - If there is no label file => caller should skip the image.
        - If there is a label file but all boxes are dropped =>
          we still create an empty label file (keep background image).
    """
    if not src_label.exists():
        return False

    text = src_label.read_text().strip()
    if not text:
        # Label file exists but is empty: keep it as empty.
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("")
        return True

    lines = text.splitlines()
    new_lines: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(parts[0])
        new_id = id_map.get(cls_id)
        if new_id is None:
            # Dropped class (e.g. spacer)
            continue
        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(new_lines))
    return True


def curate_detection_dataset(
    source_root: Path,
    dest_root: Path,
    keep_sources: Iterable[str],
    drop_classes: list[str],
) -> None:
    """Curate a merged YOLO dataset.

    Operations:
      1. Keep only images coming from prefixes in keep_sources.
      2. Drop `drop_class` from labels and renumber remaining classes.
      3. Write a fresh data.yaml in dest_root.

    Args:
        source_root: Existing merged dataset root
                     (contains data.yaml, train/val/test).
        dest_root: Destination root for curated dataset.
        keep_sources: Iterable of dataset prefixes to keep
                      (e.g. 'electric_pole_merged', ...).
        drop_classes: Class names to remove from the dataset (e.g. ['spacer']).
    """
    source_root = source_root.expanduser().resolve()
    dest_root = dest_root.expanduser().resolve()

    data_yaml = source_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Could not find data.yaml under {source_root}")

    if dest_root.exists():
        raise RuntimeError(f"Destination {dest_root} already exists; aborting to avoid overwrite.")

    keep_sources_set = set(keep_sources)

    # Load class names and build id map (drop 'spacer', renumber others)
    old_names = _load_class_names(data_yaml)
    id_map, new_names = _build_id_map(old_names, drop_classes=drop_classes)
    LOG.info("Old classes: %s", old_names)
    LOG.info("New classes (after dropping %r): %s", drop_classes, new_names)

    # Copy/curate data split by split
    for split in ("train", "val", "test"):
        src_img_dir = source_root / split / "images"
        src_lbl_dir = source_root / split / "labels"

        if not src_img_dir.is_dir():
            LOG.info(f"Warning: missing {src_img_dir}, skipping split {split}")
            continue

        dst_img_dir = dest_root / split / "images"
        dst_lbl_dir = dest_root / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        n_total = 0
        n_kept = 0

        for img_path in src_img_dir.glob("*.*"):
            if not img_path.is_file():
                continue

            n_total += 1
            stem = img_path.stem
            prefix = _dataset_prefix(stem, split)
            if prefix is None:
                # Unexpected filename, keep nothing to be safe
                continue

            if prefix not in keep_sources_set:
                continue

            # Handle labels
            src_label = src_lbl_dir / f"{stem}.txt"
            dst_label = dst_lbl_dir / f"{stem}.txt"
            has_label_file = _remap_label_file(src_label, dst_label, id_map)

            # Policy:
            # - no label file at all => skip image
            # - label file exists (even empty) => copy image
            if not has_label_file:
                if dst_label.exists():
                    dst_label.unlink()
                continue

            dst_img_path = dst_img_dir / img_path.name
            shutil.copy2(img_path, dst_img_path)
            n_kept += 1

        LOG.info(f"{split}: kept {n_kept} / {n_total} images")

    # Build new data.yaml
    src_cfg = yaml.safe_load(data_yaml.read_text())

    new_cfg: dict[str, Any] = {
        "path": str(dest_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(new_names)},
    }

    # Optionally carry over other fields (e.g. 'nc' if present)
    if "nc" in src_cfg:
        new_cfg["nc"] = len(new_names)
    else:
        new_cfg["nc"] = len(new_names)

    dest_root.mkdir(parents=True, exist_ok=True)
    content = str(yaml.safe_dump(new_cfg, sort_keys=False))
    (dest_root / "data.yaml").write_text(content)
    LOG.info(f"Curated data.yaml written to {dest_root / 'data.yaml'}")
