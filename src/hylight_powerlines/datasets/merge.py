"""Dataset merging utilities for Roboflow-exported YOLO datasets."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import Polygon

LOG = logging.getLogger(__name__)


@dataclass
class LabelMapping:
    """Holds global class mapping and per-dataset label mappings."""

    name_to_index: dict[str, int]
    dataset_label_map: dict[
        str, dict[str, str | None]
    ]  # dataset -> orig_name -> target_name or None

    @classmethod
    def from_yaml(cls, path: Path) -> LabelMapping:
        """Load label mapping from YAML file."""
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        classes: dict[int, str] = data["classes"]
        # Reverse: name -> index
        name_to_index = {name: idx for idx, name in classes.items()}

        dataset_label_map: dict[str, dict[str, str | None]] = {}
        for ds_name, mapping in data["datasets"].items():
            dataset_label_map[ds_name] = dict(mapping)

        return cls(name_to_index=name_to_index, dataset_label_map=dataset_label_map)


def load_dataset_names_yaml(dataset_root: Path) -> list[str]:
    """Load 'names' array from a Roboflow YOLO data.yaml in dataset_root."""
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        # {0: 'class0', 1: 'class1', ...}
        sorted_items = sorted(names.items(), key=lambda kv: int(kv[0]))
        return [v for _, v in sorted_items]
    if isinstance(names, list):
        return names

    raise ValueError(f"Unexpected 'names' format in {data_yaml}: {type(names)!r}")


def ensure_dest_structure(dest_root: Path) -> None:
    """Create train/val/test images/labels dirs in destination root."""
    for split in ("train", "val", "test"):
        (dest_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dest_root / split / "labels").mkdir(parents=True, exist_ok=True)


def detect_splits(dataset_root: Path) -> dict[str, str]:
    """Detect available splits in a Roboflow export.

    Returns:
        Mapping from src_split_name -> dest_split_name (one of 'train', 'val', 'test').
    """
    mapping: dict[str, str] = {}

    if (dataset_root / "train").exists():
        mapping["train"] = "train"
    if (dataset_root / "valid").exists():
        mapping["valid"] = "val"
    if (dataset_root / "val").exists():
        mapping["val"] = "val"
    if (dataset_root / "test").exists():
        mapping["test"] = "test"

    return mapping


def iter_image_files(images_dir: Path) -> list[Path]:
    """Return all image files in a directory (non-recursive)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)


def coords_to_detection_bbox(coords: list[float]) -> list[float]:
    """Convert YOLO coords (detection or segmentation) to detection bbox [xc, yc, w, h].

    Supported formats:
        - Detection: [xc, yc, w, h]  -> returned as-is.
        - Segmentation (YOLOv5/YOLOv8-style): [xc, yc, w, h, x1, y1, x2, y2, ...]
          Polygon vertices (x_i, y_i) are normalised in [0, 1].

    The bounding box is computed from the polygon using shapely. If polygon
    coordinates are missing or invalid, we fall back to the first 4 values.
    """
    # Detection: class xc yc w h
    if len(coords) <= 4:
        return coords

    # Assume YOLOv5/8 segmentation: xc, yc, w, h, x1, y1, x2, y2, ...
    poly_coords = coords[4:]
    if len(poly_coords) < 6 or len(poly_coords) % 2 != 0:
        # Not enough points or malformed; best effort: keep the bbox part.
        return coords[:4]

    # Build polygon from (xi, yi) pairs (normalised coordinates)
    points = [(poly_coords[i], poly_coords[i + 1]) for i in range(0, len(poly_coords), 2)]
    try:
        poly = Polygon(points)
        if poly.is_empty:
            return coords[:4]
        minx, miny, maxx, maxy = poly.bounds
    except Exception:
        # Shapely failed for some reason; keep original bbox
        return coords[:4]

    xc = (minx + maxx) / 2.0
    yc = (miny + maxy) / 2.0
    w = maxx - minx
    h = maxy - miny
    return [xc, yc, w, h]


def merge_one_dataset(
    ds_key: str,
    ds_root: Path,
    label_mapping: LabelMapping,
    dest_root: Path,
    counters: dict[str, int],
) -> None:
    """Merge a single Roboflow dataset into the unified YOLOv8 dataset.

    Segmentation labels are converted to detection labels:
    - If a line uses detection format (class xc yc w h), it is kept as-is (class index remapped).
    - If a line uses segmentation format (class xc yc w h x1 y1 x2 y2 ...),
      the polygon is converted to a bounding box using shapely, and only
      the detection bbox is written out.
    """
    LOG.info(f"\n=== Merging dataset '{ds_key}' from {ds_root} ===")

    orig_names = load_dataset_names_yaml(ds_root)

    ds_label_map = label_mapping.dataset_label_map.get(ds_key)
    if ds_label_map is None:
        LOG.info(f"[INFO] No label mapping defined for dataset '{ds_key}', skipping.")
        return

    # Build index -> target_index mapping for this dataset
    index_to_target_index: dict[int, int | None] = {}
    for idx, orig_name in enumerate(orig_names):
        target_name = ds_label_map.get(orig_name)
        if target_name is None:
            # Map to None -> this class will be dropped
            index_to_target_index[idx] = None
        else:
            try:
                target_index = label_mapping.name_to_index[target_name]
            except KeyError as exc:
                raise KeyError(
                    f"Target class '{target_name}' from mapping for dataset '{ds_key}' "
                    f"is not defined in 'classes' section"
                ) from exc
            index_to_target_index[idx] = target_index

    split_map = detect_splits(ds_root)
    if not split_map:
        LOG.info(f"[WARN] No train/valid/val/test splits found in {ds_root}, skipping.")
        return

    for src_split, dest_split in split_map.items():
        src_images = ds_root / src_split / "images"
        src_labels = ds_root / src_split / "labels"

        if not src_images.exists() or not src_labels.exists():
            LOG.info(
                f"[WARN] Missing images/labels for split '{src_split}' in {ds_root}, "
                "skipping this split."
            )
            continue

        dest_images = dest_root / dest_split / "images"
        dest_labels = dest_root / dest_split / "labels"

        img_files = iter_image_files(src_images)
        if not img_files:
            LOG.info(f"[INFO] No images found in {src_images}, skipping.")
            continue

        LOG.info(f"  Split '{src_split}' -> '{dest_split}': {len(img_files)} images")

        for img_path in img_files:
            stem = img_path.stem
            label_path = src_labels / f"{stem}.txt"

            # Policy:
            # - If the label file does NOT exist at all -> skip image.
            if not label_path.exists():
                continue

            new_lines: list[str] = []
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        orig_idx = int(parts[0])
                    except ValueError:
                        # Malformed line, skip
                        continue

                    target_idx = index_to_target_index.get(orig_idx)
                    if target_idx is None:
                        # Mapped to null -> drop this annotation
                        continue

                    # Convert coords to detection bbox (handles detection and segmentation)
                    try:
                        coords = [float(v) for v in parts[1:]]
                    except ValueError:
                        # Malformed coordinates, skip this annotation
                        continue

                    det_coords = coords_to_detection_bbox(coords)
                    # Normalised coords are kept as float; format with limited precision
                    det_coords_str = [f"{v:.6f}" for v in det_coords]
                    new_line = " ".join([str(target_idx), *det_coords_str])
                    new_lines.append(new_line)

            # We keep the image even if new_lines is empty (background-only image).
            counters.setdefault(dest_split, 0)
            counters[dest_split] += 1
            new_stem = f"{ds_key}_{dest_split}_{counters[dest_split]:06d}"
            new_img_path = dest_images / f"{new_stem}{img_path.suffix.lower()}"
            new_label_path = dest_labels / f"{new_stem}.txt"

            # Copy image
            shutil.copy2(img_path, new_img_path)

            # Write label file (possibly empty)
            with new_label_path.open("w", encoding="utf-8") as f_out:
                if new_lines:
                    f_out.write("\n".join(new_lines) + "\n")
                else:
                    f_out.write("")


def write_merged_data_yaml(label_mapping: LabelMapping, dest_root: Path) -> None:
    """Write YOLOv8 data.yaml for the merged dataset."""
    # Build ordered names list from name_to_index
    names: list[str | None] = [None] * len(label_mapping.name_to_index)
    for name, idx in label_mapping.name_to_index.items():
        names[idx] = name

    data: dict[str, Any] = {
        "path": str(dest_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }

    out_path = dest_root / "data.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    LOG.info(f"\nWrote merged data.yaml to {out_path}")


def merge_yolo_datasets(
    mapping_path: Path,
    source_root: Path,
    dest_root: Path,
) -> None:
    """Merge multiple YOLOv8 datasets under source_root into a unified detection dataset.

    Segmentation labels in the source datasets are converted to detection labels
    by computing bounding boxes from polygons using shapely.
    """
    label_mapping = LabelMapping.from_yaml(mapping_path)

    ensure_dest_structure(dest_root)

    counters: dict[str, int] = {}

    # Discover datasets from source_root subdirectories
    for ds_dir in sorted(source_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_key = ds_dir.name
        if ds_key not in label_mapping.dataset_label_map:
            LOG.info(f"[INFO] No mapping for dataset '{ds_key}', skipping.")
            continue
        merge_one_dataset(ds_key, ds_dir, label_mapping, dest_root, counters)

    write_merged_data_yaml(label_mapping, dest_root)
    LOG.info("\nMerge completed.")
