from pathlib import Path

import pytest
import yaml

from hylight_powerlines.datasets.curate import (
    _build_id_map,
    _dataset_prefix,
    _load_class_names,
    _remap_label_file,
    curate_detection_dataset,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _yaml_dump(data: object) -> str:
    dumped = yaml.safe_dump(data, sort_keys=False)
    if dumped is None:
        return ""
    if isinstance(dumped, bytes):
        return dumped.decode("utf-8")
    return dumped


def test_load_class_names_list_and_dict(tmp_path: Path) -> None:
    p_list = tmp_path / "list.yaml"
    _write_text(
        p_list,
        _yaml_dump({"names": ["a", "b", "c"]}),
    )
    assert _load_class_names(p_list) == ["a", "b", "c"]

    p_dict = tmp_path / "dict.yaml"
    _write_text(
        p_dict,
        _yaml_dump({"names": {"1": "b", "0": "a"}}),
    )
    assert _load_class_names(p_dict) == ["a", "b"]


def test_build_id_map_and_prefix_and_remap(tmp_path: Path) -> None:
    id_map, new_names = _build_id_map(["tower", "insulator", "spacer"], drop_classes=["spacer"])
    assert new_names == ["tower", "insulator"]
    assert id_map == {0: 0, 1: 1, 2: None}

    assert _dataset_prefix("ds_train_000001", "train") == "ds"
    assert _dataset_prefix("ds_val_000001", "train") is None

    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    _write_text(src, "2 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.2 0.2\n")
    assert _remap_label_file(src, dst, id_map) is True
    # class 2 dropped, class 1 remapped to 1
    assert dst.read_text(encoding="utf-8").strip() == "1 0.1 0.1 0.2 0.2"


def test_curate_detection_dataset_end_to_end(tmp_path: Path) -> None:
    source_root = tmp_path / "merged"
    dest_root = tmp_path / "curated"

    _write_text(
        source_root / "data.yaml",
        _yaml_dump(
            {
                "path": str(source_root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "tower", 1: "insulator", 2: "spacer"},
                "nc": 3,
            }
        ),
    )

    # train: keep one image, skip one (missing label file), drop one (wrong prefix)
    (source_root / "train/images").mkdir(parents=True, exist_ok=True)
    (source_root / "train/labels").mkdir(parents=True, exist_ok=True)
    kept_img = source_root / "train/images/ds_train_000001.jpg"
    kept_img.write_bytes(b"x")
    _write_text(source_root / "train/labels/ds_train_000001.txt", "2 0.5 0.5 0.2 0.2\n")

    missing_lbl_img = source_root / "train/images/ds_train_000002.jpg"
    missing_lbl_img.write_bytes(b"x")

    other_prefix_img = source_root / "train/images/other_train_000001.jpg"
    other_prefix_img.write_bytes(b"x")
    _write_text(source_root / "train/labels/other_train_000001.txt", "1 0.1 0.1 0.2 0.2\n")

    curate_detection_dataset(
        source_root=source_root,
        dest_root=dest_root,
        keep_sources=["ds"],
        drop_classes=["spacer"],
    )

    assert (dest_root / "data.yaml").is_file()
    assert (dest_root / "train/images").is_dir()
    assert (dest_root / "train/labels").is_dir()

    imgs = list((dest_root / "train/images").glob("*.jpg"))
    assert [p.name for p in imgs] == ["ds_train_000001.jpg"]

    lbl = (dest_root / "train/labels/ds_train_000001.txt").read_text(encoding="utf-8")
    # spacer dropped => empty label file still exists
    assert lbl == ""

    cfg = yaml.safe_load((dest_root / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["nc"] == 2
    assert list(cfg["names"].values()) == ["tower", "insulator"]

    with pytest.raises(RuntimeError):
        curate_detection_dataset(
            source_root=source_root,
            dest_root=dest_root,
            keep_sources=["ds"],
            drop_classes=["spacer"],
        )
