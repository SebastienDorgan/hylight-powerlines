from pathlib import Path

import pytest

from hylight_powerlines.datasets.merge import coords_to_detection_bbox, merge_yolo_datasets


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_coords_to_detection_bbox_detection_passthrough() -> None:
    assert coords_to_detection_bbox([0.5, 0.5, 0.2, 0.3]) == [0.5, 0.5, 0.2, 0.3]


def test_coords_to_detection_bbox_segmentation_polygon_bbox() -> None:
    # bbox part is ignored when polygon is valid; polygon is a rectangle:
    # (0.1,0.2) .. (0.4,0.6) -> xc=0.25,yc=0.4,w=0.3,h=0.4
    coords = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.4,
        0.2,
        0.4,
        0.6,
        0.1,
        0.6,
    ]
    xc, yc, w, h = coords_to_detection_bbox(coords)
    assert xc == pytest.approx(0.25)
    assert yc == pytest.approx(0.4)
    assert w == pytest.approx(0.3)
    assert h == pytest.approx(0.4)


def test_merge_yolo_datasets_end_to_end(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.yaml"
    source_root = tmp_path / "external"
    dest_root = tmp_path / "merged"

    _write_text(
        mapping_path,
        """
classes:
  0: tower
  1: insulator

datasets:
  ds_a:
    pole: tower
    insulator: insulator
    dropme: null
""".lstrip(),
    )

    ds_a = source_root / "ds_a"
    _write_text(
        ds_a / "data.yaml",
        """
names:
  0: pole
  1: insulator
  2: dropme
""".lstrip(),
    )

    # train split
    _write_bytes(ds_a / "train/images/img1.jpg", b"x")
    _write_text(
        ds_a / "train/labels/img1.txt",
        # class 0 detection + class 1 segmentation + class 2 dropped
        "0 0.500000 0.500000 0.200000 0.200000\n"
        "1 0.0 0.0 0.0 0.0 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n"
        "2 0.2 0.2 0.1 0.1\n",
    )
    # image without label file => skipped
    _write_bytes(ds_a / "train/images/img_skip.jpg", b"x")

    # valid split -> val in merged dataset
    _write_bytes(ds_a / "valid/images/img2.jpg", b"x")
    _write_text(ds_a / "valid/labels/img2.txt", "")  # empty label file => keep background image

    merge_yolo_datasets(mapping_path=mapping_path, source_root=source_root, dest_root=dest_root)

    # Expected outputs exist
    assert (dest_root / "data.yaml").is_file()
    assert (dest_root / "train/images").is_dir()
    assert (dest_root / "train/labels").is_dir()
    assert (dest_root / "val/images").is_dir()
    assert (dest_root / "val/labels").is_dir()

    train_imgs = sorted((dest_root / "train/images").glob("*.jpg"))
    val_imgs = sorted((dest_root / "val/images").glob("*.jpg"))
    assert len(train_imgs) == 1
    assert len(val_imgs) == 1

    train_label = (dest_root / "train/labels" / f"{train_imgs[0].stem}.txt").read_text(
        encoding="utf-8"
    )
    # dropped class is not present; remapped classes are 0=tower, 1=insulator
    lines = [ln for ln in train_label.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")

    # segmentation bbox computed from polygon as in coords_to_detection_bbox test
    _, xc, yc, w, h = lines[1].split()
    assert float(xc) == pytest.approx(0.25, abs=1e-6)
    assert float(yc) == pytest.approx(0.4, abs=1e-6)
    assert float(w) == pytest.approx(0.3, abs=1e-6)
    assert float(h) == pytest.approx(0.4, abs=1e-6)

    val_label = (dest_root / "val/labels" / f"{val_imgs[0].stem}.txt").read_text(encoding="utf-8")
    assert val_label == ""
