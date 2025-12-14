from pathlib import Path

from hylight_powerlines.datasets.merge import merge_yolo_datasets

if __name__ == "__main__":
    root_path = Path("data")
    merge_yolo_datasets(
        mapping_path=root_path / "detection_mapping.yaml",
        source_root=root_path / "external",
        dest_root=root_path / "detection_dataset",
    )
