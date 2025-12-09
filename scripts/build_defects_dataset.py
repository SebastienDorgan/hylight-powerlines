from pathlib import Path

from hylight_powerlines.datasets import merge_yolo_datasets


def main() -> None:
    root_path = Path("data")
    merge_yolo_datasets(
        mapping_path=root_path/"defect_label_mapping.yaml",
        source_root=root_path / "external",
        dest_root=root_path / "detection_dataset",
    )


if __name__ == "__main__":
    main()
