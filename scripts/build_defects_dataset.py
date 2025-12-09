from pathlib import Path

from hylight_powerlines.datasets import merge_yolo_datasets


def main() -> None:
    root_path = Path("data")
    merge_yolo_datasets(
        # File lives at data/defect_mapping.yaml
        mapping_path=root_path/"defect_mapping.yaml",
        source_root=root_path / "external",
        # Write to a dedicated defect dataset folder
        dest_root=root_path / "defect_detection_dataset",
    )


if __name__ == "__main__":
    main()
