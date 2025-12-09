from pathlib import Path

from hylight_powerlines.curate import curate_detection_dataset


def main() -> None:
    project_root = Path("data")
    source_root = project_root / "detection_dataset"
    dest_root = project_root / "detection_dataset_curated"

    keep_sources = [
        "electric_pole_merged",
        "power_grid_inspection",
        "pylon_components",
        "damper_defect_detection",
        "electrical_line",
    ]

    curate_detection_dataset(
        source_root=source_root,
        dest_root=dest_root,
        keep_sources=keep_sources,
        drop_class="spacer",
    )


if __name__ == "__main__":
    main()
