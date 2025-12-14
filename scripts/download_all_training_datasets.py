import zipfile
from pathlib import Path

from hylight_powerlines.datasets.roboflow import ExportFormat, RoboflowDownloader


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP archive into a target directory."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def download_and_extract(
    workspace: str,
    project: str,
    target_name: str,
    export_format: ExportFormat = "yolov8",
    max_retries: int = 2,
    base_dir: Path | None = None,
) -> None:
    """Download a Roboflow dataset and extract it under data/external/<target_name>.

    Args:
        workspace: Roboflow workspace slug.
        project: Roboflow project slug.
        target_name: Local directory base name (without extension).
        export_format: Roboflow export format (eg. "yolov8").
        max_retries: Number of retries for stale export links (404).
        base_dir: Base directory for data; defaults to data/external.
    """
    if base_dir is None:
        base_dir = Path("data/external")

    base_dir.mkdir(parents=True, exist_ok=True)
    if (base_dir / target_name).exists():
        print(f"Already downloaded {target_name}")
        return

    zip_path = base_dir / f"{target_name}.zip"
    out_dir = base_dir / target_name

    print(f"\n=== Downloading {workspace}/{project} -> {zip_path} ===")
    downloader = RoboflowDownloader(
        workspace=workspace,
        project=project,
        export_format=export_format,
    )

    downloader.download_dataset(zip_path, max_retries=max_retries)
    extract_zip(zip_path, out_dir)
    zip_path.unlink(missing_ok=True)
    print(f"Extracted to: {out_dir}")


def main() -> None:
    """Download all selected training datasets from Roboflow.

    Usage:
        export ROBOFLOW_API_KEY=...  # from your Roboflow account
        uv run python scripts/download_all_training_datasets.py
    """
    datasets: list[tuple[str, str, str]] = [
        # Geometry-only / assets
        ("motus-aeaxm", "pylon-components", "pylon_components"),
        ("electricpoles", "electric-pole-detection-merged", "electric_pole_merged"),
        ("dy-cfoxw", "combine_pole", "combine_pole"),
        ("neec", "electrical-line", "electrical_line"),
        ("ps1-project", "dataset-ps1-v2-ia0e9", "ps1_v2"),
        # Defect-focused
        ("inspladdefectdetection", "insplad-defect-detection", "insplad_defect"),
        ("hehe-2moxl", "power-grid-inspection-dvwun", "power_grid_inspection"),
        ("yolov11-tasks", "damper-defect-detection", "damper_defect_detection"),
        ("abcd-fx82k", "insulator-lgust", "insulator_defect"),
    ]

    for workspace, project, target_name in datasets:
        download_and_extract(
            workspace=workspace,
            project=project,
            target_name=target_name,
            export_format="yolov8",
            max_retries=2,
        )


if __name__ == "__main__":
    main()
