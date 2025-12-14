from pathlib import Path

from hylight_powerlines.yolo.wrappers import YoloFineTuner


def main() -> None:
    dataset_root = Path("data/defect_detection_dataset")

    tuner = YoloFineTuner(
        dataset_root=dataset_root,
        model_name="yolov8s.pt",  # or "yolo11s.pt" if you prefer
        epochs=100,
        img_size=1024,
        batch=0,  # 0 => auto-batch with your patched class
        device="0",  # or "cpu"
        project="runs/defects",
        run_name="yolov8s-defects-4cls",
        seed=0,
        overrides={
            "lr0": 0.01,
            "lrf": 0.1,
            "patience": 20,
            "weight_decay": 0.0005,
            "flipud": 0.0,
            "fliplr": 0.5,
        },
    )

    best_ckpt = tuner.train()
    tuner.validate(weights=best_ckpt)


if __name__ == "__main__":
    main()
