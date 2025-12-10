from pathlib import Path

from ultralytics import YOLO

from hylight_powerlines.yolo import YoloFineTuner


def main() -> None:
    # Root of your merged dataset (must contain data.yaml)
    dataset_root = Path("data/detection_dataset_curated")

    tuner = YoloFineTuner(
        dataset_root=dataset_root,
        # Choose the family you want to use:
        # model_name="yolo11s.pt",   # YOLOv11 small
        model_name="yolov8s.pt",  # YOLOv8 small (default in your class)
        epochs=100,  # tweak as needed
        img_size=1024,  # matches your merge resolution
        batch="auto",  # let Ultralytics decide
        device="0",  # first GPU; use "cpu" if no GPU
        project="runs/powerlines",  # where runs will be stored
        run_name="yolov8s-6cls",  # folder under project
        seed=0,
        overrides={
            # Optional but often useful overrides:
            "lr0": 0.02,  # initial learning rate
            "lrf": 0.1,  # final LR fraction
            "cos_lr": True,  # smoother LR decay
            "patience": 20,  # early stopping patience (epochs)
            "weight_decay": 0.0005,
            "degrees": 5.0,  # small extra rotation (on top of any in dataset)
            "flipud": 0.0,  # disable vertical flip (unnatural for powerlines)
            "fliplr": 0.5,  # keep horizontal flip
            "mosaic": 1.0,  # keep mosaic
        },
    )

    # Train
    best_ckpt = tuner.train()

    # (Optional) global validation via your wrapper
    tuner.validate(weights=best_ckpt)

    # Reload best checkpoint with Ultralytics YOLO
    model = YOLO(str(best_ckpt))

    # Run validation to get metrics object (needed for per-class stats)
    metrics = model.val(
        data=str(dataset_root / "data.yaml"),
        imgsz=1024,
        split="val",
    )

    # Per-class metrics: P, R, mAP50, mAP50-95
    for i, name in model.names.items():
        p, r, ap50, ap5095 = metrics.box.class_result(i)
        print(f"{i:>2} {name:>12}: P={p:.3f} R={r:.3f} mAP50={ap50:.3f} mAP50-95={ap5095:.3f}")


if __name__ == "__main__":
    main()
