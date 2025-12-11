from pathlib import Path

from hylight_powerlines.yolo import YoloPredictor

if __name__ == "__main__":

    predictor = YoloPredictor(
        weights=Path("runs/powerlines/yolov8s-6cls3/weights/best.pt"),
        img_size=1024
    )

    results = predictor.predict_on_folder(
        images_dir=Path("assets/images"),
        save_txt=True
    )
    for _ in results:
        pass
