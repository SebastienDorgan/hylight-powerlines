import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import YOLO

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


@dataclass
class YoloFineTuner:
    """Wrapper around Ultralytics YOLO for fine-tuning on a custom dataset.

    This class is intended for Roboflow-exported YOLOv11/YOLOv8 datasets and
    only covers training and validation, not inference.

    Attributes:
        dataset_root: Directory containing ``data.yaml`` (Roboflow export root).
        model_name: Pretrained YOLO checkpoint to start from, such as
            "yolo11s.pt", "yolo11m.pt" or "yolov8s.pt".
        epochs: Number of training epochs.
        img_size: Square input image size used during training.
        batch: Batch size or "auto" for automatic tuning.
        device: Device selector understood by Ultralytics, for example:
            "0" for first GPU, "cpu" for CPU only, or None for auto.
        project: Name of the training project folder created by Ultralytics.
        run_name: Name of the specific run inside the project.
        seed: Random seed for reproducibility.
        overrides: Extra Ultralytics training overrides. Keys must be valid
            YOLO.train arguments (for example: "lr0", "lrf", "patience").
    """

    dataset_root: Path
    model_name: str = "yolov8s.pt"
    epochs: int = 50
    img_size: int = 1024
    batch: int | str = "auto"
    device: str | None = None
    project: str = "runs"
    run_name: str = "yolo8-finetune"
    seed: int = 0
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()

        data_yaml = self.dataset_root / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"Could not find data.yaml in dataset_root={self.dataset_root}")

        self.data_yaml = data_yaml

        LOG.info("Initialized YoloFineTuner")
        LOG.info("  dataset_root: %s", self.dataset_root)
        LOG.info("  data.yaml:    %s", self.data_yaml)
        LOG.info("  model_name:   %s", self.model_name)

    def train(self) -> Path:
        """Run YOLO fine-tuning on the dataset.

        Returns:
            Path to the best model checkpoint (``best.pt``) produced by training.

        Raises:
            RuntimeError: If the expected ``best.pt`` file is not found.
        """
        LOG.info("Starting YOLO training")
        LOG.info("  epochs:   %s", self.epochs)
        LOG.info("  img_size: %s", self.img_size)
        LOG.info("  batch:    %s", self.batch)
        LOG.info("  device:   %s", self.device)
        LOG.info("  project:  %s", self.project)
        LOG.info("  run_name: %s", self.run_name)

        # Load base model
        model = YOLO(self.model_name)  # type: ignore[no-untyped-call]

        # Map "auto" to 0 (Ultralytics convention for automatic batch size)
        if isinstance(self.batch, str):
            if self.batch != "auto":
                raise ValueError(
                    f"Invalid batch value {self.batch!r}. "
                    "Use an int/float or 'auto'."
                )
            batch_value: int | float = 0
        else:
            batch_value = self.batch

        # Base arguments for YOLO.train
        train_args: dict[str, Any] = {
            "data": str(self.data_yaml),
            "epochs": self.epochs,
            "imgsz": self.img_size,
            "batch": batch_value,
            "device": self.device,
            "project": self.project,
            "name": self.run_name,
            "seed": self.seed,
        }

        # Allow the caller to override any YOLO argument
        if self.overrides:
            LOG.info("Applying extra overrides: %s", self.overrides)
            train_args.update(self.overrides)

        # Run training
        results = model.train(**train_args)  # type: ignore[no-untyped-call]
        LOG.info("Training completed. Ultralytics results: %s", results)

        run_dir = Path(self.project) / self.run_name
        best_ckpt = run_dir / "weights" / "best.pt"

        if not best_ckpt.is_file():
            raise RuntimeError(
                f"Training did not produce best.pt at expected location: {best_ckpt}"
            )

        LOG.info("Best model checkpoint: %s", best_ckpt)
        return best_ckpt

    def validate(self, weights: Path | None = None) -> Any:
        """Run YOLO validation on the dataset.

        Args:
            weights: Optional path to a trained checkpoint. If None, it uses
                ``model_name`` (the pretrained checkpoint) for validation.

        Returns:
            Ultralytics validation results object.
        """
        if weights is not None:
            LOG.info("Running validation with weights: %s", weights)
            model = YOLO(str(weights))  # type: ignore[no-untyped-call]
        else:
            LOG.info(
                "Running validation with original model_name checkpoint: %s",
                self.model_name,
            )
            model = YOLO(self.model_name)  # type: ignore[no-untyped-call]

        results = model.val(data=str(self.data_yaml))  # type: ignore[no-untyped-call]
        LOG.info("Validation completed. Results: %s", results)
        return results


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@dataclass
class YoloPredictor:
    """Wrapper around Ultralytics YOLO for inference only.

    This class is independent from training. It loads a given checkpoint and
    exposes convenience methods for prediction.

    Attributes:
        weights: Path to a trained checkpoint, such as ``best.pt``.
        device: Device selector understood by Ultralytics, for example:
            "0" for first GPU, "cpu" for CPU only, or None for auto.
        img_size: Optional input image size (square). If None, Ultralytics
            will use its default.
        conf: Default confidence threshold used for predictions.
    """

    weights: Path
    device: str | None = None
    img_size: int | None = None
    conf: float = 0.25

    def __post_init__(self) -> None:
        self.weights = Path(self.weights).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"weights file not found: {self.weights}")

        LOG.info("Loading YOLO model for inference: %s", self.weights)
        self._model = YOLO(str(self.weights))  # type: ignore[no-untyped-call]

    def predict_on_folder(
        self,
        images_dir: Path | str,
        save: bool = True,
        save_txt: bool = False,
        save_conf: bool = False,
    ) -> Any:
        """Run inference on all images in a folder.

        Args:
            images_dir:
                Directory containing images to run inference on.
            save:
                If True, Ultralytics saves annotated images under its default
                runs/predict directory.
            save_txt:
                If True, Ultralytics saves label text files for each image.
            save_conf:
                If True and ``save_txt`` is True, confidences are included
                in label files.

        Returns:
            Ultralytics prediction results object.
        """
        images_dir = Path(images_dir).expanduser().resolve()
        if not images_dir.is_dir():
            raise NotADirectoryError(f"images_dir is not a directory: {images_dir}")

        LOG.info("Running prediction on folder: %s", images_dir)

        predict_args: dict[str, Any] = {
            "source": str(images_dir),
            "save": save,
            "save_txt": save_txt,
            "save_conf": save_conf,
            "conf": self.conf,
            "device": self.device,
        }
        if self.img_size is not None:
            predict_args["imgsz"] = self.img_size

        results = self._model.predict(  # type: ignore[no-untyped-call]
            **predict_args
        )
        LOG.info("Prediction on folder completed.")
        return results

    def predict_on_image(
        self,
        image_path: Path | str,
        save: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
    ) -> Any:
        """Run inference on a single image.

        Args:
            image_path:
                Path to the input image.
            save:
                If True, Ultralytics saves an annotated copy of the image.
            save_txt:
                If True, Ultralytics saves a label text file for the image.
            save_conf:
                If True and ``save_txt`` is True, confidences are included
                in the label file.

        Returns:
            Ultralytics prediction results object.
        """
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"image file not found: {image_path}")

        LOG.info("Running prediction on image: %s", image_path)

        predict_args: dict[str, Any] = {
            "source": str(image_path),
            "save": save,
            "save_txt": save_txt,
            "save_conf": save_conf,
            "conf": self.conf,
            "device": self.device,
        }
        if self.img_size is not None:
            predict_args["imgsz"] = self.img_size

        results = self._model.predict(  # type: ignore[no-untyped-call]
            **predict_args
        )
        LOG.info("Prediction on image completed.")
        return results
