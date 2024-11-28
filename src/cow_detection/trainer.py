from cow_detection.yolo_base import YOLOBase
from cow_detection.configs import TrainingConfig, DetectionConfig

import torch
from pathlib import Path
from typing import Union, Optional
import yaml


class YOLOTrainer(YOLOBase):
    """
    Trainer class for YOLO model.

    Attributes:
        model_path (Union[str, Path]): Path to the model file.
        training_config (TrainingConfig): Training configuration for the model.
        detection_config (DetectionConfig): Detection configuration for the model.

    Methods:
      train(data_yaml_path: Union[str, Path], epochs: Optional[int] = 50, image_size: Optional[int] = 320, batch_size: Optional[int] = 16) -> Tuple[Model, dict]: Trains the model using the provided data and configuration.
      _create_data_config(train_path: Path, val_path: Path) -> dict: Creates the data configuration for training the model.
      _verify_data_paths(train_path: Path, val_path: Path): Verifies that the training and validation directories exist.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        training_config: Optional[TrainingConfig] = None,
        detection_config: Optional[DetectionConfig] = None,
    ):
        super().__init__(model_path, detection_config)

        self.training_config = training_config or TrainingConfig()
        self._load_model()

    def train(
        self,
        data_yaml_path: Union[str, Path],
        epochs: Optional[int] = 50,
        image_size: Optional[int] = 320,
        batch_size: Optional[int] = 16,
    ):
        """
        Trains the model using the provided data and configuration.

        Args:
            data_yaml_path (Union[str, Path]): Path to the data YAML file.
            epochs (Optional[int]): Number of epochs to train the model for.
            image_size (Optional[int]): Size of the input image.
            batch_size (Optional[int]): Batch size for training.

        Returns:
            Tuple[Model, dict]: Trained model and training results
        """

        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()

        else:
            print("Using CPU")
            return

        data_yaml_path = Path(data_yaml_path).resolve()
        data_dir = data_yaml_path.parent

        train_path = (data_dir / "train" / "images").resolve()
        val_path = (data_dir / "valid" / "images").resolve()
        self._verify_data_paths(train_path, val_path)

        data_config = self._create_data_config(train_path, val_path)
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data_config, f, sort_keys=False)

        torch.cuda.set_device(0)
        self.results = self.model.train(
            data=data_yaml_path,
            epochs=epochs or self.training_config.epochs,
            imgsz=image_size or self.training_config.image_size,
            batch=batch_size or self.training_config.batch_size,
            device=self.training_config.device,
            patience=self.training_config.patience,
            save=True,
            plots=True,
            augment=self.training_config.augment,
            dropout=self.training_config.dropout,
            **{
                k: v
                for k, v in vars(self.training_config).items()
                if k in ["mosaic", "mixup", "copy_paste"]
            },
        )

        return self.model, self.results

    def _create_data_config(self, train_path: Path, val_path: Path) -> dict:
        """
        Creates the data configuration for training the model.

        Args:
            train_path (Path): Path to the training data directory.
            val_path (Path): Path to the validation data directory.

        Returns:
            dict: Data configuration for training the model.
        """

        config = {
            "train": str(train_path),
            "val": str(val_path),
            "nc": 1,
            "names": ["cow"],
        }

        augment_params = {
            k: v
            for k, v in vars(self.training_config).items()
            if k
            in [
                "mosaic",
                "mixup",
                "copy_paste",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "flipud",
                "fliplr",
                "hsv_h",
                "hsv_s",
                "hsv_v",
            ]
        }

        config.update(augment_params)
        return config

    def _verify_data_paths(self, train_path: Path, val_path: Path):
        """
        Verifies that the training and validation directories exist.

        Args:
            train_path (Path): Path to the training data directory.
            val_path (Path): Path to the validation data directory.

        Raises:
            FileNotFoundError: If the training or validation directories do not exist
        """

        if not train_path.exists():
            raise FileNotFoundError(f"Training directory not found: {train_path}")

        if not val_path.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_path}")
