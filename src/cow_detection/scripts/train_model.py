import argparse
from cow_detection.configs import TrainingConfig, DetectionConfig
from cow_detection.trainer import YOLOTrainer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for cow detection")
    parser.add_argument(
        "--data_yaml", type=str, required=True, help="Path to data configuration YAML"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )

    args = parser.parse_args()

    detection_config = DetectionConfig(
        conf_threshold=0.4,
        iou_threshold=0.70,
        min_area=300,
        max_overlap_ratio=0.65,
        enable_temporal=False,
    )

    training_config = TrainingConfig(
        epochs=args.epochs, image_size=640, batch_size=args.batch_size, dropout=0.3
    )

    trainer = YOLOTrainer(
        model_path="yolov9c.pt",
        training_config=training_config,
        detection_config=detection_config,
    )

    model, results = trainer.train(
        args.data_yaml, epochs=args.epochs, batch_size=args.batch_size
    )

    # Optional: save model or log results
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
