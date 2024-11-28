from cow_detection.detection_tracker import DetectionTracker
from cow_detection.configs import DetectionConfig

import logging
from typing import Union, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLOBase:
    """
    Loads the selected YOLO model from ultralytics and initializes the detection tracker.

    Attributes:
      model_path (Union[str, Path]): Path to the YOLO model configuration file.
      detection_config (Optional[DetectionConfig]): Configuration for the detection tracker.
      tracker (DetectionTracker): Detection tracker for the model.
      model (YOLO): YOLO model from ultralytics.

    Methods:
      _load_model(weights_path: Optional[Path]) -> bool: Loads the YOLO model from ultralytics with the specified weights.
      _compute_intersection(box1: np.ndarray, box2: np.ndarray) -> float: Computes the intersection area between two bounding boxes.
      post_process_detection(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: Post-processes the raw detections by applying NMS and size filtering.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        detection_config: Optional[DetectionConfig] = None,
    ):
        self.model_path = Path(model_path)
        self.detection_config = detection_config or DetectionConfig()
        self.tracker = DetectionTracker(self.detection_config.temporal_window)
        self.model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, weights_path: Optional[Path] = None) -> bool:
        """
        Loads the YOLO model from ultralytics with the specified weights.

        Args:
            weights_path (Optional[Path]): Path to the weights file to load.

        Returns:
            bool: True if the model was successfully loaded, False otherwise.
        """

        try:
            if weights_path is not None and weights_path.exists():
                self.model = YOLO(str(weights_path))
                self.model.to(self.device)
                self.model.load(str(weights_path))

                logging.info(
                    f"Loaded base model from {self.model_path} with weights from {weights_path}"
                )

            else:
                self.model = YOLO(str(self.model_path))
                self.model.to(self.device)

                logging.info(f"Loaded base model from {self.model_path}")

            return True

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

    def _compute_intersection(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Computes the intersection area between two bounding boxes.

        Args:
            box1 (np.ndarray): The first bounding box.
            box2 (np.ndarray): The second bounding box.

        Returns:
            float: The intersection area.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        return max(0, x2 - x1) * max(0, y2 - y1)

    def post_process_detection(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-processes the raw detections by applying NMS and size filtering.

        Args:
            boxes (np.ndarray): The detected bounding boxes.
            scores (np.ndarray): The detection scores.
            class_ids (np.ndarray): The detected class IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The post-processed detections.
        """

        if len(boxes) == 0:
            return boxes, scores, class_ids

        if self.detection_config.enable_nms:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.detection_config.conf_threshold,
                self.detection_config.iou_threshold,
            )
            boxes = boxes[indices.flatten()]
            scores = scores[indices.flatten()]
            class_ids = class_ids[indices.flatten()]

        if self.detection_config.enable_size_filter:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            mask = areas >= self.detection_config.min_area
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

        final_boxes, final_scores, final_class_ids = [], [], []
        for idx in range(len(boxes)):
            overlap_too_high = False
            box_area = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])

            for j in range(len(final_boxes)):
                intersection = self._compute_intersection(boxes[idx], final_boxes[j])
                if intersection / box_area > self.detection_config.max_overlap_ratio:
                    overlap_too_high = True
                    break

            if not overlap_too_high:
                final_boxes.append(boxes[idx])
                final_scores.append(scores[idx])
                final_class_ids.append(class_ids[idx])

        return np.array(final_boxes), np.array(final_scores), np.array(final_class_ids)
