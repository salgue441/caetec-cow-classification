import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ultralytics import YOLO
from cow_detection.configs import DetectionConfig
from cow_detection.detection_tracker import DetectionTracker


class YOLOInference:
    """
    YOLO inference class for object detection.

    Attributes:
        model (YOLO): YOLO model from ultralytics.
        detection_config (DetectionConfig): Detection configuration for the model.
        tracker (DetectionTracker): Detection tracker for the model.

    Methods:
        post_process(results) -> Any: Post-processes the raw detection results.
        predict_image(image_path: str, save_path: Optional[str] = None, conf_threshold: float = 0.3,
                     iou_threshold: float = 0.7, render_boxes: bool = True) -> Union[Tuple[np.ndarray, Any], Dict[str, Any]]:
            Performs object detection on the input image and returns either the annotated image and detection results
            or a dictionary with detection information.
        get_detection_summary(results) -> Dict[str, Any]:
            Returns a summary of the detection results including count and probabilities.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        weights_path: Optional[str] = None,
        detection_config: Optional[DetectionConfig] = None,
    ):
        """
        Initialize YOLO model with optional custom weights
        """
        if weights_path:
            self.model = YOLO(weights_path)
        else:
            self.model = YOLO(model_path)

        self.detection_config = detection_config or DetectionConfig()
        self.tracker = DetectionTracker(self.detection_config.temporal_window)

    def post_process(self, results):
        """
        Post-processes the raw detection results.

        Args:
            results: Raw detection results from the model.

        Returns:
            Any: Post-processed detection results.
        """
        if results.boxes is None or len(results.boxes) == 0:
            return results

        boxes = results.boxes.cpu().numpy()
        areas, confidences = [], []

        for box in boxes:
            xyxy = box.xyxy[0]
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            areas.append(area)
            confidences.append(float(box.conf[0]))

        areas = np.array(areas)
        confidences = np.array(confidences)
        mask = (areas >= self.detection_config.min_area) & (
            confidences >= self.detection_config.conf_threshold
        )

        results.boxes = results.boxes[mask]
        return results

    def get_detection_summary(self, results) -> Dict[str, Any]:
        """
        Returns a summary of the detection results including count and probabilities.

        Args:
            results: Post-processed detection results.

        Returns:
            Dict[str, Any]: Dictionary containing detection summary.
        """
        detection_summary = {"detection_count": 0, "probabilities": []}

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.cpu().numpy()
            detection_summary["detection_count"] = len(boxes)
            detection_summary["probabilities"] = [float(box.conf[0]) for box in boxes]

        return detection_summary

    def predict_image(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        conf_threshold: float = None,
        iou_threshold: float = None,
        render_boxes: bool = True,
    ) -> Union[Tuple[np.ndarray, Any], Dict[str, Any]]:
        """
        Performs object detection on the input image and returns either the annotated
        image and detection results or a dictionary with detection information.

        Args:
            image_path (str): Path to the input image.
            save_path (Optional[str]): Path to save the annotated image.
            conf_threshold (float, optional): Confidence threshold for detection.
                                              If None, uses the value from detection_config.
            iou_threshold (float, optional): IoU threshold for detection.
                                             If None, uses the value from detection_config.
            render_boxes (bool): Whether to render bounding boxes on the image or just return detection info.

        Returns:
            Union[Tuple[np.ndarray, Any], Dict[str, Any]]:
                If render_boxes=True: Annotated image and detection results.
                If render_boxes=False: Dictionary with detection summary.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else self.detection_config.conf_threshold
        )
        iou_threshold = (
            iou_threshold
            if iou_threshold is not None
            else self.detection_config.iou_threshold
        )

        results = self.model(
            image, verbose=False, conf=conf_threshold, iou=iou_threshold, max_det=20
        )[0]

        results = self.post_process(results)
        if not render_boxes:
            return self.get_detection_summary(results)

        annotated_image = image.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.cpu().numpy()
            confidences = [float(box.conf[0]) for box in boxes]
            sorted_indices = np.argsort(confidences)[::-1]

            for idx in sorted_indices:
                box = boxes[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"cow {confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                (label_width, label_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                label_patch = (
                    np.ones(
                        (label_height + 2 * baseline, label_width + 2 * baseline, 3),
                        dtype=np.uint8,
                    )
                    * 255
                )

                cv2.putText(
                    label_patch,
                    label,
                    (baseline, label_height),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )

                label_y = max(y1 - label_height - baseline, 0)
                label_x = max(x1, 0)

                if (
                    label_y + label_height <= image.shape[0]
                    and label_x + label_width <= image.shape[1]
                ):
                    try:
                        roi = annotated_image[
                            label_y : label_y + label_height + 2 * baseline,
                            label_x : label_x + label_width + 2 * baseline,
                        ]

                        alpha = 0.7
                        cv2.addWeighted(label_patch, alpha, roi, 1 - alpha, 0, roi)

                        annotated_image[
                            label_y : label_y + label_height + 2 * baseline,
                            label_x : label_x + label_width + 2 * baseline,
                        ] = roi
                    except Exception as e:
                        cv2.putText(
                            annotated_image,
                            label,
                            (label_x, label_y + label_height),
                            font,
                            font_scale,
                            (0, 0, 0),
                            thickness,
                        )

        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(save_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            )

        return annotated_image, results
