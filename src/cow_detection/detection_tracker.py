from collections import deque
import numpy as np


class DetectionTracker:
    """
    Class to track the bounding boxes of detected cows
    and apply temporal filtering.

    Attributes:
      window_size (int): Number of frames to consider for
                         temporal filtering.
      bbox_history (deque): Deque to store the bounding box history.

    Methods:
      update: Updates the detection history with the new detection and returns
              the filtered bounding box.
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.bbox_history = deque(maxlen=window_size)

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Updates the detection history with the new detections and returns the
        filtered detections.

        Args:
            detections (np.ndarray): The detections to be added to the history.

        Returns:
            np.ndarray: The filtered detections.
        """
        if len(detections) == 0:
            return detections

        self.detection_history.append(detections)

        if len(self.detection_history) < 2:
            return detections

        weights = np.linspace(0.5, 1.0, len(self.detection_history))
        weights /= np.sum(weights)

        smoothed = np.zeros_like(detections)
        for i, (det, w) in enumerate(zip(self.detection_history, weights)):
            if det.shape == detections.shape:
                smoothed += w * det

        return smoothed
