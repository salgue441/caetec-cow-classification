from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    """
    Defines the training configuration for the model. 

    Attributes:
      epochs (int): Number of epochs to train the model.
      lr (float): Learning rate for the optimizer.
      image_size (int): Size of the image to be used for training.
      batch_size (int): Batch size for training.
      device (str): Device to use for training.
      patience (int): Number of epochs to wait before early stopping.
      dropout (float): Dropout rate for the model.
      augment (bool): Whether to use data augmentation.
      mosaic (float): Probability of using mosaic augmentation.
      mixup (float): Probability of using mixup augmentation.
      copy_paste (float): Probability of using copy-paste augmentation.
      degrees (float): Maximum rotation angle for data augmentation.
      translate (float): Maximum translation for data augmentation.
      scale (float): Maximum scaling factor for data augmentation.
      shear (float): Maximum shear angle for data augmentation.
      perspective (float): Maximum perspective distortion for data augmentation.
      flipud (float): Probability of flipping the image vertically.
      fliplr (float): Probability of flipping the image horizontally.
      hsv_h (float): Maximum hue change for HSV augmentation.
      hsv_s (float): Maximum saturation change for HSV augmentation.
      hsv_v (float): Maximum value change for HSV augmentation.
    """

    epochs: int = 100
    lr: float = 0.001
    image_size: int = 250
    batch_size: int = 16
    device: str = "0"
    patience: int = 20
    dropout: float = 0.2
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.3
    copy_paste: float = 0.3
    degrees: float = 45.0
    translate: float = 0.2
    scale: float = 0.5
    shear: float = 10.0
    perspective: float = 0.0005
    flipud: float = 0.5
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

@dataclass
class DetectionConfig:
    """
    Defines the detection configuration for the model.

    Attributes:
      conf_threshold (float): Confidence threshold for object detection.
      iou_threshold (float): IOU threshold for non-maximum suppression.
      min_area (int): Minimum area of the bounding box.
      max_overlap_ratio (float): Maximum overlap ratio for temporal filtering.
      temporal_window (int): Number of frames to consider for 
                             temporal filtering.
      enable_nms (bool): Whether to use non-maximum suppression.
      enable_temporal (bool): Whether to use temporal filtering.
      enable_size_filter (bool): Whether to use size filtering.
    """

    conf_threshold: float = 0.30
    iou_threshold: float = 0.60
    min_area: int = 500
    max_overlap_ratio: float = 0.8
    temporal_window: int = 3
    enable_nms: bool = True
    enable_temporal: bool = True
    enable_size_filter: bool = True

@dataclass
class ConsistencyConfig:
    """
    Defines the consistency configuration for the model.

    Attributes:
      num_inference_passes (int): Number of inference passes for consistency.
      confidence_threshold (float): Confidence threshold for consistency.
      consistency_threshold (float): Consistency threshold for consistency.
      enable_tta (bool): Whether to use test-time augmentation.
      tta_scales (List[float]): List of scales for test-time augmentation.
      tta_flips (bool): Whether to use flips for test-time augmentation.
      min_detection_votes (int): Minimum number of detection votes.
      calibrate_confidence (bool): Whether to calibrate confidence.
      confidence_scaling_factor (float): Scaling factor for confidence.
      min_relative_size (float): Minimum relative size of the bounding box.
      max_relative_size (float): Maximum relative size of the bounding box.
    """

    num_inference_passes: int = 3
    confidence_threshold: float = 0.3
    consistency_threshold: float = 0.7
    enable_tta: bool = True
    tta_scales: Optional[List[float]] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    tta_flips: bool = True
    min_detection_votes: int = 2
    calibrate_confidence: bool = True
    confidence_scaling_factor: float = 1.2
    min_relative_size: float = 0.01
    max_relative_size: float = 0.8