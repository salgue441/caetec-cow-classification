# Inference API Documentation

This document provides detailed reference information for the cow detection inference API.

## Overview

The inference API allows you to detect cows in images programmatically using the YOLOv9-based detection model. It supports both visual output (with bounding boxes) and data-only output modes.

## Classes

### YOLOInference

The main class for performing object detection on images.

```python
from cow_detection.inference import YOLOInference
```

#### Constructor

```python
YOLOInference(
    model_path: str = "yolov8m.pt",
    weights_path: Optional[str] = None,
    detection_config: Optional[DetectionConfig] = None
)
```

**Parameters:**

| Parameter          | Type                      | Description                 | Default      |
| ------------------ | ------------------------- | --------------------------- | ------------ |
| `model_path`       | str                       | Path to the base YOLO model | "yolov8m.pt" |
| `weights_path`     | Optional[str]             | Path to custom weights      | None         |
| `detection_config` | Optional[DetectionConfig] | Detection configuration     | None         |

**Example:**

```python
from cow_detection.inference import YOLOInference
from cow_detection.configs import DetectionConfig

# Create with default configuration
detector = YOLOInference(
    model_path="yolov9c.pt",
    weights_path="runs/detect/train14/weights/best.pt"
)

# Create with custom configuration
config = DetectionConfig(
    conf_threshold=0.5,
    iou_threshold=0.6,
    min_area=400
)
detector = YOLOInference(
    model_path="yolov9c.pt",
    weights_path="runs/detect/train14/weights/best.pt",
    detection_config=config
)
```

#### Methods

##### predict_image

```python
predict_image(
    image_path: str,
    save_path: Optional[str] = None,
    conf_threshold: float = None,
    iou_threshold: float = None,
    render_boxes: bool = True
) -> Union[Tuple[np.ndarray, Any], Dict[str, Any]]
```

Performs object detection on an input image and returns either the annotated image and detection results (when `render_boxes=True`) or a dictionary with detection information (when `render_boxes=False`).

**Parameters:**

| Parameter        | Type          | Description                                   | Default     |
| ---------------- | ------------- | --------------------------------------------- | ----------- |
| `image_path`     | str           | Path to the input image                       | Required    |
| `save_path`      | Optional[str] | Path to save the annotated image              | None        |
| `conf_threshold` | float         | Confidence threshold for detection            | From config |
| `iou_threshold`  | float         | IoU threshold for detection                   | From config |
| `render_boxes`   | bool          | Whether to render bounding boxes on the image | True        |

**Returns:**

- When `render_boxes=True`: A tuple of `(annotated_image, results)` where:

  - `annotated_image` is a numpy array containing the image with bounding boxes
  - `results` is the raw detection results object

- When `render_boxes=False`: A dictionary with:
  - `detection_count`: Number of detected objects
  - `probabilities`: List of confidence scores for each detection

**Example with visual output:**

```python
# Process and save annotated image
image, results = detector.predict_image(
    image_path="path/to/image.jpg",
    save_path="output.jpg",
    render_boxes=True
)

# Display the result with matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.title(f"Found {len(results.boxes)} cows")
plt.show()
```

**Example with data-only output:**

```python
# Get detection data without rendering
detection_data = detector.predict_image(
    image_path="path/to/image.jpg",
    render_boxes=False
)

print(f"Found {detection_data['detection_count']} cows")
for i, prob in enumerate(detection_data['probabilities'], 1):
    print(f"Cow {i}: confidence {prob:.2f}")
```

##### post_process

```python
post_process(results) -> Any
```

Post-processes the raw detection results based on the detection configuration.

**Parameters:**

| Parameter | Type | Description                          |
| --------- | ---- | ------------------------------------ |
| `results` | Any  | Raw detection results from the model |

**Returns:**

- Filtered detection results based on confidence threshold and minimum area

**Note:** This method is generally used internally but can be called directly for custom processing pipelines.

##### get_detection_summary

```python
get_detection_summary(results) -> Dict[str, Any]
```

Returns a summary of the detection results including count and probabilities.

**Parameters:**

| Parameter | Type | Description                      |
| --------- | ---- | -------------------------------- |
| `results` | Any  | Post-processed detection results |

**Returns:**

- Dictionary with:
  - `detection_count`: Number of detected objects
  - `probabilities`: List of confidence scores for each detection

### DetectionConfig

Configuration class for detection parameters.

```python
from cow_detection.configs import DetectionConfig
```

#### Constructor

```python
DetectionConfig(
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    min_area: float = 100,
    max_overlap_ratio: float = 0.7,
    enable_temporal: bool = False,
    temporal_window: int = 5
)
```

**Parameters:**

| Parameter           | Type  | Description                              | Default |
| ------------------- | ----- | ---------------------------------------- | ------- |
| `conf_threshold`    | float | Confidence threshold for detection       | 0.3     |
| `iou_threshold`     | float | IoU threshold for detection              | 0.5     |
| `min_area`          | float | Minimum area for valid detections        | 100     |
| `max_overlap_ratio` | float | Maximum overlap ratio between detections | 0.7     |
| `enable_temporal`   | bool  | Enable temporal consistency              | False   |
| `temporal_window`   | int   | Window size for temporal consistency     | 5       |

**Example:**

```python
from cow_detection.configs import DetectionConfig

# Create a custom configuration
config = DetectionConfig(
    conf_threshold=0.4,
    iou_threshold=0.7,
    min_area=300,
    max_overlap_ratio=0.65,
    enable_temporal=False
)
```

## Working with Detection Results

The `results` object returned by `predict_image` (when `render_boxes=True`) contains the detection information in the `boxes` attribute:

```python
image, results = detector.predict_image("path/to/image.jpg")

if results.boxes is not None and len(results.boxes) > 0:
    boxes = results.boxes.cpu().numpy()

    for box in boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get confidence score
        confidence = float(box.conf[0])

        print(f"Bounding box: ({x1}, {y1}, {x2}, {y2}), Confidence: {confidence:.2f}")
```

## Processing Multiple Images

Example of processing multiple images in a directory:

```python
import os
from pathlib import Path
from cow_detection.inference import YOLOInference

detector = YOLOInference(weights_path="path/to/weights.pt")
input_dir = "data/images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Process all JPG images in the directory
for image_path in Path(input_dir).glob("*.jpg"):
    output_path = os.path.join(output_dir, f"{image_path.stem}_detected.jpg")

    # Option 1: Visual output
    image, results = detector.predict_image(
        image_path=str(image_path),
        save_path=output_path,
        render_boxes=True
    )

    # Option 2: Data-only output
    summary = detector.predict_image(
        image_path=str(image_path),
        render_boxes=False
    )

    print(f"{image_path.name}: {summary['detection_count']} detections")
```

## Advanced Usage

### Using Custom Thresholds for Specific Images

```python
# Default configuration for most images
detector = YOLOInference(weights_path="path/to/weights.pt")

# Process regular images with default settings
regular_image, regular_results = detector.predict_image("regular_image.jpg")

# Process challenging image with custom thresholds
challenging_image, challenging_results = detector.predict_image(
    image_path="challenging_image.jpg",
    conf_threshold=0.2,  # Lower confidence threshold
    iou_threshold=0.5    # Custom IoU threshold
)
```

### Working with Raw Image Data

If you have image data as a numpy array instead of a file:

```python
import cv2
import numpy as np
from cow_detection.inference import YOLOInference

# Create detector
detector = YOLOInference(weights_path="path/to/weights.pt")

# Load image with OpenCV
image_path = "path/to/image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Process using detector's model directly
results = detector.model(image, verbose=False)[0]
results = detector.post_process(results)

# Get detection summary
summary = detector.get_detection_summary(results)
print(f"Detected {summary['detection_count']} cows")
```

## Error Handling

When using the inference API, handle potential errors:

```python
from cow_detection.inference import YOLOInference

detector = YOLOInference(weights_path="path/to/weights.pt")

try:
    image, results = detector.predict_image("path/to/image.jpg")
    print(f"Detected {len(results.boxes) if results.boxes is not None else 0} cows")
except ValueError as e:
    print(f"Error processing image: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

- The inference API automatically uses GPU acceleration when available
- For batch processing of many images, use `render_boxes=False` for better performance
- The first inference call may take longer due to model loading and initialization
- Image size affects processing time and memory usage

## Command Line Interface

The inference API is also accessible through the command line interface:

```bash
# Visual output mode
run-cow-inference --image path/to/image.jpg --output results/detected.jpg

# CLI output mode
run-cow-inference --image path/to/image.jpg --no-render

# Batch processing
run-cow-inference --dir path/to/images/ --output results/
```

See the [Inference Guide](../guides/inference.md) for more details on the command line interface.
