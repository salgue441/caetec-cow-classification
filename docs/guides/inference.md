# Inference Guide

This guide explains how to use the cow detection model for inference on images and directories.

## Inference Models

The cow detection system offers two primary modes of operation:

1. **Visual Output Mode**: produces annotated images with bounding boxes around detected cows.
2. **CLI Output Mode**: provides detection data as text output without visual rendering.

## Command Line Interface (CLI)

### Basic Usage

The main entry point for running inferences is the `run-cow-inference` command:

```bash
# Basic usage with visual output
run-cow-inference --image path/to/image.jpg --output results/detected.jpg
```

### Single Image Processing

#### Visual Output (Default)

Process a single image and generate an annotated output with bounding boxes:

```bash
run-cow-inference --image path/to/image.jpg --output results/detected.jpg
```

To display the result using matplotlib after processing:

```bash
run-cow-inference --image path/to/image.jpg --output results/detected.jpg --show
```

#### CLI Output

Process a single image and get detection information without visual output:

```bash
run-cow-inference --image path/to/image.jpg --no-render
```

Example output:

```plaintext
Image: cow.jpg
  Detections: 5
  Detection 1: probability 0.87
  Detection 2: probability 0.76
  Detection 3: probability 0.75
  Detection 4: probability 0.71
  Detection 5: probability 0.47
```

## Batch Processing

### Process a Directory of Images

Process all images in a directory with visualization:

```bash
run-cow-inference --dir path/to/images/ --output results/
```

This will:

1. Process all images in the specified directory.
2. Save annotated images in the `results/` directory.
3. Display the results using matplotlib.

Process all images with CLI output only:

```bash
run-cow-inference --dir path/to/images/ --no-render
```

Example output for directory processing:

```plaintext
Image: cow1.jpg
  Detections: 3
  Detection 1: probability 0.92
  Detection 2: probability 0.88
  Detection 3: probability 0.65

Image: cow2.jpg
  Detections: 2
  Detection 1: probability 0.79
  Detection 2: probability 0.72

[...]

Summary:
  Processed 24 images
  Total detections: 67
  Average detections per image: 2.79
```

## Configuration Parameters

### Model Selection

Select which model and weights to use

```bash
run-cow-inference --image path/to/image.jpg \
                  --model yolov9c.pt \
                  --weights custom_weights.pt
```

### Detection Threshold

Adjust detection sensitivity and filtering:

```bash
run-cow-inference --image path/to/image.jpg \
                  --conf 0.6 \
                  --iou 0.7 \
                  --min-area 350
```

## Complete Parameter List

| Parameter     | Description                          | Default                                |
| ------------- | ------------------------------------ | -------------------------------------- |
| `--image`     | Path to the input image              | Required                               |
| `--dir`       | Path to directory of images          | None                                   |
| `--output`    | Path to the output `image/directory` | output.jpg                             |
| `--weights`   | Path to model weights                | `runs/detect/train[N]/weights/best.pt` |
| `--model`     | Base model architecture              | `yolov9c.pt`                           |
| `--conf`      | Confidence threshold for detections  | 0.4                                    |
| `--iou`       | IoU threshold for NMS                | 0.7                                    |
| `--min-area`  | Minimum detection area               | 300                                    |
| `--no-render` | Don't render bounding boxes          | False                                  |
| `--show`      | Show the output image                | False                                  |

## Example Use Cases

### High-Confidence Detection

For applications where false positives must be minimized:

```bash
run-cow-inference --dir path/to/images/ --conf 0.7 --min-area 400
```

### Low-Latency Processing

For faster processing when speed is critical:

```bash
run-cow-inference --dir path/to/images/ --no-render
```

### Integration with Other Tools

For integration with other systems, CLI output can be parsed:

```bash
run-cow-inference --image input.jpg --no-render > detections.txt
```

## Programmatic Usage

You can also use the inference system programmatically in your Python code:

```python
from cow_detection.inference import YOLOInference
from cow_detection.configs import DetectionConfig
import matplotlib.pyplot as plt

# Configure detector
detection_config = DetectionConfig(
    conf_threshold=0.4,
    iou_threshold=0.7,
    min_area=300
)

# Initialize detector
detector = YOLOInference(
    model_path="yolov9c.pt",
    weights_path="path/to/weights.pt",
    detection_config=detection_config
)

# Method 1: Visual output mode
image, results = detector.predict_image(
    image_path="path/to/image.jpg",
    save_path="output.jpg",
    render_boxes=True
)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.title(f"Detected {len(results.boxes)} cows")
plt.show()

# Method 2: CLI output mode
detection_summary = detector.predict_image(
    image_path="path/to/image.jpg",
    render_boxes=False
)

print(f"Detected {detection_summary['detection_count']} cows")
for i, prob in enumerate(detection_summary['probabilities'], 1):
    print(f"Cow {i}: {prob:.2f}")
```

### Processing Multiple Files Programmatically

```python
import os
from pathlib import Path
from cow_detection.inference import YOLOInference

detector = YOLOInference(weights_path="path/to/weights.pt")

# Process all images in a directory
input_dir = "data/images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for image_path in Path(input_dir).glob("*.jpg"):
    output_path = os.path.join(output_dir, f"{image_path.stem}_detected.jpg")

    # Visual mode
    detector.predict_image(
        image_path=str(image_path),
        save_path=output_path,
        render_boxes=True
    )

    # Or CLI mode
    results = detector.predict_image(
        image_path=str(image_path),
        render_boxes=False
    )

    print(f"{image_path.name}: {results['detection_count']} detections")
```

## Performance Considerations

- **CLI Output Mode**: for processing large datasets, use the CLI output mode (`--no-render`) to avoid rendering overhead.
- **Batch Processing**: when processing directories, results are processed sequentially with a progress bar.
- **GPU Acceleration**: GPU acceleration is used automatically when available.
- **Image Size**: larger images require more processing time but may provide better detection accuracy.

## Troubleshooting

### Common Issues

#### No Detection Found

If the model doesn't detect any cows:

1. Try lowering the confidence threshold (`--conf 0.2`).
2. Check if the image contains cows with typical appearance.
3. Verify that the model weights are loaded correctly.

#### Poor Detection Quality

If detections are inaccurate:

1. Try a different model or weights.
2. Adjust the confidence and IoU thresholds.
3. Consider retraining the model on data more similar to your target images.

#### Error: "CUDA out of memory"

If you encounter GPU memory errors:

1. Process images one at a time instead of in a directory.
2. Use a smaller model variant.
3. Process images at a lower resolution.

## Next Steps

- Read the [API Documentation](api/inference.md) for more details on the inference module.
- Check the [Training Guide](training.md) for training custom models.
- Explore the [System Architecture]() for a deeper understanding of the detection.
