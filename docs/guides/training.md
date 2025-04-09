# Training Guide

This guide explains how to train or fine-tune cow detection models using your own dataset.

## Data Preparation

### Dataset Structure

Organize your dataset into the YOLO format:

```plaintext
data/
├── train/
│   ├── images/
│   │   ├── cow1.jpg
│   │   ├── cow2.jpg
│   │   └── ...
│   └── labels/
│       ├── cow1.txt
│       ├── cow2.txt
│       └── ...
├── valid/
│   ├── images/
│   │   ├── valid_cow1.jpg
│   │   └── ...
│   └── labels/
│       ├── valid_cow1.txt
│       └── ...
└── data.yaml
```

### Label Format

YOLO format labels are text files with one line per object

```plaintext
<class> <x_center> <y_center> <width> <height>
```

Where:

- `<class>` is the object class index (0 for cow)
- `<width>`, and `<height>` are normalized values (0 to 1) relative to the image size.
- `<x_center>` and `<y_center>` are the center coordinates of the bounding box

Example label file content:

```plaintext
0 0.342 0.579 0.416 0.362
0 0.742 0.651 0.198 0.279
```

## Configuration File

Create a `data.yml` file with dataset information:

```yaml
# data.yaml
path: ./data # Dataset root directory
train: train/images # Train images relative to 'path'
val: valid/images # Validation images relative to 'path'

nc: 1 # Number of classes
names: ["cow"] # Class names
```

## Training Process

### Basic Training

For basic training with default parameters

```bash
train-cow-model --data_yml data/data.yml
```

This will:

1. Load the default YOLO9 model
2. Train for 50 epochs with batch size of 8
3. Save results to `runs/detect/train[N]`

### Command Line Parameters

The training script support these command-line arguments:

| Parameter      | Description                | Default  | Example           |
| -------------- | -------------------------- | -------- | ----------------- |
| `--data_yml`   | Path to data configuration | Required | `data/data.yml`   |
| `--epochs`     | Number of training epochs  | 50       | `--epochs 100`    |
| `--batch_size` | Batch size for training    | 16       | `--batch_size 32` |

## Example Usage

Training with custom epochs and batch size

```bash
python train_model.py --data_yaml data/data.yaml --epochs 100 --batch_size 8
```

## Configuration Settings

The training process uses two configuration classes:

### Detection Configuration

The `DetectionConfig` class controls detection parameters:

```python
detection_config = DetectionConfig(
    conf_threshold=0.4,    # Confidence threshold for detections
    iou_threshold=0.70,    # IoU threshold for non-maximum suppression
    min_area=300,          # Minimum area for valid detections
    max_overlap_ratio=0.65,# Maximum overlap ratio between detections
    enable_temporal=False  # Temporal consistency (for video)
)
```

### Training Configuration

The `TrainingConfig` class controls training parameters:

```python
training_config = TrainingConfig(
    epochs=50,             # Number of training epochs
    image_size=640,        # Input image size
    batch_size=16,         # Batch size
    dropout=0.3            # Dropout rate for regularization
)
```

## Monitor Training

The training process provides console output with training progress, including:

- Loss values (box_loss, cls_loss, dfl_loss)
- Metrics on validation data (precision, recall, mAP)
- GPU memory usage and processing speed

Example output:

````plaintext
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  0/49      5.54G     0.9011    0.02678      1.103         16        640: 100%|██| 131/131 [00:43<00:00,  4.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██| 17/17 [00:07<00:00,  1.89it/s]
                   all        267        284      0.895      0.737      0.823      0.573
                   ```
````

## Training Results

After training completes, the model and results are available in the return values

```python
model, results = trainer.train(args.data_yaml, epochs=args.epochs, batch_size=args.batch_size)
```

The trained model weights are typically saved in the `runs/detect/train[N]/weights/` directory, where `[N]` is the training run number.

## Tips for Better Performance

- **Data Quality**: ensure diverse images with different lightning, angles, and backgrounds.
- **Validation Split**: maintain a ~20% validation set that represents your target use case.
- **Batch Size**: use the largest batch size your GPU can handle (reduce if you get OOM errors).
- **Image Size**: larger image sizes often improve detection of small objects but require more memory.
- **Epochs**: train long enough to let the model converge (watch validation metrics).
- **Augmentation**: data augmentation is applied automatically, but can be customized in extended configurations.
- **Learing Rate**: the default learning rate schedule works well for most cases.

## Troubleshooting

### Out Of Memory (OOM) Errors

If you encounter OOM errors during training, try:

1. Reduce batch size (e.g., `--batch_size 8` or even `--batch_size 4`)
2. Use a smaller base model if available

### Training Time

Training time depends on:

- GPU capability
- Dataset size
- Image size
- Batch size
- Number of epochs

A typical training run on a modern GPU might take several hours to complete.

## Next Steps

After training a model, you can:

1. Test it using the [Inference Guide](./inference.md)
2. Deploy it using the [Deployment Guide](./deployment/README.md)
3. Fine-tune it further with more data or different parameters
