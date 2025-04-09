# 🐄 CowVision

Advanced computer vision system for automated cow detection using state-of-the-art YOLO models.

## ✨ Features

- **Multiple Inference Modes**: visual output or CLI-only for automation
- **Batch Processing**: process entire directories with a single command
- **Flexible Configuration**: customize confidence thresholds, IoU, and more
- **Easy Integration**: simple Python API for integration into larger systems

## 🚀 Installation

### Prerequisites

- Python 3.10
- Conda (recommended) or venv

### Setup

#### Docker

#### Manual Install

```bash
# Clone repository
git clone https://github.com/salgue441/caetec-cow-classification
cd caetec-cow-classification

# Create and activate environment
conda create -n cowvision python=3.10
conda activate cowvision

# Install dependencies
pip install -e .
```

## 📊 Usage

### Training

```bash
# Train with default parameters
train-cow-model --data_yaml data/data.yaml

# Train with custom parameters
train-cow-model --data_yaml data/data.yaml --epochs 100 --batch_size 16 --img_size 640
```

### Inference

#### Single Image

```bash
# Visual output (with bounding boxes)
run-cow-inference --image path/to/image.jpg --output results/detected.jpg

# CLI output only
run-cow-inference --image path/to/image.jpg --no-render
```

#### Batch Processing

```bash
# Process directory with visualization
run-cow-inference --dir path/to/images/ --output results/

# Process directory with CLI output only
run-cow-inference --dir path/to/images/ --no-render
```

#### Advanced Options

```bash
# Custom model and high confidence threshold
run-cow-inference --dir path/to/images/ \
                  --model yolov9c.pt \
                  --weights custom_weights.pt \
                  --conf 0.6 \
                  --min-area 350
```

## 🏗️ Architecture

```plaintext
cow-detection/
├── src/
│   └── cow_detection/       # Core package
│       ├── configs/         # Configuration classes
│       ├── data/            # Data handling utilities
│       ├── detection/       # Detection algorithms
│       ├── inference.py     # Inference module
│       ├── tracker.py       # Object tracking
│       └── utils/           # Helper functions
├── scripts/                 # Training and utility scripts
├── data/                    # Dataset storage
├── models/                  # Pre-trained models
├── tests/                   # Testing suite
└── docs/                    # Documentation
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

If you have any questions, feel free to reach out to [Vaqueros de Datos](mailto:vaquerosdedatos@gamil.com)
