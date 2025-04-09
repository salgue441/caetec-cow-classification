# Installation Guide

This guide walks you through the process of setting up the Cow Detection system on your machine.

## Prerequisites

- Python 3.10 or higher
- CUDA 11.7+ (for GPU acceleration, optional but not recommended)
- 8GB+ of RAM
- Git

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/salgue441/caetec-cow-classification
cd caetec-cow-classification

# Create conda environment
conda create -n cowvision python=3.10
conda activate cowvision

# Install dependencies
pip install -e .
```

### Option 2: Using Virtualenv (venv)

```bash
# Clone repository
git clone https://github.com/salgue441/caetec-cow-classification
cd caetec-cow-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## GPU Acceleration Setup (Optional)

For optimal performance, we recommend using GPU acceleration:

1. Ensure you have a compatible NVIDIA GPU.
2. Install CUDA Toolkit 11.7 or higher from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
3. Install cuDNN from the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn).
4. Install PyTorch with CUDA support:

```bash
# For CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Installing Pre-trained Models

The system comes with scripts to download pre-trained weights:

```bash
# Download YOLOv9 base model
python scripts/download_models.py --model yolov9c

# Download our pre-trained cow detection weights
python scripts/download_models.py --weights cow_detection
```

## Verification

To verify your installation, run a simple inference test:

```bash
# Test on a sample image
run-cow-inference --image data/sample_images/test.jpg --no-render
```

You should see an ouptut similar to:

```plaintext
Image: test.jpg
  Detections: 3
  Detection 1: probability 0.92
  Detection 2: probability 0.87
  Detection 3: probability 0.76
```

To verify the visual functionality:

```bash
run-cow-inference --image data/sample_images/test.jpg --output test_output.jpg --show
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'cow_detection'

Ensure you've installed the package in development mode with:

```bash
pip install -e
```

#### CUDA not available

If you're getting CPU-only inference when you have a GPU:

1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"
3. Reinstall PyTorch with the correct CUDA version

#### Permission Denied on Scripts

If you encounter permission issues running the script:

```bash
chmod +x scripts/*.py
```

## Next Steps

Now that you have installed the Cow Detection System, you can:

- Read the [Training Guide](./training.md) to train custom models
- Check the [Inference Guide](./inference.md) for running inference on images
- Explore the [API Documentation](../api/inference.md) for advanced usage

## System Requirements

For optimal performance, we recommend:

| Component | Minimum                  | Recommended              |
| --------- | ------------------------ | ------------------------ |
| CPU       | 4 cores                  | 8+ cores                 |
| RAM       | 8 GB                     | 16 GB+                   |
| GPU       | -                        | NVIDIA with 6GB+ VRAM    |
| Storage   | 5 GB                     | 20 GB+ SSD               |
| OS        | Ubuntu 20.04, Windows 10 | Ubuntu 22.04, Windows 11 |
