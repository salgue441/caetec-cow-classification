# 🐄 Cow Detection Project

## Overview

Advanced machine learning project for automated cow detection using state-of-the-art YOLO models.

## 🛠 Project Structure

- `src/cow_detection`: main package code
- `scripts/`: scripts for training and inference
- `data/`: dataset storage
- `docs/`: documentation and resources

## 🚀 Quick Start

1. Setup Environment

```bash
git clone https://github.com/salgue441/caetec-cow-classification
cd caetec-cow-classification

conda create -n cow-detection python=3.10
conda activate cow-detection

# Install dependencies
pip install -e .
```

2. Prepare data

Organize your dataset in the following structure:

```bash
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

3. Training

```bash
train-cow-model --data_yaml data/data.yaml --epochs 100 --batch_size 16
```

4. Inference for single images

```bash
run-cow-inference --image path/to/your/image.jpg --output detected_cows.jpg
```

## 🔍 Contact

If you have any questions, feel free to reach out to me at [Vaqueros de Datos](mailto:vaquerosdedatos@gamil.com)
