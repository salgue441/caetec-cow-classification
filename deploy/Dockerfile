# Use Python slim as base image for ARM architecture
FROM arm64v8/python:3.9-slim

# Configuration arguments
ARG CAMERA_ID=0
ARG IMAGE_WIDTH=1920
ARG IMAGE_HEIGHT=1080

# Environment variables
ENV CAMERA_ID=$CAMERA_ID \
    IMAGE_WIDTH=$IMAGE_WIDTH \
    IMAGE_HEIGHT=$IMAGE_HEIGHT \
    PYTHONUNBUFFERED=1

# Enhanced system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    v4l-utils \
    git \
    wget \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies in specific order
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    # Install PyTorch and torchvision first
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu && \
    # Install other requirements
    pip install --no-cache-dir -r requirements.txt && \
    # Install ultralytics with version that supports YOLO9c
    pip install --no-cache-dir ultralytics>=8.1.0

# Copy files
COPY models/ /app/models/
COPY script.py .

# Create logs directory
RUN mkdir -p /app/logs

# Metadata
LABEL maintainer="Luis Arturo <inarrituzanetti@outlook.com>"
LABEL version="1.0"
LABEL description="YOLO Cow Detection for Raspberry Pi"

# Volume for logs
VOLUME ["/app/logs"]

# Add a healthcheck to verify Python and critical libraries
HEALTHCHECK CMD python -c "import torch; import ultralytics; print('Health check passed')" || exit 1

# Run script
CMD ["python", "script.py"]