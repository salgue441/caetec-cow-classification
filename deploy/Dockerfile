# Use Python slim as base image for ARM architecture
# FROM arm64v8/python:slim
FROM arm64v8/python:3.9-slim

# Argumentos para configuración
ARG CAPTURE_INTERVAL=10
ARG CAMERA_ID=0
ARG IMAGE_WIDTH=1920
ARG IMAGE_HEIGHT=1080

# Variables de entorno
ENV CAPTURE_INTERVAL=$CAPTURE_INTERVAL \
  CAMERA_ID=$CAMERA_ID \
  IMAGE_WIDTH=$IMAGE_WIDTH \
  IMAGE_HEIGHT=$IMAGE_HEIGHT \
  PYTHONUNBUFFERED=1

# Instalar dependencias del sistema y herramientas de compilación
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
  && rm -rf /var/lib/apt/lists/*


# Crear y establecer el directorio de trabajo
WORKDIR /app

# Copiar los requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN python -m pip install --upgrade pip && \
  pip install --no-cache-dir setuptools wheel && \
  pip install --no-cache-dir -r requirements.txt

# Copiar el código y modelos
COPY models/ /app/models/
COPY script.py .

# Crear directorio para logs
RUN mkdir -p /app/logs

# Metadatos de la imagen
LABEL maintainer="Luis Arturo <inarrituzanetti@outlook.com>"
LABEL version="1.0"
LABEL description="YOLO Cow Detection for Raspberry Pi"

# Volumen para los logs
VOLUME ["/app/logs"]

# Comando para ejecutar la aplicación
CMD ["python", "script.py"]