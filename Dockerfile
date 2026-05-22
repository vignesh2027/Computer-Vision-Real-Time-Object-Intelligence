FROM python:3.11-slim

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-download YOLOv8n weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || true

EXPOSE 8000 8501

# Default: start FastAPI. Override CMD for Streamlit.
CMD ["uvicorn", "cv_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- GPU variant ---
# To use CUDA, build with:
#   docker build --build-arg BASE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 .
# and replace the FROM line above with the CUDA base image.
# torch/torchvision will auto-detect the GPU at runtime.
