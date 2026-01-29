# Dockerfile
# Base: CUDA 12 + cuDNN on Ubuntu 22.04 for NVIDIA GPUs
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Recommended by RunPod: build linux/amd64 images for Serverless
# (use: docker build --platform=linux/amd64 ...)
# https://deepwiki.com/runpod/docs/3.3-templates-and-docker-images
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Copy files
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py ./handler.py
COPY test_input.json ./test_input.json

# Environment:
# - Set at deploy time in RunPod console (Secrets): HF_TOKEN=<your_HF_access_token>
# - Optional: FLUX_MODEL_ID to override model, e.g., black-forest-labs/FLUX.1-schnell
ENV FLUX_MODEL_ID="black-forest-labs/FLUX.1-dev"

# Start the RunPod worker
CMD ["python", "-u", "handler.py"]
