# GPU-Accelerated Computer Vision Engine
# Requires the NVIDIA Container Toolkit on the host to actually see a GPU
# (docker run --gpus all ...) — see NFR-04 (Portability).

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Build is deliberately NOT run at image-build time in Phase 1 —
# CPU/CUDA sources don't exist yet. From Phase 2 onward:
#   RUN cmake -S . -B build && cmake --build build -j
