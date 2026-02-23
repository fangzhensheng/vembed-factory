# ==========================================================================
# vembed-factory Docker image
# ==========================================================================
# Build:  docker build -t vembed-factory .
# Run:    docker run --gpus all -it vembed-factory bash
# ==========================================================================

# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata first (Docker layer cache)
COPY pyproject.toml README.md LICENSE ./
COPY vembed/__init__.py vembed/__init__.py

# Install dependencies only (no project itself yet)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -e ".[lora,tracking]"

# Copy full source & install the project
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -e .

# Verify installation
RUN python -c "import vembed; print(f'vembed-factory v{vembed.__version__} installed successfully')"

# Default: drop into shell
CMD ["bash"]
