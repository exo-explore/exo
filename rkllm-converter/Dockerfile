# RKLLM Model Converter (x86_64 ONLY)
# Converts HuggingFace models to .rkllm format for Rockchip NPU
#
# IMPORTANT: RKLLM-Toolkit only runs on x86_64 Linux.
#
# On x86_64 systems:
#   docker build -t rkllm-converter .
#   docker run -it --rm -v $(pwd)/output:/workspace/output rkllm-converter
#
# On ARM64 systems (DGX Spark, Apple Silicon):
#   # First, enable QEMU emulation:
#   docker run --privileged --rm tonistiigi/binfmt --install amd64
#
#   # Then build with buildx:
#   docker buildx create --use --name mybuilder
#   docker buildx build --platform linux/amd64 -t rkllm-converter --load .
#
#   # Run with platform flag:
#   docker run --platform linux/amd64 -it --rm \
#              -v $(pwd)/output:/workspace/output \
#              rkllm-converter
#
# Using ARG for platform allows buildx to override
ARG TARGETPLATFORM=linux/amd64
FROM python:3.10-slim

LABEL maintainer="exo-rkllama"
LABEL description="RKLLM Model Converter for Rockchip RK3588/RK3576 NPU"
LABEL version="1.2.3"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# RKLLM Toolkit version - wheel is inside the cloned repo
ARG RKLLM_VERSION=1.2.3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone RKNN-LLM repo and install toolkit from wheel
RUN git clone --depth 1 --branch release-v${RKLLM_VERSION} \
    https://github.com/airockchip/rknn-llm.git /tmp/rknn-llm && \
    pip install --no-cache-dir /tmp/rknn-llm/rkllm-toolkit/packages/rkllm_toolkit-*-cp310-cp310-linux_x86_64.whl && \
    rm -rf /tmp/rknn-llm

# Install additional dependencies for model loading
RUN pip install --no-cache-dir \
    torch \
    transformers>=4.40.0 \
    sentencepiece \
    accelerate \
    tiktoken \
    huggingface_hub \
    safetensors \
    tqdm \
    pyyaml

# Copy conversion scripts
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.py /workspace/scripts/*.sh 2>/dev/null || true

# Create directories for models and output
RUN mkdir -p /workspace/models /workspace/output /workspace/cache

# Set HuggingFace cache directory
ENV HF_HOME=/workspace/cache
ENV TRANSFORMERS_CACHE=/workspace/cache

# Default command
CMD ["python", "/workspace/scripts/convert.py", "--help"]
