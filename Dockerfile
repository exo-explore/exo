FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS rust-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
    -y --default-toolchain nightly --profile minimal

RUN python3.13 -m ensurepip --upgrade \
    && python3.13 -m pip install --no-cache-dir maturin

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/

RUN maturin build \
    --release \
    --manylinux off \
    --manifest-path rust/exo_pyo3_bindings/Cargo.toml \
    --features "pyo3/extension-module,pyo3/experimental-async" \
    --interpreter python3.13 \
    --out /wheels

FROM node:22-slim AS dashboard-builder

WORKDIR /build/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci

COPY dashboard/ ./
RUN npm run build

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS runtime

# Set CUDA_HOME so MLX can find CUDA headers for JIT kernel compilation
ENV CUDA_HOME=/usr/local/cuda

# Set LD_LIBRARY_PATH so MLX can find CUDA libraries from nvidia pip packages
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.13/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libssl3 \
    software-properties-common \
    cmake \
    build-essential \
    libblas-dev \
    liblapack-dev \
    liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH"

WORKDIR /app

COPY --from=rust-builder /wheels/*.whl /tmp/wheels/
COPY --from=dashboard-builder /build/dashboard/build ./dashboard/build/

# Use Docker-specific pyproject.toml without workspace configuration
COPY pyproject.docker.toml pyproject.toml
COPY uv.lock README.md ./
COPY src/ ./src/

# Install dependencies using uv sync (no workspace validation needed)
# 1. Create venv
# 2. Install the pre-built native extension wheel
# 3. Install the project itself without deps (they're already installed)
RUN uv venv --python python3.13 \
    && uv sync \
    && uv pip install /tmp/wheels/*.whl \
    && uv pip install . --no-deps \
    # Add Python development headers for building MLX from source
    && apt-get update && apt-get install -y python3.13-dev && rm -rf /var/lib/apt/lists/* \
    && uv pip install "mlx[cuda13]==0.30.4" \
    && rm -rf /tmp/wheels

# Pre-download tiktoken vocab file for openai_harmony
# This prevents runtime download failures in restricted network environments
# See: https://github.com/exo-explore/exo/issues/1038
RUN mkdir -p /app/tiktoken_cache \
    && curl -sSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken \
       -o /app/tiktoken_cache/o200k_base.tiktoken

ENV TIKTOKEN_ENCODINGS_BASE=/app/tiktoken_cache

EXPOSE 52415

# Run directly from venv to avoid uv re-syncing from lockfile
# (lockfile has mlx[cpu] for Linux, but we need mlx[cuda13])
ENTRYPOINT ["/app/.venv/bin/exo"]
