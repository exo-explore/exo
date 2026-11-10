FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS rust-builder

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

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    UV_INSTALL_DIR=/usr/local/bin

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
        python3.13-dev \
        python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH"

WORKDIR /app

COPY --from=rust-builder /wheels/*.whl /tmp/wheels/
COPY --from=dashboard-builder /build/dashboard/build ./dashboard/build/

COPY pyproject.toml uv.lock README.md ./
# uv validates workspace members even when Docker installs prebuilt wheels instead.
COPY bench/pyproject.toml ./bench/pyproject.toml
COPY rust/exo_pyo3_bindings/pyproject.toml ./rust/exo_pyo3_bindings/pyproject.toml
COPY resources/ ./resources/
COPY src/ ./src/

RUN uv venv --python python3.13 \
    && uv sync --extra cuda13 --no-install-project --no-install-workspace \
    && uv pip install /tmp/wheels/*.whl \
    && uv pip install . --no-deps \
    && rm -rf /tmp/wheels

# MLX-LM expects this stream helper, while current Linux CUDA MLX wheels expose
# the equivalent API as mx.new_stream(). Keep the compatibility shim local to
# the Docker image until upstream Linux CUDA wheels catch up.
RUN /app/.venv/bin/python - <<'PY'
from pathlib import Path
import site
site_packages = Path(site.getsitepackages()[0])
(site_packages / "mlx_cuda_compat.py").write_text(
    "import mlx.core as mx\n"
    "if not hasattr(mx, 'new_thread_local_stream') and hasattr(mx, 'new_stream'):\n"
    "    mx.new_thread_local_stream = mx.new_stream\n"
)
(site_packages / "mlx_cuda_compat.pth").write_text("import mlx_cuda_compat\n")
PY

# Pre-download tiktoken vocab file for openai_harmony.
# This prevents runtime download failures in restricted network environments.
# See: https://github.com/exo-explore/exo/issues/1038
RUN mkdir -p /app/tiktoken_cache \
    && curl -sSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken \
       -o /app/tiktoken_cache/o200k_base.tiktoken

ENV TIKTOKEN_ENCODINGS_BASE=/app/tiktoken_cache

COPY docker/entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 52415

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/.venv/bin/exo"]
