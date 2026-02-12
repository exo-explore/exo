# Stage 1: Build the dashboard
FROM node:22-slim AS dashboard
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ .
RUN npm run build

# Stage 2: Build and run exo
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    protobuf-compiler \
    iptables \
    && rm -rf /var/lib/apt/lists/*

# Install Rust nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml Cargo.toml uv.lock README.md ./
COPY rust/ ./rust/
COPY bench/pyproject.toml ./bench/pyproject.toml

# Copy source and resources
COPY src/ ./src/
COPY resources/ ./resources/

# Copy built dashboard from stage 1
COPY --from=dashboard /app/dashboard/build ./dashboard/build/

# Install Python deps and build Rust bindings, then clean up build artifacts
# to keep the layer small (Rust target/ and cargo registry can be 1-2 GB)
RUN uv sync && rm -rf /app/rust/target /root/.cargo/registry /root/.cargo/git

# Wrap g++ with -fpermissive to fix MLX CPU JIT compilation with GCC 14
# (GCC 14 treats _Float128/_Float32/_Float64 as built-in types, conflicting with MLX-generated code)
RUN mv /usr/bin/g++ /usr/bin/g++.real && \
    printf '#!/bin/sh\nexec /usr/bin/g++.real -fpermissive "$@"\n' > /usr/bin/g++ && \
    chmod +x /usr/bin/g++

CMD [".venv/bin/exo", "-v"]
