# syntax=docker/dockerfile:1
#
# Build a Linux container for the exo distributed-inference runtime
# (upstream project).
#
# exo needs Python 3.13, is synced with `uv` (uv_build backend + a Rust
# workspace that compiles the `exo_rs` networking bindings), and serves a
# Svelte dashboard it builds from Node. This image installs those pinned
# dependencies, compiles the Rust bindings, builds the dashboard, and boots
# the node with the documented `uv run exo` command (API/dashboard on 52415).
#
# The default image uses exo's base dependencies (boots the node, API and
# dashboard; no inference backend). To enable a CPU/CUDA backend at build
# time pass an extra from pyproject.toml, e.g.:
#
#   docker build --build-arg EXO_EXTRAS=mlx-cpu .
#
# (Those extras pull torch, making the image much larger.)

FROM python:3.13-slim-bookworm AS base

ARG EXO_EXTRAS=
# The host only has an IPv4 uplink; force Node/npm to resolve registries over
# IPv4 first instead of picking an unreachable IPv6 AAAA record.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    NODE_OPTIONS=--dns-result-order=ipv4first

# System build dependencies: C/c++ toolchain + cmake (native wheels & zenoh),
# python dev headers (pyo3), and Node/npm for the dashboard build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        build-essential \
        pkg-config \
        cmake \
        ninja-build \
        libssl-dev \
        python3-dev \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# uv (>= 0.8.6 required by the project). Copy the standalone binary from the
# official image so it doesn't bake Rust toolchain into the base.
COPY --from=ghcr.io/astral-sh/uv:0.8.6 /uv /uv/bin/uv
ENV PATH="/uv/bin:${PATH}"

# Rust toolchain to build the exo_rs bindings (PyO3). The project documents a
# nightly toolchain requirement for the Rust bindings.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_MAX_RETRIES=10 \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain nightly --profile minimal

WORKDIR /app
COPY . .

# Resolve the project lock: installs pinned python deps, builds the Rust
# workspace member(s) and installs the `exo` console script. No dev group.
RUN if [ -n "$EXO_EXTRAS" ]; then \
        uv sync --frozen --no-dev --extra "$EXO_EXTRAS"; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Build the frontend (served from the API root).
RUN cd dashboard \
    && npm ci \
    && npm run build \
    && cd ..

# Keep the image lean-ish: drop apt lists and uv's cached wheels.
RUN rm -rf /root/.cache/uv
ENV EXO_EXTRAS="$EXO_EXTRAS"

# Non-root runtime user. exo follows the XDG Base Directory for config,
# data, cache and logs, which we map onto a persistent volume.
RUN useradd --create-home --uid 1000 exo \
    && mkdir -p /data \
    && chown -R exo:exo /data /app

USER exo
ENV XDG_CONFIG_HOME=/data/config \
    XDG_DATA_HOME=/data/data \
    XDG_CACHE_HOME=/data/cache

VOLUME /data
EXPOSE 52415

CMD ["uv", "run", "exo"]