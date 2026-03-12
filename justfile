export NIX_CONFIG := "extra-experimental-features = nix-command flakes"

fmt:
    treefmt || nix fmt

lint:
    uv run ruff check --fix

test:
    uv run pytest src

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v nvidia-smi &>/dev/null; then
        arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        export TORCH_CUDA_ARCH_LIST="$arch"
        export VLLM_TARGET_DEVICE=cuda
    fi
    uv pip install "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" wheel jinja2
    find ~/.cache/uv/git-v0 -name CMakeCache.txt -delete 2>/dev/null || true
    uv sync --extra cuda

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

rust-rebuild:
    cargo run --bin stub_gen
    uv sync --reinstall-package exo_pyo3_bindings

build-dashboard:
    #!/usr/bin/env bash
    cd dashboard
    npm install
    npm run build

package:
    uv run pyinstaller packaging/pyinstaller/exo.spec

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
    rm -rf dashboard/node_modules
    rm -rf dashboard/.svelte-kit
    rm -rf dashboard/build
