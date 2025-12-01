# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

exo is a distributed inference framework that allows running AI models across multiple heterogeneous devices (Mac, Linux, NVIDIA GPUs, Raspberry Pi, etc.) connected in a P2P network. Unlike master-worker architectures, all exo nodes are equal peers that automatically discover each other and partition models based on available memory.

## Prerequisites

- Python >= 3.12 (required due to asyncio issues in earlier versions)
- For Linux with NVIDIA GPU: NVIDIA driver, CUDA toolkit, cuDNN library

## Common Commands

### Installation
```sh
pip install -e .
# or with venv
source install.sh
```

### Running exo
```sh
exo                                    # Start node with auto-discovery
exo run llama-3.2-3b                   # Run single model with CLI
exo run llama-3.2-3b --prompt "..."    # Run with custom prompt
exo --inference-engine mlx             # Explicitly choose inference engine (mlx/tinygrad/dummy)
exo --discovery-module udp             # Choose discovery module (udp/tailscale/manual)
exo eval llama-3.2-3b --data path/to/data  # Evaluate model
exo train llama-3.2-3b --data path/to/data --iters 100  # Train model
```

### Testing
```sh
# Run inference engine tests
TEMPERATURE=0 python3 -m exo.inference.test_inference_engine

# Run individual test files
python3 ./test/test_tokenizers.py
python3 ./test/test_model_helpers.py

# Tests use pytest with async support
pytest exo/orchestration/test_node.py
```

### Code Formatting
```sh
pip3 install -e '.[formatting]'
python3 format.py ./exo              # Format all code in exo/
```

### Debugging
```sh
DEBUG=9 exo                          # Enable debug logs (0-9)
TINYGRAD_DEBUG=2 exo                 # Enable tinygrad-specific debug logs (1-6)
```

### Environment Variables
```sh
EXO_HOME=~/.cache/exo/downloads      # Model storage location (default)
HF_ENDPOINT=https://hf-mirror.com    # HuggingFace mirror endpoint for restricted regions
```

### Performance (Apple Silicon)
```sh
./configure_mlx.sh                   # Optimize GPU memory allocation for Apple Silicon
```

## Architecture

### Core Components

**Node (`exo/orchestration/node.py`)**: Central coordinator that manages inference, peer communication, and topology. Handles prompt processing across distributed shards.

**Inference Engines (`exo/inference/`)**:
- `mlx/` - MLX backend for Apple Silicon
- `tinygrad/` - tinygrad backend for cross-platform support (Linux, NVIDIA)
- Selection is automatic based on platform (Apple Silicon → MLX, Linux → tinygrad)

**Model Partitioning (`exo/topology/`)**:
- `ring_memory_weighted_partitioning_strategy.py` - Default strategy that partitions model layers proportionally to device memory
- Shards (`exo/inference/shard.py`) represent a portion of a model (start_layer to end_layer)

**Peer Discovery (`exo/networking/`)**:
- `udp/` - Default UDP broadcast discovery
- `tailscale/` - Tailscale VPN-based discovery
- `manual/` - Manual peer configuration via JSON
- `grpc/` - gRPC-based peer communication

**API Layer (`exo/api/chatgpt_api.py`)**: ChatGPT-compatible REST API served on port 52415. Supports `/v1/chat/completions` endpoint.

**Model Registry (`exo/models.py`)**: Maps model names (e.g., "llama-3.2-3b") to HuggingFace repos for each inference engine. Models downloaded to `~/.cache/exo/downloads` by default.

### Data Flow

1. API receives chat completion request
2. Node builds Shard from model name and current topology partition
3. Node orchestrates inference across peer nodes via gRPC
4. Each node runs its shard layers and passes activations to next peer
5. Final tokens returned to API

### Key Patterns

- Async throughout using `asyncio` with `uvloop` for performance
- `AsyncCallbackSystem` for event handling (token generation, status updates)
- Shards are dynamically assigned based on topology changes
- Models support multiple inference engines via `model_cards` repo mappings

## Development History

### AMD iGPU Support (Ryzen AI Max+ 395)
- **Base commit**: 5f18faec1758ffa79d6db1d463b4b8bca211a0a8
- **Branch**: feature/amd-igpu-support
- **Objective**: Enable GPU inference on Ryzen AI Max+ 395 iGPU
- **Status**: ✅ Completed

#### Issues Resolved

**Issue: ModuleNotFoundError: No module named 'tinygrad.runtime.autogen.am'**
- **Root cause**: The tinygrad version specified in setup.py (commit `ec120ce6b9ce8e4ff4b5692566a683ef240e8bc8`) did not include the `am` autogen module required for AMD GPU support
- **Solution**: Updated tinygrad to commit `759b41ab913359520fc0a49976a842829e63be3d` (v0.11.0), which includes the complete autogen infrastructure including the `am` module
- **Files modified**: `setup.py`
- **Testing**: Confirmed working with test prompt - GPU inference operational on Radeon 8060S
