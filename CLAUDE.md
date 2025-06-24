# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

exo is a distributed AI inference framework that enables running large language models across multiple heterogeneous devices in a peer-to-peer network. It creates a unified GPU from everyday devices like iPhones, iPads, Macs, NVIDIA GPUs, and Raspberry Pis.

## Common Development Commands

### Running exo

```bash
# Basic start (auto-discovers peers)
exo

# Run a specific model directly
exo run llama-3.2-3b --prompt "What is the meaning of exo?"

# Start with specific inference engine
exo --inference-engine mlx  # For Apple Silicon
exo --inference-engine tinygrad  # Cross-platform

# Manual peer discovery
exo --discovery-module manual --discovery-config-path config.json

# Wait for peers before starting
exo --wait-for-peers 2
```

### Testing

Tests are individual Python files that can be run directly:
```bash
python test/test_tokenizers.py
python exo/topology/test_ring_memory_weighted_partitioning_strategy.py
```

### Code Formatting

```bash
# Install formatting dependencies
pip install -e '.[formatting]'

# Format code with yapf
python format.py ./exo
```

### Installation

```bash
# From source with venv
source install.sh

# Or direct pip install
pip install -e .
```

## Architecture

### Core Design Principles

1. **Peer-to-Peer Architecture**: No master-worker hierarchy. Any device can join/leave without affecting others.
2. **Ring-based Model Partitioning**: Models are split into shards (layer ranges) distributed in a ring topology weighted by device memory.
3. **Memory-weighted Distribution**: Devices contribute proportionally to their available memory.
4. **Streaming Inference**: Token-by-token streaming for real-time responses.
5. **Protocol Buffers**: Used for efficient gRPC inter-node communication.

### Key Components

- **`exo/inference/`**: Inference engines (MLX for Apple Silicon, tinygrad for cross-platform)
- **`exo/networking/`**: P2P networking modules (UDP discovery, gRPC communication, Tailscale)
- **`exo/orchestration/`**: Node coordination and request routing
- **`exo/topology/`**: Partitioning strategies for model distribution
- **`exo/api/`**: ChatGPT-compatible REST API implementation
- **`exo/download/`**: Model downloading and shard management

### Request Flow

1. Client sends request to any node's ChatGPT API endpoint (port 52415)
2. Node determines optimal shard distribution based on topology
3. Request is processed through the ring of nodes
4. Results stream back through the initiating node

### Environment Variables

- `EXO_HOME`: Model storage directory (default: `~/.cache/exo/downloads`)
- `HF_ENDPOINT`: HuggingFace mirror for model downloads
- `DEBUG`: Debug logging level (0-9)
- `TINYGRAD_DEBUG`: Tinygrad-specific debug level (1-6)

### Key Entry Points

- Main CLI: `exo/main.py:run()` - Entry point for the `exo` command
- API Server: `exo/api/chatgpt_api.py:ChatGPTAPI` - ChatGPT-compatible API
- Node: `exo/orchestration/node.py:Node` - Core orchestration logic

### Debugging Tips

1. Use `DEBUG=9 exo` for verbose logging
2. Check node discovery with `--discovery-timeout` parameter
3. Monitor topology visualization at `http://localhost:52415`
4. For networking issues, try manual discovery with a config file

### Platform-Specific Notes

- **macOS**: Run `./configure_mlx.sh` for optimal MLX performance
- **Linux with NVIDIA**: Ensure CUDA toolkit and cuDNN are installed
- **All platforms**: Requires Python 3.12+ for asyncio compatibility