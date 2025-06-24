# Deepseek Model Support in Exo

This document describes how to run Deepseek models using the tinygrad inference engine on non-Mac systems.

## Overview

Exo now supports Deepseek V3 and Deepseek R1 models on non-Apple Silicon devices through the tinygrad inference engine. This enables cross-platform inference for these advanced models that were previously only available on Mac systems with MLX.

## Supported Models

The following Deepseek models are now available for tinygrad:

- **deepseek-v3** - Deepseek V3 base model (671B parameters)
- **deepseek-r1** - Deepseek R1 reasoning model (671B parameters)

## Platform Support

These models can run on:
- **Linux with NVIDIA GPUs** (CUDA backend)
- **Linux/Windows with CPUs** (CLANG backend)
- **ARM devices** (CPU inference)

## Usage

### Basic Usage

Run a Deepseek model with automatic engine selection:

```bash
# On non-Mac systems, tinygrad will be automatically selected
exo run deepseek-v3 --prompt "What is the meaning of life?"
```

### Explicit Engine Selection

Force tinygrad engine (useful for testing):

```bash
# Explicitly use tinygrad
exo run deepseek-v3 --inference-engine tinygrad --prompt "Explain quantum computing"

# Run Deepseek R1 for reasoning tasks
exo run deepseek-r1 --inference-engine tinygrad --prompt "Solve this step by step: What is 15% of 240?"
```

### Interactive Mode

Start an interactive session:

```bash
# Start interactive session with Deepseek V3
exo run deepseek-v3 --inference-engine tinygrad

# Start with Deepseek R1 for reasoning
exo run deepseek-r1 --inference-engine tinygrad
```

### Distributed Inference

Run across multiple devices:

```bash
# Device 1 (primary)
exo run deepseek-v3 --inference-engine tinygrad

# Device 2 (joins automatically)
exo --inference-engine tinygrad
```

## Model Architecture Features

### Deepseek V3
- **Mixture of Experts (MoE)**: 64 routed experts + 2 shared experts
- **Multi-Query Attention**: Efficient attention mechanism with LoRA compression
- **Advanced RoPE**: Custom rotary position embeddings
- **Large Context**: Supports up to 163K tokens

### Deepseek R1
- **Reasoning Optimized**: Specialized for step-by-step reasoning tasks
- **Same Architecture**: Based on Deepseek V3 with reasoning-specific training
- **Chain-of-Thought**: Excellent for complex problem solving

## Performance Notes

### Memory Requirements
- **Deepseek V3/R1**: ~300GB+ VRAM for full precision
- **Recommended**: Use distributed inference across multiple GPUs
- **Minimum**: 24GB VRAM per device with proper sharding

### Hardware Recommendations
- **Best**: Multiple NVIDIA A100/H100 GPUs
- **Good**: RTX 4090/3090 (24GB VRAM) in distributed setup
- **Minimum**: RTX 3080 (10GB) with aggressive sharding

### Environment Variables

Optimize performance with these settings:

```bash
# Enable CUDA optimizations
export CUDA=1

# Set tinygrad debug level (0-6)
export TINYGRAD_DEBUG=1

# For CPU inference (slower but works without GPU)
export CLANG=1
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Use more aggressive sharding
   exo run deepseek-v3 --inference-engine tinygrad --wait-for-peers 2
   ```

2. **Model Download Issues**
   ```bash
   # Set Hugging Face cache directory
   export HF_HOME=/path/to/large/storage
   ```

3. **CUDA Not Found**
   ```bash
   # Install CUDA toolkit and ensure it's in PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

### Debug Mode

Enable verbose logging:

```bash
DEBUG=9 exo run deepseek-v3 --inference-engine tinygrad
```

## Implementation Details

### Model Detection
The system automatically detects Deepseek models by checking the `model_type` field in `config.json`:

```json
{
  "model_type": "deepseek_v3",
  ...
}
```

### Weight Conversion
Hugging Face weights are automatically converted to tinygrad format, including:
- MoE expert weight reorganization
- Attention projection mappings
- LoRA parameter handling

### Sharding Strategy
Models are sharded across devices using:
- Layer-based distribution
- Memory-weighted allocation
- Ring topology for inference

## Comparison with MLX

| Feature | MLX (Mac Only) | Tinygrad (Cross-Platform) |
|---------|----------------|---------------------------|
| Platform | Apple Silicon | Linux, Windows, Mac |
| Performance | Optimized for Metal | CUDA/OpenCL optimized |
| Memory Usage | Efficient | Configurable |
| Model Support | All Deepseek variants | V3 and R1 |

## Contributing

To add support for additional Deepseek models:

1. Add model configuration in `exo/inference/tinygrad/inference.py`
2. Update model registry in `exo/models.py`
3. Test with your hardware configuration
4. Submit a pull request

## References

- [Deepseek V3 Paper](https://arxiv.org/abs/2412.19437)
- [Tinygrad Documentation](https://github.com/tinygrad/tinygrad)
- [Exo Documentation](https://github.com/exo-explore/exo)