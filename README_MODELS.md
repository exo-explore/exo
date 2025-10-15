# Small MoE (Mixture of Experts) Models for MLX on 16GB Mac Minis

## Overview
This document lists small MoE models available in MLX format that would fit on 16GB Mac minis, focusing on models under 8GB total that could benefit from pipeline parallelism.

## Currently Supported MoE Models in Exo

### DeepSeek V2 and V3 (Already Integrated)
- **Location**: `/Users/mini1/Movies/exo/exo/inference/mlx/models/deepseek_v2.py` and `deepseek_v3.py`
- **Architecture**: MoE with experts and routing mechanisms
- **Features**: Already includes expert handling and weight sanitization for MoE architectures
- **Pipeline Support**: Built-in sharding support with `n_routed_experts` parameter

## Available Small MoE Models for MLX

### 1. Qwen1.5-MoE-A2.7B
- **Total Parameters**: 14B (2.7B activated per token)
- **Non-embedding Parameters**: 2.0B (approximately 1/3 of Qwen1.5-7B)
- **Architecture**: 
  - 4 shared experts (always activated)
  - 60 routing experts 
  - Top-2 gating strategy
  - 8 experts per MLP with top-2 routing
- **Performance**: 75% decrease in training costs, 1.74x faster inference
- **Memory Requirements**: ~3-4GB for 4-bit quantized version (estimated)
- **MLX Support**: Planned/Limited (as of search date)

### 2. Qwen3-30B-A3B (2025)
- **Total Parameters**: 30.5B (3.3B activated per inference)
- **Architecture**:
  - 48 layers
  - 128 total experts (8 activated per inference)
  - 32,768 token context length
- **Performance**: Outperforms QwQ-32B with 10x fewer activated parameters
- **Memory Requirements**: 
  - 4-bit quantized: ~17GB (too large for 16GB Mac mini)
  - 6-bit quantized: ~25GB 
  - 8-bit quantized: ~32GB
- **MLX Support**: Yes (mlx-lm>=0.24.0)

### 3. Mixtral-8x7B Quantized Variants
- **Total Parameters**: 45B (13B activated during inference)
- **Architecture**: 8 experts per MLP, top-2 routing
- **Memory Requirements**:
  - 4-bit quantized: 26.4GB (too large for 16GB)
  - Mixed 4-bit attention + 2-bit experts: ~13GB (borderline for 16GB)
  - Mixed quantization approaches: 18.2GB
- **MLX Support**: Yes (mlx-community)
- **Context**: 32k tokens

## Models That Fit 16GB Mac Minis

### Recommended Small MoE Models (Under 8GB):

1. **Qwen1.5-MoE-A2.7B (4-bit quantized)**
   - Estimated size: ~3-4GB
   - Activated parameters: 2.7B
   - Excellent performance/size ratio
   - Status: Limited MLX support

### Alternative Dense Models for Comparison:

1. **Qwen3-0.6B** (Dense)
   - Parameters: 600M
   - Layers: 28
   - Context: 32k tokens
   - Size: <1GB quantized

2. **Qwen3-4B** (Dense) 
   - Parameters: 4B
   - Size: ~2-3GB quantized
   - Available in mlx-community

3. **Phi-3.5-mini** (Dense)
   - Parameters: 3.8B
   - 4-bit quantized size: ~2GB
   - Good for iOS/Mac devices

## Pipeline Parallelism Benefits for MoE Models

### Why MoE Models Benefit from Pipeline Parallelism:

1. **Expert Distribution**: Different experts can be distributed across different Mac minis
2. **Load Balancing**: Routing decisions can distribute computation
3. **Memory Efficiency**: Only activated experts need to be loaded per device
4. **Layer Distribution**: MoE layers can be split while keeping routing logic

### Current Exo Implementation:
The DeepSeek V2/V3 models already include:
- Expert weight handling (`n_routed_experts`)
- Weight sanitization for distributed execution
- Shard-aware model loading
- Identity blocks for non-participating layers

## Recommended Next Steps:

1. **Test Qwen1.5-MoE-A2.7B** with current exo infrastructure
2. **Add Mixtral support** with proper quantization
3. **Implement expert-aware sharding** for optimal distribution
4. **Test memory usage** on actual 16GB Mac minis

## Layer Counts and Architecture Details:

### Qwen1.5-MoE-A2.7B:
- Layers: Similar to base Qwen1.5 architecture (~24-28 layers estimated)
- MoE layers: 8 experts per MLP layer
- Routing: Top-2 expert selection

### DeepSeek V2/V3:
- Configurable layer count via `num_hidden_layers`
- Expert count via `n_routed_experts` parameter
- Already pipeline-ready with shard support

## Memory Estimation for 16GB Mac Mini:
- System overhead: ~2GB
- Available for model: ~14GB
- Recommended model size: <8GB (leaves room for KV cache and processing)
- Target: 4-bit quantized MoE models with <5B activated parameters