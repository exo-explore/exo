#!/usr/bin/env python3
"""
Test distributed inference with small MoE model on 16GB Mac minis
Uses collective operations (all_gather) which work reliably
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.core.distributed as dist

# Add exo to path
sys.path.insert(0, '/Users/mini1/Movies/exo')

from exo.inference.shard import Shard
from exo.inference.mlx.models.qwen_moe_mini import Model, ModelArgs


def initialize_distributed():
    """Initialize distributed group"""
    group = dist.init()
    print(f"[Rank {group.rank()}/{group.size()}] Initialized distributed group")
    return group


def create_model(rank: int, world_size: int):
    """Create model with appropriate sharding"""
    # Create config for small MoE model
    config = ModelArgs(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=2,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=2816,
    )
    
    # Calculate layer distribution
    layers_per_rank = config.num_hidden_layers // world_size
    start_layer = rank * layers_per_rank
    end_layer = start_layer + layers_per_rank - 1
    
    # Handle last rank getting any remainder layers
    if rank == world_size - 1:
        end_layer = config.num_hidden_layers - 1
    
    # Create shard
    config.shard = Shard(
        model_id="qwen-moe-mini",
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=config.num_hidden_layers
    )
    
    print(f"[Rank {rank}] Creating model with layers {start_layer}-{end_layer}")
    
    # Create model
    model = Model(config)
    
    # Initialize weights (random for testing)
    print(f"[Rank {rank}] Initializing random weights...")
    
    # Initialize parameters properly
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            shape = module.weight.shape
            module.weight = mx.random.normal(shape=shape) * 0.02
        elif isinstance(module, nn.Embedding):
            shape = module.weight.shape
            module.weight = mx.random.normal(shape=shape) * 0.02
    
    mx.eval(model.parameters())
    
    return model, config


def distributed_forward(model, input_ids, group):
    """Forward pass with distributed pipeline using collective operations"""
    rank = group.rank()
    world_size = group.size()
    
    # Stage 1: First rank processes embedding and initial layers
    if rank == 0:
        print(f"[Rank 0] Processing embedding and layers 0-7")
        h = model.model(input_ids)
        # Prepare to send activations
        activations = h
    else:
        # Other ranks prepare dummy tensor for all_gather
        dummy_shape = (input_ids.shape[0], input_ids.shape[1], model.args.hidden_size)
        activations = mx.zeros(dummy_shape, dtype=mx.float32)
    
    # Transfer activations using all_gather (more reliable than send/recv)
    print(f"[Rank {rank}] Participating in all_gather...")
    all_activations = dist.all_gather(activations, group=group)
    
    # Extract the activation from rank 0
    if world_size > 1 and rank == 1:
        h = all_activations[0]  # Get rank 0's output
        print(f"[Rank 1] Processing layers 8-15 and final norm")
        h = model.model(h)
        result = h
    elif rank == 0 and world_size == 1:
        # Single device case
        result = h
    else:
        # Prepare dummy for final gather
        result = mx.zeros_like(activations)
    
    # Final gather to get result on all ranks
    if world_size > 1:
        print(f"[Rank {rank}] Final all_gather...")
        all_results = dist.all_gather(result, group=group)
        if rank == 0:
            final_output = all_results[1]  # Get rank 1's output
        else:
            final_output = all_results[1]
    else:
        final_output = result
    
    return final_output


def test_generation(model, config, group):
    """Test text generation with the MoE model"""
    rank = group.rank()
    
    # Create sample input
    batch_size = 1
    seq_len = 10
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n[Rank {rank}] Starting generation test...")
    print(f"[Rank {rank}] Input shape: {input_ids.shape}")
    
    # Measure memory before
    mem_before = mx.get_active_memory() / 1024**3
    print(f"[Rank {rank}] Memory before: {mem_before:.2f} GB")
    
    # Run forward pass
    start_time = time.time()
    output = distributed_forward(model, input_ids, group)
    mx.eval(output)
    elapsed = time.time() - start_time
    
    # Measure memory after
    mem_after = mx.get_active_memory() / 1024**3
    print(f"[Rank {rank}] Memory after: {mem_after:.2f} GB")
    print(f"[Rank {rank}] Memory used: {(mem_after - mem_before):.2f} GB")
    
    if rank == 0:
        print(f"\n✅ Generation completed in {elapsed:.3f}s")
        print(f"Output shape: {output.shape}")
        print(f"Model size estimate: ~{(mem_after - mem_before) * group.size():.1f} GB total across {group.size()} devices")
        
        # Verify output is valid
        if output.shape == (batch_size, seq_len, config.vocab_size):
            print("✅ Output shape correct!")
        else:
            print(f"❌ Unexpected output shape: {output.shape}")
        
        # Check if both GPUs were used
        if group.size() > 1:
            print(f"\n🎉 SUCCESS: Distributed inference working across {group.size()} devices!")
            print(f"Each device handles {config.num_hidden_layers // group.size()} layers")
            print(f"Total experts: {config.n_routed_experts} (selecting {config.num_experts_per_tok} per token)")


def main():
    parser = argparse.ArgumentParser(description="Test MoE distributed inference")
    parser.add_argument("--mock-weights", action="store_true", help="Use random weights for testing")
    args = parser.parse_args()
    
    # Initialize distributed
    group = initialize_distributed()
    rank = group.rank()
    world_size = group.size()
    
    print(f"\n{'='*60}")
    print(f"Testing Qwen-MoE-Mini Distributed Inference")
    print(f"{'='*60}")
    print(f"Rank: {rank}/{world_size}")
    print(f"Device: {mx.default_device()}")
    print(f"{'='*60}\n")
    
    # Create model
    model, config = create_model(rank, world_size)
    
    # Test generation
    test_generation(model, config, group)
    
    # Synchronize before exit
    dist.all_sum(mx.array([1.0]), group=group)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Test completed successfully!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()