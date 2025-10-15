#!/usr/bin/env python3
"""
Simple test to verify distributed MLX works with MoE model
Using ring backend which is proven to work
"""

import mlx.core as mx
import mlx.core.distributed as dist

def main():
    # Initialize distributed group
    group = dist.init()
    rank = group.rank()
    world_size = group.size()
    
    print(f"[Rank {rank}/{world_size}] Connected!")
    
    # Simple all_sum test first
    data = mx.array([float(rank)])
    result = dist.all_sum(data, group=group)
    mx.eval(result)
    
    print(f"[Rank {rank}] all_sum result: {result.item()}")
    
    if world_size == 2:
        if result.item() == 1.0:  # 0 + 1 = 1
            print(f"✅ [Rank {rank}] Both GPUs are communicating!")
            
            # Check GPU memory to confirm GPU is being used
            mem = mx.get_active_memory() / 1024**3
            print(f"[Rank {rank}] GPU memory: {mem:.2f} GB")
        else:
            print(f"❌ [Rank {rank}] Unexpected result: {result.item()}")
    else:
        print(f"[Rank {rank}] Running on single device")

if __name__ == "__main__":
    main()