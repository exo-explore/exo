import asyncio
import time
import statistics
from exo.inference.inference_engine import get_inference_engine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.models import model_base_shards
from exo.inference.tokenizers import resolve_tokenizer
import psutil
import os
import uuid
from exo.orchestration.standard_node import StandardNode
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
import argparse
import gc
from typing import List, Tuple

async def track_memory() -> Tuple[float, float]:
    """Get current and peak memory usage in MB."""
    gc.collect()  # Force garbage collection
    process = psutil.Process(os.getpid())
    current = process.memory_info().rss / 1024 / 1024
    
    # Get memory info for child processes too
    children_mem = sum(child.memory_info().rss for child in process.children(recursive=True))
    children_mem = children_mem / 1024 / 1024
    
    return current + children_mem, process.memory_info().peak_rss / 1024 / 1024

async def run_inference_test(
    node: StandardNode,
    tokenizer,
    shard,
    prompt: str,
) -> dict:
    """Run inference test with an initialized node."""
    
    # Track memory at key points
    memory_points = []
    
    # Initial memory
    initial_memory, _ = await track_memory()
    memory_points.append(("Initial", initial_memory))
    
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Memory after prompt formatting
    current_memory, _ = await track_memory()
    memory_points.append(("After prompt format", current_memory))
    
    print("Running inference pass...")
    start_time = time.time()
    
    request_id = str(uuid.uuid4())
    callback_id = f"test-{request_id}"
    callback = node.on_token.register(callback_id)
    
    try:
        # Start generation
        await node.process_prompt(shard, formatted_prompt, None, request_id=request_id)
        current_memory, _ = await track_memory()
        memory_points.append(("After starting generation", current_memory))
        
        # Track memory during generation
        token_count = 0
        last_memory_check = time.time()
        memory_check_interval = 1.0  # Check memory every second
        
        while True:
            try:
                _, tokens, is_finished = await callback.wait(
                    lambda _request_id, tokens, is_finished: _request_id == request_id,
                    timeout=1.0
                )
                token_count = len(tokens)
                
                # Periodically check memory
                if time.time() - last_memory_check >= memory_check_interval:
                    current_memory, _ = await track_memory()
                    memory_points.append((f"During generation ({token_count} tokens)", current_memory))
                    last_memory_check = time.time()
                
                if is_finished:
                    break
                    
            except asyncio.TimeoutError:
                continue
        
        end_time = time.time()
        latency = end_time - start_time

        print("\nGenerated text:")
        print(tokenizer.decode(tokens))
        
        # Final memory check
        final_memory, peak_memory = await track_memory()
        memory_points.append(("Final", final_memory))
        
    finally:
        node.on_token.deregister(callback_id)
    
    # Print detailed memory profile
    print("\n=== Memory Profile ===")
    print(f"{'Stage':<30} {'Memory (MB)':<15} {'Delta (MB)':<15}")
    print("-" * 60)
    for i, (stage, memory) in enumerate(memory_points):
        delta = memory - memory_points[0][1] if i > 0 else 0
        print(f"{stage:<30} {memory:.1f} MB {delta:>+.1f} MB")
    
    return {
        "model": shard.model_id,
        "latency": latency,
        "tokens": token_count,
        "tokens_per_second": token_count / latency,
        "initial_memory_mb": initial_memory,
        "memory_increase_mb": final_memory - initial_memory,
        "peak_memory_mb": peak_memory,
        "memory_profile": memory_points
    }

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run quantization test for a specific model')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., "llama-3.1-8b")')
    parser.add_argument('--prompt', type=str, required=True, help='Test prompt')
    parser.add_argument('--quant', type=str, choices=['int8', 'nf4', 'none'], default='none',
                       help='Quantization level (int8, nf4, none)')
    
    args = parser.parse_args()
    quant = args.quant if args.quant != 'none' else None
    
    print(f"\n=== Testing {args.model} with quantization {quant or 'none'} ===")
    
        # Initialize downloader
    downloader = HFShardDownloader()
    
    # Create engine
    engine = get_inference_engine("tinygrad", downloader, quantize=quant)
    
    # Get model shard
    shard = model_base_shards.get(args.model, {}).get(engine.__class__.__name__)
    if not shard:
        print(f"Unsupported model: {args.model}")
        return
            
    # Create node
    node = StandardNode(
        str(uuid.uuid4()),
        None,
        engine,
        None,
            partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
            max_generate_tokens=512,
            shard_downloader=downloader
        )
        
    # Initialize topology
    node.topology.update_node(node.id, node.device_capabilities)
    
    # Get tokenizer
    tokenizer = await resolve_tokenizer(shard.model_id)
    
    # Ensure model is loaded
    await engine.ensure_shard(shard)
        
    # Run inference test
    result = await run_inference_test(
        node,
        tokenizer,
        shard,
        args.prompt
        )
    result['quantization'] = quant or 'none'
        
    # Print results
    print("\n=== Results ===")
    print(f"{'Model':<20} {'Quant':<8} {'Latency':<12} {'Tokens':<8} {'Tokens/sec':<12} {'Initial MB':<12} {'Peak MB':<10} {'Increase MB':<12}")
    print("-" * 95)
    print(f"{result['model']:<20} {result['quantization']:<8} {result['latency']:.2f}s "
          f"{result['tokens']:<8} {result['tokens_per_second']:.2f} "
          f"{result['initial_memory_mb']:.1f} {result['peak_memory_mb']:.1f} "
          f"{result['memory_increase_mb']:.1f}")
        

if __name__ == "__main__":
    asyncio.run(main())