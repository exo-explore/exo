import asyncio
import time
import statistics
from typing import List, Optional
from exo.inference.inference_engine import get_inference_engine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.models import model_base_shards
from exo.inference.tokenizers import resolve_tokenizer
import psutil
import os
import numpy as np
import json
from exo.orchestration.standard_node import StandardNode
import uuid
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy

async def profile_inference(
    model_name: str,
    prompt: str,
    quantization: Optional[str] = None,
    num_runs: int = 1,
    downloader: Optional[HFShardDownloader] = None
) -> dict:
    """Profile inference performance for a given model and quantization level."""
    
    # Use passed downloader or create new one
    downloader = downloader or HFShardDownloader()
    engine = get_inference_engine("tinygrad", downloader, quantize=quantization)
    
    # Create node (similar to main.py)
    node = StandardNode(
        str(uuid.uuid4()),  # random node id
        None,              # no server needed
        engine,
        None,              # no discovery needed
        partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
        max_generate_tokens=512,
        shard_downloader=downloader
    )
    
    # Initialize topology without networking
    node.topology.update_node(node.id, node.device_capabilities)
    
    # Get model shard
    shard = model_base_shards.get(model_name, {}).get(engine.__class__.__name__)
    if not shard:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Get proper tokenizer and format prompt
    tokenizer = await resolve_tokenizer(shard.model_id)
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Ensure model is loaded before starting inference
    await engine.ensure_shard(shard)
    
    # Measure initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Profile multiple runs
    latencies = []
    token_counts = []
    peak_memory = initial_memory
    
    print(f"Running {num_runs} inference passes...")
    for i in range(num_runs):
        start_time = time.time()
        
        # Use node's callback system to get tokens
        request_id = str(uuid.uuid4())
        callback_id = f"test-{request_id}"
        callback = node.on_token.register(callback_id)
        
        try:
            await node.process_prompt(shard, formatted_prompt, None, request_id=request_id)
            _, tokens, _ = await callback.wait(
                lambda _request_id, tokens, is_finished: _request_id == request_id and is_finished,
                timeout=300
            )
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            token_counts.append(len(tokens))
            
            # Print generated text for verification
            if i == 0:
                print("\nGenerated text:")
                print(tokenizer.decode(tokens))
            
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
        finally:
            node.on_token.deregister(callback_id)
            
        # Clear any cached states between runs
        await engine.clear_cache()
    
    return {
        "model": model_name,
        "quantization": quantization or "fp32",
        "avg_latency": statistics.mean(latencies),
        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "avg_tokens": statistics.mean(token_counts),
        "tokens_per_second": statistics.mean(token_counts) / statistics.mean(latencies),
        "initial_memory_mb": initial_memory,
        "memory_increase_mb": peak_memory - initial_memory,
        "peak_memory_mb": peak_memory
    }

    
async def main():
    models_to_test = ["llama-3.1-8b"]
    quantization_levels = [None, "int8", "nf4"]
    test_prompt = "What is the meaning of exo?"
    
    # Initialize downloader once at the start
    downloader = HFShardDownloader()
    
    results = []
    for model in models_to_test:
        print(f"\n=== Testing {model} ===")
        for quant in quantization_levels:
            # Pass the downloader instance
            result = await profile_inference(
                model, 
                test_prompt, 
                quantization=quant,
                downloader=downloader
            )
            results.append(result)

    
    # Print results table
    print("\n=== Results ===")
    print(f"{'Model':<15} {'Quant':<8} {'Avg Latency':<12} {'Tokens/sec':<10} {'Memory (MB)':<12}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<15} {r['quantization']:<8} {r['avg_latency']:.2f}s "
              f"{r['tokens_per_second']:.2f} {r['memory_increase_mb']:.1f}")

if __name__ == "__main__":
    asyncio.run(main()) 