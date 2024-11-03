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

async def profile_inference(
    model_name: str,
    prompt: str,
    quantization: Optional[str] = None,
    num_runs: int = 3
) -> dict:
    """Profile inference performance for a given model and quantization level."""
    
    # Initialize components
    downloader = HFShardDownloader()
    engine = get_inference_engine("tinygrad", downloader, quantize=quantization)
    
    # Get model shard
    shard = model_base_shards.get(model_name, {}).get(engine.__class__.__name__)
    if not shard:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Setup tokenizer
    tokenizer = await resolve_tokenizer(shard.model_id)
    encoded_prompt = tokenizer.encode(prompt)
    
    # Ensure model is downloaded
    await downloader.ensure_shard(shard)
    
    # Warmup run
    print(f"\nWarmup run for {model_name} ({quantization or 'fp32'})...")
    _ = await engine.infer_prompt(model_name, shard, encoded_prompt)
    
    # Measure memory after model loading
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
    post_load_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = post_load_memory - initial_memory
    
    # Profile multiple runs
    latencies = []
    token_counts = []
    peak_memory = post_load_memory
    
    print(f"Running {num_runs} inference passes...")
    for i in range(num_runs):
        start_time = time.time()
        tokens = await engine.infer_prompt(model_name, shard, encoded_prompt)
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
        token_counts.append(len(tokens))
        
        print(f"Run {i+1}: Generated {len(tokens)} tokens in {latency:.2f}s")
        
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
    
    return {
        "model": model_name,
        "quantization": quantization or "fp32",
        "avg_latency": statistics.mean(latencies),
        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "avg_tokens": statistics.mean(token_counts),
        "tokens_per_second": statistics.mean(token_counts) / statistics.mean(latencies),
        "initial_memory_mb": initial_memory,
        "memory_increase_mb": memory_increase,
        "peak_memory_mb": peak_memory
    }

async def main():
    models_to_test = ["llama-3.1-8b"]  # Add more models as needed
    quantization_levels = [None, "int8", "nf4"]
    test_prompt = "Explain the concept of quantum computing in simple terms."
    
    results = []
    
    for model in models_to_test:
        print(f"\n=== Testing {model} ===")
        for quant in quantization_levels:
            try:
                result = await profile_inference(model, test_prompt, quant)
                results.append(result)
            except Exception as e:
                print(f"Error testing {model} with {quant or 'fp32'} quantization: {str(e)}")
    
    # Print results table
    print("\n=== Results ===")
    print(f"{'Model':<15} {'Quant':<8} {'Avg Latency':<12} {'Tokens/sec':<10} {'Memory (MB)':<12}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<15} {r['quantization']:<8} {r['avg_latency']:.2f}s "
              f"{r['tokens_per_second']:.2f} {r['memory_increase_mb']:.1f}")

if __name__ == "__main__":
    asyncio.run(main()) 