import asyncio
import time
import psutil
import os
import argparse
from exo.inference.inference_engine import get_inference_engine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.models import model_base_shards
from exo.inference.tokenizers import resolve_tokenizer

async def run_inference_test(
    engine,
    tokenizer,
    shard,
    prompt: str,
) -> dict:
    """Run inference test directly with the engine."""
    
    # Measure initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Simplify to single run metrics
    latencies = []
    token_counts = []
    peak_memory = initial_memory
    
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("Running inference pass...")
    start_time = time.time()
    
    # Collect generated tokens
    generated_tokens = []
    
    # Run inference
    result, _, is_finished = await engine.infer_prompt("test", shard, formatted_prompt)
    while not is_finished and len(generated_tokens) < 512:  # Max 512 tokens
        if result.size == 1:
            generated_tokens.append(result.item())
            # Get next token
            result, _, is_finished = await engine.infer_tensor("test", shard, result)
    
    end_time = time.time()
    latency = end_time - start_time
    latencies.append(latency)
    token_counts.append(len(generated_tokens))
    
    print("\nGenerated text:")
    print(tokenizer.decode(generated_tokens))
    
    current_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = max(peak_memory, current_memory)
    
    return {
        "model": shard.model_id,
        "avg_latency": latencies[0],
        "avg_tokens": token_counts[0],
        "tokens_per_second": token_counts[0] / latencies[0],
        "initial_memory_mb": initial_memory,
        "memory_increase_mb": peak_memory - initial_memory,
        "peak_memory_mb": peak_memory
    }

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run quantization test for a specific model')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., "llama-3.1-8b")')
    parser.add_argument('--prompt', type=str, required=True, help='Test prompt')
    parser.add_argument('--quant', type=str, choices=['int8', 'nf4'], required=False,
                       help='Quantization level (int8, nf4)')
    
    args = parser.parse_args()
    
    print(f"\n=== Testing {args.model} with quantization {args.quant} ===")
    
    # Initialize downloader
    downloader = HFShardDownloader()
    
    # Create engine
    engine = get_inference_engine("tinygrad", downloader, quantize=args.quant)
    
    # Get model shard
    shard = model_base_shards.get(args.model, {}).get(engine.__class__.__name__)
    if not shard:
        print(f"Unsupported model: {args.model}")
        return
    
    # Get tokenizer
    tokenizer = await resolve_tokenizer(shard.model_id)
    
    # Ensure model is loaded
    await engine.ensure_shard(shard)
    
    # Run inference test
    result = await run_inference_test(
        engine,
        tokenizer,
        shard,
        args.prompt
    )
    result['quantization'] = args.quant
    
    # Print results
    print("\n=== Results ===")
    print(f"{'Model':<15} {'Quant':<8} {'Avg Latency':<12} {'Tokens/sec':<10} {'Memory (MB)':<12}")
    print("-" * 65)
    print(f"{result['model']:<15} {result['quantization']:<8} {result['avg_latency']:.2f}s "
          f"{result['tokens_per_second']:.2f} {result['memory_increase_mb']:.1f}")

if __name__ == "__main__":
    asyncio.run(main())