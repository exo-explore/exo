import asyncio
import time
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

async def run_inference_test(
    node: StandardNode,
    tokenizer,
    shard,
    prompt: str,
) -> dict:
    """Run inference test with an initialized node."""
    
    # Measure memory before model loading
    pre_load_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # Ensure model is loaded
    await node.engine.ensure_shard(shard)
    
    # Measure memory after model loading
    post_load_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    peak_memory = post_load_memory
    
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("Running inference pass...")
    # Single run only
    start_time = time.time()
    
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

        print("\nGenerated text:")
        print(tokenizer.decode(tokens))
        
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
        
    finally:
        node.on_token.deregister(callback_id)
    
    return {
        "model": shard.model_id,
        "latency": latency,
        "tokens": len(tokens),
        "tokens_per_second": len(tokens) / latency,
        "pre_load_memory_mb": pre_load_memory,
        "model_load_increase_mb": post_load_memory - pre_load_memory,
        "inference_increase_mb": peak_memory - post_load_memory,
        "peak_memory_mb": peak_memory
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
    print(f"{'Model':<20} {'Quant':<8} {'Latency':<12} {'Tokens':<8} {'Tokens/sec':<12} {'Pre-load MB':<12} {'Model Load Increase MB':<12} {'Inference Increase MB':<12} {'Peak MB':<10} {'Increase MB':<12}")
    print("-" * 95)
    print(f"{result['model']:<20} {result['quantization']:<8} {result['latency']:.2f}s "
          f"{result['tokens']:<8} {result['tokens_per_second']:.2f} "
          f"{result['pre_load_memory_mb']:.1f} {result['model_load_increase_mb']:.1f} "
          f"{result['inference_increase_mb']:.1f} {result['peak_memory_mb']:.1f} "
          f"{result['memory_increase_mb']:.1f}")
        

if __name__ == "__main__":
    asyncio.run(main())