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

async def run_inference_test(
    node: StandardNode,
    tokenizer,
    shard,
    prompt: str,
    num_runs: int = 1,
) -> dict:
    """Run inference test with an initialized node."""
    
    # Measure initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Profile multiple runs
    latencies = []
    token_counts = []
    peak_memory = initial_memory
    
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print(f"Running {num_runs} inference passes...")
    for i in range(num_runs):
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
            latencies.append(latency)
            token_counts.append(len(tokens))
            
            if i == 0:
                print("\nGenerated text:")
                print(tokenizer.decode(tokens))
            
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
        finally:
            node.on_token.deregister(callback_id)
    
    return {
        "model": shard.model_id,
        "avg_latency": statistics.mean(latencies),
        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "avg_tokens": statistics.mean(token_counts),
        "tokens_per_second": statistics.mean(token_counts) / statistics.mean(latencies),
        "initial_memory_mb": initial_memory,
        "memory_increase_mb": peak_memory - initial_memory,
        "peak_memory_mb": peak_memory
    }

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run quantization test for a specific model')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., "llama-3.1-8b")')
    parser.add_argument('--prompt', type=str, required=True, help='Test prompt')
    parser.add_argument('--quant', type=str, choices=['int8', 'nf4', 'fp32'], required=True,
                       help='Quantization level (fp32 for no quantization)')
    
    args = parser.parse_args()
    quant = None if args.quant == 'fp32' else args.quant
    
    print(f"\n=== Testing {args.model} with quantization {args.quant} ===")
    
    try:
        import gc
        gc.collect()
        
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
        result['quantization'] = args.quant
        
        # Print results
        print("\n=== Results ===")
        print(f"{'Model':<15} {'Quant':<8} {'Avg Latency':<12} {'Tokens/sec':<10} {'Memory (MB)':<12}")
        print("-" * 65)
        print(f"{result['model']:<15} {result['quantization']:<8} {result['avg_latency']:.2f}s "
              f"{result['tokens_per_second']:.2f} {result['memory_increase_mb']:.1f}")
        
    except Exception as e:
        print(f"Error running test: {str(e)}")

if __name__ == "__main__":
    os.environ["TINYGRAD_CACHE_DIR"] = ".cache"
    asyncio.run(main())