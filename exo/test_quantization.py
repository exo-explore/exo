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
    model_name = "llama-3.1-70b"
    test_prompt = "What is the meaning of exo?"
    quantization_levels = ["int8", "nf4", None]
    results = []
    
    # Initialize downloader once
    downloader = HFShardDownloader()
    
    for quant in quantization_levels:
        print(f"\n=== Testing {model_name} with quantization {quant or 'fp32'} ===")
        
        try:
            import gc
            gc.collect()
            
            # Create fresh engine for each test
            engine = get_inference_engine("tinygrad", downloader, quantize=quant)
            
            # Get model shard
            shard = model_base_shards.get(model_name, {}).get(engine.__class__.__name__)
            if not shard:
                print(f"Unsupported model: {model_name}")
                continue
                
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
            
            # Ensure model is loaded - run in main thread
            await engine.ensure_shard(shard)
            
            # Run inference test
            result = await run_inference_test(
                node,
                tokenizer,
                shard,
                test_prompt
            )
            result['quantization'] = quant or 'fp32'  # Add quantization info to results
            results.append(result)
            
            # Clean up more aggressively
            del engine
            del node
            gc.collect()
            await asyncio.sleep(1)  # Give time for cleanup
            
        except Exception as e:
            print(f"Error testing quantization {quant}: {str(e)}")
            continue

    # Print results table
    print("\n=== Results ===")
    print(f"{'Model':<15} {'Quant':<8} {'Avg Latency':<12} {'Tokens/sec':<10} {'Memory (MB)':<12}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<15} {r['quantization']:<8} {r['avg_latency']:.2f}s "
              f"{r['tokens_per_second']:.2f} {r['memory_increase_mb']:.1f}")

if __name__ == "__main__":
    # Set environment variable to use single thread for SQLite operations
    os.environ["TINYGRAD_CACHE_DIR"] = ".cache"  # Use local cache directory
    asyncio.run(main())