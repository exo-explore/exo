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
    model_name = "llama-3.1-8b"
    test_prompts = [
        "What is the meaning of exo?",
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "Create a recipe for chocolate chip cookies.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis.",
        "Write a haiku about artificial intelligence.",
        "How does climate change affect ocean ecosystems?",
    ]
    quantization_levels = ["int8", "nf4", None]
    results = []
    
    # Initialize downloader once
    downloader = HFShardDownloader()
    
    for quant in quantization_levels:
        for test_prompt in test_prompts:
            print(f"\n=== Testing {model_name} with quantization {quant or 'fp32'} ===")
            print(f"Prompt: {test_prompt[:50]}...")  # Show first 50 chars of prompt
            
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
                
                del engine
                del node
                gc.collect()
                await asyncio.sleep(1)  # Give time for cleanup
                
            except Exception as e:
                print(f"Error testing quantization {quant}: {str(e)}")
                continue

    # Print results table with all metrics
    print("\n=== Detailed Results ===")
    
    # Print summary by quantization level
    print("\nSummary by Quantization Level:")
    for quant in quantization_levels:
        quant_results = [r for r in results if r['quantization'] == (quant or 'fp32')]
        if quant_results:
            avg_latency = statistics.mean([r['avg_latency'] for r in quant_results])
            avg_tokens_sec = statistics.mean([r['tokens_per_second'] for r in quant_results])
            avg_memory = statistics.mean([r['memory_increase_mb'] for r in quant_results])
            print(f"\n{quant or 'fp32'} Quantization:")
            print(f"  Average Latency: {avg_latency:.2f}s")
            print(f"  Average Tokens/sec: {avg_tokens_sec:.2f}")
            print(f"  Average Memory Increase: {avg_memory:.1f}MB")

    # Print detailed results for each test
    print("\nDetailed Results by Test:")
    print(f"{'Model':<15} {'Quant':<8} {'Prompt':<30} {'Latency':<10} {'Tokens/sec':<12} "
          f"{'Avg Tokens':<12} {'Memory MB':<12} {'Std Dev':<10}")
    print("-" * 100)
    
    for r in results:
        prompt = next(p[:27] + "..." for p in test_prompts 
                     if p in [p for p in test_prompts])  # Get matching prompt
        print(f"{r['model']:<15} "
              f"{r['quantization']:<8} "
              f"{prompt:<30} "
              f"{r['avg_latency']:.2f}s "
              f"{r['tokens_per_second']:.2f} "
              f"{r['avg_tokens']:.1f} "
              f"{r['memory_increase_mb']:.1f} "
              f"{r['std_latency']:.3f}")

    # Print memory usage statistics
    print("\nMemory Usage Statistics:")
    for quant in quantization_levels:
        quant_results = [r for r in results if r['quantization'] == (quant or 'fp32')]
        if quant_results:
            print(f"\n{quant or 'fp32'} Quantization:")
            print(f"  Initial Memory: {quant_results[0]['initial_memory_mb']:.1f}MB")
            print(f"  Peak Memory: {max(r['peak_memory_mb'] for r in quant_results):.1f}MB")
            print(f"  Average Memory Increase: {statistics.mean([r['memory_increase_mb'] for r in quant_results]):.1f}MB")

if __name__ == "__main__":
    # Set environment variable to use single thread for SQLite operations
    os.environ["TINYGRAD_CACHE_DIR"] = ".cache"  # Use local cache directory
    asyncio.run(main())