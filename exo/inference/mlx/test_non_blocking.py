import asyncio
import time
import numpy as np
from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard
from exo.models import build_base_shard
from collections import deque
from statistics import mean, median

async def test_non_blocking():
    # Setup
    shard_downloader = NewShardDownloader()
    engine = MLXDynamicShardInferenceEngine(shard_downloader)
    _shard = build_base_shard("llama-3.1-8b", "MLXDynamicShardInferenceEngine")
    shard = Shard(_shard.model_id, _shard.start_layer, _shard.n_layers - 1, _shard.n_layers)
    await engine.ensure_shard(shard)
    
    queue = asyncio.Queue()
    measurements = deque(maxlen=1000000)
    running = True

    async def mlx_worker():
        try:
            start_time = time.time()
            count = 0
            while running and (time.time() - start_time) < 5:  # Hard time limit
                start = time.perf_counter_ns()
                await engine.infer_prompt("req1", shard, "test prompt")
                duration = (time.perf_counter_ns() - start) / 1_000_000  # Convert to ms
                count += 1
                print(f"MLX operation {count} took: {duration:.3f}ms")
        except asyncio.CancelledError:
            pass
        finally:
            print(f"\nTotal MLX operations completed: {count}")
            print(f"Average rate: {count/5:.1f} ops/second")

    async def latency_producer():
        try:
            start_time = time.perf_counter_ns()
            count = 0
            while running:
                await queue.put(time.perf_counter_ns())
                count += 1
                await asyncio.sleep(0)  # Yield to event loop without delay
            duration = (time.perf_counter_ns() - start_time) / 1e9  # Convert to seconds
            print(f"\nProducer iterations: {count}")
            print(f"Producer rate: {count/duration:.1f} iterations/second")
        except asyncio.CancelledError:
            pass

    async def latency_consumer():
        try:
            while running:
                timestamp = await queue.get()
                latency = (time.perf_counter_ns() - timestamp) / 1_000_000  # Convert to ms
                measurements.append(latency)
                queue.task_done()
        except asyncio.CancelledError:
            pass

    tasks = [
        asyncio.create_task(mlx_worker()),
        asyncio.create_task(latency_producer()),
        asyncio.create_task(latency_consumer())
    ]
    
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=6)
    except asyncio.TimeoutError:
        print("\nTest timed out")
    finally:
        running = False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"\nFinal measurement count: {len(measurements)}")

if __name__ == "__main__":
    asyncio.run(test_non_blocking())
