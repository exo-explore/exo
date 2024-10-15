# inspired by https://github.com/ml-explore/mlx/blob/main/benchmarks/python/batch_matmul_bench.py

import time
import mlx.core as mx

def _benchmark_matmul(n, dtype):
    a = mx.random.uniform(shape=(n, n), dtype=dtype)
    b = mx.random.uniform(shape=(n, n), dtype=dtype)

    # warmup
    for _ in range(5):
        x = mx.eval(a@b)
        mx.synchronize()

    num_iters = 100
    tic = time.perf_counter()
    for _ in range(num_iters):
        x = mx.eval(a@b)
        mx.synchronize()
    toc = time.perf_counter()

    sec = (toc - tic) / num_iters
    tflops = (2 * (n ** 3) / sec) / 1e12
    return tflops

def mlx_benchmark_tflops():
    n = 2048
    f32 = round(_benchmark_matmul(n, dtype=mx.float32), 2)
    f16 = round(_benchmark_matmul(n, dtype=mx.float16), 2)
    return (f32, f16, 0.0)

if __name__ == "__main__":
    print(*mlx_benchmark_tflops())
