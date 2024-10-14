# inspired by https://github.com/ml-explore/mlx/blob/main/benchmarks/python/batch_matmul_bench.py

import time
import mlx.core as mx

def measure_runtime(fn, *args, **kwargs):
    # warmup
    for _ in range(5):
        x = mx.eval(fn(*args, **kwargs))

    num_iters = 100
    tic = time.perf_counter()
    for _ in range(num_iters):
        x = mx.eval(fn(*args, **kwargs))
    toc = time.perf_counter()

    sec = (toc - tic) / num_iters
    return sec

def time_matmul(n=1024, dtype=mx.float32, device=mx.gpu):
    mx.random.seed(333)
    a = mx.random.uniform(shape=(n, n), dtype=dtype, stream=device)
    b = mx.random.uniform(shape=(n, n), dtype=dtype, stream=device)
    mx.eval(a, b)
    sec = measure_runtime(mx.matmul, a, b, stream=device)
    tflops = (2 * (n ** 3) / sec) / 1e12
    return tflops

def mlx_benchmark_tflops():
    f32 = round(time_matmul(n=1024), 2)
    f16 = round(time_matmul(n=1024, dtype=mx.float16), 2)
    return (f32, f16, 0.0)

if __name__ == "__main__":
    print(*mlx_benchmark_tflops())
