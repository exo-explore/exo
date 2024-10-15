import time
from tinygrad import Tensor
from tinygrad import dtypes

def measure_runtime(fn, *args, **kwargs):
    # warmup
    for _ in range(5):
        fn(*args, **kwargs).realize()

    num_iters = 100
    tic = time.perf_counter()
    for _ in range(num_iters):
        fn(*args, **kwargs).realize()
    toc = time.perf_counter()

    sec = (toc - tic) / num_iters
    return sec

def time_matmul(n=1024, dtype=dtypes.float32):
    Tensor.manual_seed(333)
    a = Tensor.rand(n, n, dtype=dtype)
    b = Tensor.rand(n, n, dtype=dtype)
    a.realize()
    b.realize()
    sec = measure_runtime(Tensor.matmul, a, b)
    tflops = (2 * (n ** 3) / sec) / 1e12
    return tflops

def tinygrad_benchmark_tflops():
    f32 = round(time_matmul(n=1024, dtype=dtypes.float32), 2)
    f16 = round(time_matmul(n=1024, dtype=dtypes.float16), 2)
    return (f32, f16, 0.0)

if __name__ == "__main__":
    print(*tinygrad_benchmark_tflops())