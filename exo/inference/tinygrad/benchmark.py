import time
from tinygrad import Tensor
from tinygrad import Device, dtypes
from exo.topology.device_flops import DeviceFlops

def _benchmark_matmul(n, dtype) -> float:
    a = Tensor.rand(n, n, dtype=dtype)
    b = Tensor.rand(n, n, dtype=dtype)

    # warmup
    for _ in range(5):
        (a@b).realize()
        Device.default.synchronize()
    
    num_iters = 100
    tic = time.perf_counter()
    for _ in range(num_iters):
        (a@b).realize()
        Device.default.synchronize()
    toc = time.perf_counter()

    sec = (toc - tic) / num_iters
    tflops = (2 * (n ** 3) / sec) / 1e12
    return tflops

def tinygrad_benchmark_tflops() -> DeviceFlops:
    n = 2048
    f32 = round(_benchmark_matmul(n, dtype=dtypes.float32), 2)
    f16 = round(_benchmark_matmul(n, dtype=dtypes.float16), 2)
    return DeviceFlops(fp32=f32, fp16=f16, int8=0.0)

if __name__ == "__main__":
    print(*tinygrad_benchmark_tflops())