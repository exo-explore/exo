import torch
import time

def _benchmark_tflops(n, dtype='f32', num_iterations=100):
    backend = None
    if torch.backends.mps.is_available():
        backend = torch.mps
        device = torch.device("mps")
    elif torch.cuda.is_available():
        backend = torch.cuda
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if dtype == 'f32':
        A = torch.randn((n, n), device=device, dtype=torch.float32)
        B = torch.randn((n, n), device=device, dtype=torch.float32)
    elif dtype == 'f16':
        A = torch.randn((n, n), device=device, dtype=torch.float16)
        B = torch.randn((n, n), device=device, dtype=torch.float16)
    elif dtype == 'int8':
        A = (torch.randint(-128, 127, (n, n), device=device, dtype=torch.int8)).float()
        B = (torch.randint(-128, 127, (n, n), device=device, dtype=torch.int8)).float()
    else:
        raise ValueError("Unsupported data type. Use 'f32', 'f16', or 'int8'.")

    if backend:
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            C = torch.mm(A, B)
            backend.synchronize()
        elapsed_time = time.perf_counter() - start_time
    else:
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            C = torch.mm(A, B)
        elapsed_time = time.perf_counter() - start_time

    flops_per_iteration = 2 * (n ** 3)
    total_flops = flops_per_iteration * num_iterations
    tflops = (total_flops / elapsed_time) / 1e12

    return float(f"{tflops:.2f}")


def benchmark():
    fp32=_benchmark_tflops(2048)
    fp16=_benchmark_tflops(2048, dtype='f16')
    int8=_benchmark_tflops(2048, dtype='int8')

    return (fp32, fp16, int8)


if __name__ == '__main__':
    print(*benchmark())