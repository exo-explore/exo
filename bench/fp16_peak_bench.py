"""Measure actual achievable FP16 matmul throughput on this GPU.

Establishes the practical peak TFLOPS for comparison with SDPA utilization.

Usage: uv run python bench/fp16_peak_bench.py
"""

import time

import mlx.core as mx

WARMUP = 3
ITERS = 5


def bench_matmul(m: int, n: int, k: int) -> tuple[float, float]:
    """Returns (time_ms, tflops)."""
    a = mx.random.normal(shape=(m, k)).astype(mx.float16)
    b = mx.random.normal(shape=(k, n)).astype(mx.float16)
    mx.eval(a, b)

    for _ in range(WARMUP):
        c = a @ b
        mx.eval(c)
        mx.synchronize()

    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        mx.eval(c)
        mx.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    flops = 2.0 * m * n * k
    tflops = flops / avg / 1e12
    del a, b
    mx.clear_cache()
    return avg * 1000, tflops


def bench_batched_matmul(b: int, m: int, n: int, k: int) -> tuple[float, float]:
    """Batched matmul to simulate GQA attention."""
    a = mx.random.normal(shape=(b, m, k)).astype(mx.float16)
    bt = mx.random.normal(shape=(b, k, n)).astype(mx.float16)
    mx.eval(a, bt)

    for _ in range(WARMUP):
        c = a @ bt
        mx.eval(c)
        mx.synchronize()

    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        c = a @ bt
        mx.eval(c)
        mx.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    flops = 2.0 * b * m * n * k
    tflops = flops / avg / 1e12
    del a, bt
    mx.clear_cache()
    return avg * 1000, tflops


def main():
    print("=== FP16 Matmul Peak Throughput ===")
    print()

    # Large square matmuls (saturate the GPU)
    print(f"{'Shape':>30} | {'Time (ms)':>10} | {'TFLOPS':>8}")
    print("-" * 55)

    for size in [1024, 2048, 4096, 8192]:
        t, tf = bench_matmul(size, size, size)
        print(f"{'['+str(size)+','+str(size)+'] @ ['+str(size)+','+str(size)+']':>30} | {t:>10.1f} | {tf:>8.2f}")

    print()
    print("=== Attention-like matmuls (Qwen3.5 config) ===")
    print(f"head_dim=256, qL=2048")
    print()
    print(f"{'Operation':>40} | {'Time (ms)':>10} | {'TFLOPS':>8}")
    print("-" * 65)

    # QK^T: [2048, 256] @ [256, kL] per head
    # With 32 Q heads: batched [32, 2048, 256] @ [32, 256, kL]
    # But GQA means K has only 2 heads, expanded to 32
    for kl in [32768, 131072, 262144]:
        # Single head QK^T
        t, tf = bench_matmul(2048, kl, 256)
        print(f"{'QK^T 1-head [2048,256]@[256,'+str(kl//1024)+'K]':>40} | {t:>10.1f} | {tf:>8.2f}")

    print()

    # Batched: 32 Q heads
    for kl in [32768, 131072, 262144]:
        t, tf = bench_batched_matmul(32, 2048, kl, 256)
        print(f"{'QK^T 32-head [32,2048,256]@[32,256,'+str(kl//1024)+'K]':>40} | {t:>10.1f} | {tf:>8.2f}")

    print()

    # Full SDPA compute budget: QK + PV for all 32 heads
    for kl in [32768, 131072, 262144]:
        # QK: [32, 2048, 256] @ [32, 256, kL]
        tqk, _ = bench_batched_matmul(32, 2048, kl, 256)
        # PV: [32, 2048, kL] @ [32, kL, 256]
        tpv, _ = bench_batched_matmul(32, 2048, 256, kl)
        total = tqk + tpv
        flops = 2 * (2.0 * 32 * 2048 * kl * 256)  # QK + PV
        tflops = flops / (total / 1000) / 1e12
        print(f"{'QK+PV 32h kL='+str(kl//1024)+'K':>40} | {total:>10.1f} | {tflops:>8.2f}")


if __name__ == "__main__":
    main()
