"""Benchmark chunkwise vs sequential DeltaNet kernel for prefill."""

import time

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_chunkwise,
    gated_delta_kernel,
    gated_delta_ops,
)


def bench(fn, *args, warmup=3, repeats=5, label=""):
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(*out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(*out)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    best = min(times)
    print(f"  {label:30s}  avg={avg*1000:8.2f} ms  best={best*1000:8.2f} ms")
    return avg


def main():
    # Qwen3.5 dimensions (per node: 30 layers, 22 DeltaNet + 8 attention)
    Hk, Hv, Dk, Dv = 16, 64, 192, 128
    B = 1

    print(f"Qwen3.5 dims: Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}")
    print()

    A_log = mx.zeros((Hv,))
    dt_bias = mx.ones((Hv,))

    for T in [64, 256, 512, 1024, 2048, 4096]:
        print(f"T={T}")
        mx.random.seed(0)

        q = mx.random.normal(shape=(B, T, Hk, Dk)).astype(mx.float16)
        k = mx.random.normal(shape=(B, T, Hk, Dk)).astype(mx.float16)
        v = mx.random.normal(shape=(B, T, Hv, Dv)).astype(mx.float16)
        a = -7.0 + mx.random.normal(shape=(B, T, Hv)).astype(mx.float16) * 0.3
        b = mx.random.normal(shape=(B, T, Hv)).astype(mx.float16)
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b.astype(mx.float32))
        mx.eval(q, k, v, a, b, g, beta, state)

        # Sequential Metal kernel
        bench(
            gated_delta_kernel,
            q, k, v, a, b, A_log, dt_bias, state,
            label="sequential kernel",
        )

        # Chunkwise with various chunk sizes
        for cs in [32, 64, 128]:
            if cs <= T:
                bench(
                    gated_delta_chunkwise,
                    q, k, v, g, beta, state, None, cs,
                    label=f"chunkwise C={cs}",
                )

        print()


if __name__ == "__main__":
    main()
