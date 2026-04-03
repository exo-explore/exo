"""Profile SDPA at different GQA ratios to isolate the GQA cache-miss overhead.

If GQA data reuse (L2 cache misses on shared K/V) is the bottleneck,
then reducing the GQA ratio should show disproportionate speedup.

Also profiles with different numbers of KV heads to isolate
per-head vs cross-head effects.

Usage: uv run python bench/sdpa_gqa_profile.py
"""

import time

import mlx.core as mx

HEAD_DIM = 256
BATCH = 1
QUERY_LEN = 2048
KV_BITS = 4
KV_GROUP_SIZE = 64

WARMUP = 2
ITERS = 5


def bench_sdpa_config(num_q_heads: int, num_kv_heads: int, kv_len: int) -> float:
    """Benchmark SDPA for a specific head configuration."""
    gqa = num_q_heads // num_kv_heads

    # Create float16 Q, K, V (skip dequantize since it's <0.2%)
    queries = mx.random.normal(shape=(BATCH, num_q_heads, QUERY_LEN, HEAD_DIM)).astype(mx.float16)
    keys = mx.random.normal(shape=(BATCH, num_kv_heads, kv_len, HEAD_DIM)).astype(mx.float16)
    values = mx.random.normal(shape=(BATCH, num_kv_heads, kv_len, HEAD_DIM)).astype(mx.float16)
    mx.eval(queries, keys, values)

    scale = HEAD_DIM**-0.5

    # Warmup
    for _ in range(WARMUP):
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask="causal")
        mx.eval(out)
        mx.synchronize()

    # Measure
    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask="causal")
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)

    del queries, keys, values
    mx.clear_cache()
    return sum(times) / len(times)


def main():
    print("=== Part 1: GQA ratio sweep (fixed total Q heads=32) ===")
    print(f"Config: head_dim={HEAD_DIM}, query_len={QUERY_LEN}")
    print()
    print(f"{'q_heads':>8} | {'kv_heads':>9} | {'GQA':>4} | {'kL=32K (ms)':>12} | {'kL=131K (ms)':>13} | {'kL=262K (ms)':>13} | {'ms/head 262K':>12}")
    print("-" * 100)

    configs = [
        (32, 2),   # Qwen3.5 actual: GQA 16:1
        (32, 4),   # GQA 8:1
        (32, 8),   # GQA 4:1
        (32, 16),  # GQA 2:1
        (32, 32),  # MHA (no GQA)
    ]

    for q_heads, kv_heads in configs:
        gqa = q_heads // kv_heads
        t32k = bench_sdpa_config(q_heads, kv_heads, 32768)
        t131k = bench_sdpa_config(q_heads, kv_heads, 131072)
        t262k = bench_sdpa_config(q_heads, kv_heads, 262144)
        ms_per_head = t262k * 1000 / q_heads
        print(f"{q_heads:>8} | {kv_heads:>9} | {gqa:>4} | {t32k*1000:>12.1f} | {t131k*1000:>13.1f} | {t262k*1000:>13.1f} | {ms_per_head:>12.2f}")

    print()
    print("=== Part 2: K/V size in L2 check ===")
    print("If GQA L2 misses are the issue, KV that fits in L2 should be proportionally faster")
    print()
    # KV cache per KV head: kv_len * 256 * 2 bytes
    # L2 ~ 48 MB. With 2 KV heads: 48 MB / 2 / 256 / 2 = ~49K tokens fit in L2
    # With 32 KV heads: 48 MB / 32 / 256 / 2 = ~3K tokens fit in L2
    # So at GQA 16:1 with 2 KV heads: K fits in L2 up to ~49K context

    print(f"{'kv_len':>8} | {'KV MB (2 heads)':>15} | {'fits L2?':>9} | {'time (ms)':>10} | {'ms/1K kL':>9}")
    print("-" * 70)

    for kv_len in [8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144]:
        kv_mb = 2 * kv_len * 256 * 2 / 1e6  # 2 KV heads, FP16
        fits = "yes" if kv_mb < 48 else "no"
        t = bench_sdpa_config(32, 2, kv_len)
        ms_per_1k = t * 1000 / (kv_len / 1024)
        print(f"{kv_len:>8} | {kv_mb:>15.1f} | {fits:>9} | {t*1000:>10.1f} | {ms_per_1k:>9.3f}")


if __name__ == "__main__":
    main()
