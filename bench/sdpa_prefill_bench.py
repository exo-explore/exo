"""SDPA prefill microbenchmark for Qwen3.5-397B attention config.

Measures dequantize and SDPA time separately at various context lengths
to identify where optimization effort should go.

Usage: uv run python bench/sdpa_prefill_bench.py
"""

import os
import time

import mlx.core as mx

# Qwen3.5-397B attention config (per node with PP=2, 30 layers, 8 attention layers)
HEAD_DIM = 256
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
BATCH = 1
QUERY_LEN = 2048  # prefill chunk size per node
KV_BITS = 4
KV_GROUP_SIZE = 64

# Context lengths to test
CONTEXT_LENGTHS = [2048, 8192, 16384, 32768, 65536, 131072, 196608, 262144]

WARMUP = 2
ITERS = 5


def make_quantized_kv(kv_len: int):
    """Create quantized KV cache tensors matching Qwen3.5 config."""
    el_per_int = 8 * mx.uint32.size // KV_BITS
    shape_data = (BATCH, NUM_KV_HEADS, kv_len, HEAD_DIM // el_per_int)
    shape_scales = (BATCH, NUM_KV_HEADS, kv_len, HEAD_DIM // KV_GROUP_SIZE)

    k_data = mx.zeros(shape_data, dtype=mx.uint32)
    k_scales = mx.zeros(shape_scales, dtype=mx.float16)
    k_biases = mx.zeros(shape_scales, dtype=mx.float16)
    v_data = mx.zeros(shape_data, dtype=mx.uint32)
    v_scales = mx.zeros(shape_scales, dtype=mx.float16)
    v_biases = mx.zeros(shape_scales, dtype=mx.float16)

    # Fill with random data so compute isn't trivially optimized away
    k_data = mx.random.randint(0, 255, shape=shape_data).astype(mx.uint32)
    k_scales = mx.random.normal(shape=shape_scales).astype(mx.float16)
    k_biases = mx.random.normal(shape=shape_scales).astype(mx.float16)
    v_data = mx.random.randint(0, 255, shape=shape_data).astype(mx.uint32)
    v_scales = mx.random.normal(shape=shape_scales).astype(mx.float16)
    v_biases = mx.random.normal(shape=shape_scales).astype(mx.float16)

    mx.eval(k_data, k_scales, k_biases, v_data, v_scales, v_biases)
    return (k_data, k_scales, k_biases), (v_data, v_scales, v_biases)


def bench_dequantize(q_keys, q_values, iters: int) -> float:
    """Time just the dequantize step."""
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        dk = mx.dequantize(*q_keys, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        dv = mx.dequantize(*q_values, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        mx.eval(dk, dv)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_sdpa(queries, keys, values, iters: int) -> float:
    """Time just the fused SDPA (with already-dequantized KV)."""
    scale = HEAD_DIM**-0.5
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask="causal"
        )
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_combined(queries, q_keys, q_values, iters: int) -> float:
    """Time dequantize + SDPA together (the actual code path)."""
    scale = HEAD_DIM**-0.5
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        dk = mx.dequantize(*q_keys, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        dv = mx.dequantize(*q_values, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        out = mx.fast.scaled_dot_product_attention(
            queries, dk, dv, scale=scale, mask="causal"
        )
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main():
    threshold = os.environ.get("MLX_SDPA_FUSED_THRESHOLD", "16384")
    chunk_threshold = os.environ.get("MLX_SDPA_CHUNK_THRESHOLD", "65536")
    print(f"MLX_SDPA_FUSED_THRESHOLD={threshold}")
    print(f"MLX_SDPA_CHUNK_THRESHOLD={chunk_threshold}")
    print(f"Config: head_dim={HEAD_DIM}, q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, "
          f"kv_bits={KV_BITS}, query_len={QUERY_LEN}")
    print()
    print(f"{'kL':>8} | {'deq (ms)':>10} | {'sdpa (ms)':>10} | {'total (ms)':>10} | "
          f"{'deq %':>7} | {'KV MB':>8} | {'deq MB':>8}")
    print("-" * 85)

    for kv_len in CONTEXT_LENGTHS:
        # Check if we have enough memory (rough estimate)
        kv_bytes_fp16 = BATCH * NUM_KV_HEADS * kv_len * HEAD_DIM * 2 * 2  # K + V, fp16
        kv_mb_fp16 = kv_bytes_fp16 / 1e6

        kv_bytes_quant = BATCH * NUM_KV_HEADS * kv_len * HEAD_DIM * KV_BITS / 8 * 2
        kv_mb_quant = kv_bytes_quant / 1e6

        # Create data
        q_keys, q_values = make_quantized_kv(kv_len)
        queries = mx.random.normal(shape=(BATCH, NUM_Q_HEADS, QUERY_LEN, HEAD_DIM)).astype(mx.float16)
        mx.eval(queries)

        # Pre-dequantize for isolated SDPA bench
        dk = mx.dequantize(*q_keys, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        dv = mx.dequantize(*q_values, group_size=KV_GROUP_SIZE, bits=KV_BITS)
        mx.eval(dk, dv)

        # Warmup
        bench_dequantize(q_keys, q_values, WARMUP)
        bench_sdpa(queries, dk, dv, WARMUP)
        bench_combined(queries, q_keys, q_values, WARMUP)

        # Measure
        t_deq = bench_dequantize(q_keys, q_values, ITERS)
        t_sdpa = bench_sdpa(queries, dk, dv, ITERS)
        t_combined = bench_combined(queries, q_keys, q_values, ITERS)

        deq_pct = (t_deq / t_combined * 100) if t_combined > 0 else 0

        print(f"{kv_len:>8} | {t_deq*1000:>10.2f} | {t_sdpa*1000:>10.2f} | "
              f"{t_combined*1000:>10.2f} | {deq_pct:>6.1f}% | {kv_mb_quant:>8.1f} | {kv_mb_fp16:>8.1f}")

        # Free memory
        del q_keys, q_values, queries, dk, dv
        mx.clear_cache()


if __name__ == "__main__":
    main()
