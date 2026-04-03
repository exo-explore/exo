"""Decomposed SDPA: standalone matmuls + online softmax.

Tests whether decomposing FlashAttention into separate matmul calls
achieves better throughput by leveraging the optimized matmul kernel.

Usage: uv run python bench/sdpa_decomposed_bench.py
"""

import time

import mlx.core as mx

HEAD_DIM = 256
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
GQA_FACTOR = NUM_Q_HEADS // NUM_KV_HEADS
BATCH = 1
QUERY_LEN = 2048

WARMUP = 2
ITERS = 5
NEG_INF = -1e9


def decomposed_sdpa(queries, keys, values, scale, chunk_size):
    """SDPA via standalone matmuls + online softmax."""
    B, Hq, qL, D = queries.shape
    Hkv = keys.shape[1]
    kL = keys.shape[2]
    gqa = Hq // Hkv

    # Expand K/V for GQA: [B, Hkv, kL, D] -> [B, Hq, kL, D]
    if gqa > 1:
        keys = mx.repeat(keys, gqa, axis=1)
        values = mx.repeat(values, gqa, axis=1)

    # Scale queries once
    queries = queries * scale

    # Online softmax state
    O = mx.zeros((B, Hq, qL, D), dtype=queries.dtype)
    max_scores = mx.full((B, Hq, qL, 1), NEG_INF, dtype=mx.float32)
    sum_exp = mx.zeros((B, Hq, qL, 1), dtype=mx.float32)

    for k_start in range(0, kL, chunk_size):
        k_end = min(k_start + chunk_size, kL)
        K_chunk = keys[:, :, k_start:k_end, :]  # [B, Hq, cs, D]
        V_chunk = values[:, :, k_start:k_end, :]

        # S = Q @ K^T: [B, Hq, qL, D] @ [B, Hq, D, cs] -> [B, Hq, qL, cs]
        S = queries @ K_chunk.transpose(0, 1, 3, 2)
        S = S.astype(mx.float32)

        # Causal mask
        q_indices = mx.arange(kL - qL, kL).reshape(1, 1, qL, 1)
        k_indices = mx.arange(k_start, k_end).reshape(1, 1, 1, -1)
        causal = q_indices >= k_indices
        S = mx.where(causal, S, NEG_INF)

        # Online softmax update
        chunk_max = S.max(axis=-1, keepdims=True)
        new_max = mx.maximum(max_scores, chunk_max)

        # Correction factor for previous accumulations
        correction = mx.exp(max_scores - new_max)

        # P = exp(S - new_max) for this chunk
        P = mx.exp(S - new_max)

        # Update running sum
        sum_exp = sum_exp * correction + P.sum(axis=-1, keepdims=True)

        # Update O
        P = P.astype(queries.dtype)
        O = O * correction.astype(queries.dtype) + (P @ V_chunk)

        max_scores = new_max

    O = O / sum_exp.astype(queries.dtype)
    return O


def fused_sdpa(queries, keys, values, scale):
    """Standard fused SDPA via mx.fast."""
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask="causal"
    )


def bench(fn, *args, name=""):
    # Warmup
    for _ in range(WARMUP):
        out = fn(*args)
        mx.eval(out)
        mx.synchronize()

    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return avg


def main():
    scale = HEAD_DIM**-0.5
    print(f"Config: head_dim={HEAD_DIM}, q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, qL={QUERY_LEN}")
    print()

    for kv_len in [32768, 65536, 131072, 262144]:
        queries = mx.random.normal(shape=(BATCH, NUM_Q_HEADS, QUERY_LEN, HEAD_DIM)).astype(mx.float16)
        keys = mx.random.normal(shape=(BATCH, NUM_KV_HEADS, kv_len, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal(shape=(BATCH, NUM_KV_HEADS, kv_len, HEAD_DIM)).astype(mx.float16)
        mx.eval(queries, keys, values)

        # Fused baseline
        t_fused = bench(fused_sdpa, queries, keys, values, scale, name="fused") * 1000

        # Decomposed with different chunk sizes
        results = [f"kL={kv_len//1024}K: fused={t_fused:.0f}ms"]

        for cs in [4096, 8192, 16384, 32768]:
            if cs > kv_len:
                continue
            t_decomp = bench(decomposed_sdpa, queries, keys, values, scale, cs, name=f"cs={cs}") * 1000
            speedup = t_fused / t_decomp
            results.append(f"cs={cs}: {t_decomp:.0f}ms ({speedup:.2f}x)")

        print("  ".join(results))

        del queries, keys, values
        mx.clear_cache()


if __name__ == "__main__":
    main()
