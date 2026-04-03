"""Decomposed SDPA v2: per-KV-head processing, no GQA expansion.

Avoids the 16x memory expansion by processing each KV head with its
16 query heads using a single matmul with reshaped tensors.

Usage: uv run python bench/sdpa_decomposed_v2_bench.py
"""

import time

import mlx.core as mx

HEAD_DIM = 256
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
GQA_FACTOR = NUM_Q_HEADS // NUM_KV_HEADS  # 16
BATCH = 1
QUERY_LEN = 2048

WARMUP = 2
ITERS = 5
NEG_INF = -1e9


def decomposed_sdpa_v2(queries, keys, values, scale, chunk_size):
    """SDPA via per-KV-head matmuls, no GQA expansion."""
    B, Hq, qL, D = queries.shape
    Hkv = keys.shape[1]
    kL = keys.shape[2]
    gqa = Hq // Hkv

    queries = queries * scale

    # Reshape Q to group by KV head: [B, Hkv, gqa, qL, D]
    Q_grouped = queries.reshape(B, Hkv, gqa, qL, D)

    O = mx.zeros((B, Hkv, gqa, qL, D), dtype=queries.dtype)
    max_scores = mx.full((B, Hkv, gqa, qL, 1), NEG_INF, dtype=mx.float32)
    sum_exp = mx.zeros((B, Hkv, gqa, qL, 1), dtype=mx.float32)

    for k_start in range(0, kL, chunk_size):
        k_end = min(k_start + chunk_size, kL)
        cs = k_end - k_start
        # K_chunk: [B, Hkv, cs, D] -> [B, Hkv, 1, D, cs] for broadcast with gqa dim
        K_chunk = keys[:, :, k_start:k_end, :].transpose(0, 1, 3, 2)  # [B, Hkv, D, cs]
        K_chunk = K_chunk[:, :, None, :, :]  # [B, Hkv, 1, D, cs]

        V_chunk = values[:, :, k_start:k_end, :]  # [B, Hkv, cs, D]
        V_chunk = V_chunk[:, :, None, :, :]  # [B, Hkv, 1, cs, D]

        # S = Q_grouped @ K_chunk: [B, Hkv, gqa, qL, D] @ [B, Hkv, 1, D, cs]
        # -> [B, Hkv, gqa, qL, cs]  (broadcasts gqa dim)
        S = (Q_grouped @ K_chunk).astype(mx.float32)

        # Causal mask
        q_indices = mx.arange(kL - qL, kL).reshape(1, 1, 1, qL, 1)
        k_indices = mx.arange(k_start, k_end).reshape(1, 1, 1, 1, cs)
        S = mx.where(q_indices >= k_indices, S, NEG_INF)

        # Online softmax
        chunk_max = S.max(axis=-1, keepdims=True)
        new_max = mx.maximum(max_scores, chunk_max)
        correction = mx.exp(max_scores - new_max)
        P = mx.exp(S - new_max)

        sum_exp = sum_exp * correction + P.sum(axis=-1, keepdims=True)

        P = P.astype(queries.dtype)
        # P @ V: [B, Hkv, gqa, qL, cs] @ [B, Hkv, 1, cs, D] -> [B, Hkv, gqa, qL, D]
        O = O * correction.astype(queries.dtype) + (P @ V_chunk)
        max_scores = new_max

    O = O / sum_exp.astype(queries.dtype)
    return O.reshape(B, Hq, qL, D)


def fused_sdpa(queries, keys, values, scale):
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask="causal"
    )


def bench(fn, *args):
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
    return sum(times) / len(times)


def main():
    scale = HEAD_DIM**-0.5
    print(f"Config: head_dim={HEAD_DIM}, q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, qL={QUERY_LEN}")
    print()

    for kv_len in [32768, 131072, 262144]:
        queries = mx.random.normal(shape=(BATCH, NUM_Q_HEADS, QUERY_LEN, HEAD_DIM)).astype(mx.float16)
        keys = mx.random.normal(shape=(BATCH, NUM_KV_HEADS, kv_len, HEAD_DIM)).astype(mx.float16)
        values = mx.random.normal(shape=(BATCH, NUM_KV_HEADS, kv_len, HEAD_DIM)).astype(mx.float16)
        mx.eval(queries, keys, values)

        t_fused = bench(fused_sdpa, queries, keys, values, scale) * 1000

        results = [f"kL={kv_len//1024}K: fused={t_fused:.0f}ms"]
        for cs in [4096, 16384]:
            if cs > kv_len:
                continue
            t = bench(decomposed_sdpa_v2, queries, keys, values, scale, cs) * 1000
            results.append(f"v2_cs={cs}: {t:.0f}ms ({t_fused/t:.2f}x)")

        print("  ".join(results))
        del queries, keys, values
        mx.clear_cache()


if __name__ == "__main__":
    main()
