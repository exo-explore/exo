# TurboQuant KV Cache Integration Guide

Implementation spec for integrating TurboQuant KV cache compression into exo's MLX inference pipeline. This document is intended for Claude Code to use during implementation.

## Motivation

On 2x Mac Studio M4 Max (128GB each) running Qwen3.5-397B-A17B-4bit via pipeline parallelism over TB5 RDMA:

- Current decode: ~31-36 tok/s (target: consistent 40 tok/s)
- Current prefill TTFT: 1.5-2.6s
- KV cache dequantization accounts for ~40% of prefill attention time (measured in `bench/sdpa_prefill_bench.py`)
- Current quantization: `QuantizedKVCache` with 4-bit affine quantization (`mx.quantize`/`mx.dequantize`)

TurboQuant replaces the dequantize-then-SDPA path with an approximate dot-product estimation that avoids full dequantization, reducing compute during both prefill (chunked attention over long KV cache) and decode (per-token attention over growing cache).

## TurboQuant Algorithm Summary

TurboQuant uses two-stage compression:

### Stage 1: PolarQuant (MSE-optimal)
1. Precondition input vectors with orthogonal rotation matrix R (QR decomposition, generated once at init)
2. Convert rotated Cartesian coordinates to polar via recursive dimension pairing (arctan2)
3. Quantize angles to `bits` precision (default 3-bit, range mapping to [0, 2^bits - 1])
4. Store: quantized angles (int8) + radius (float16/32)
5. Decompress: reverse polar-to-Cartesian + inverse rotation (R^T)

### Stage 2: QJL Residual Correction (Quantized Johnson-Lindenstrauss)
1. Compute residual: `residual = x - pq_decompress(pq_compress(x))`
2. Project residual onto random Gaussian matrix R: `projection = residual @ R`
3. Sign-quantize: store only signs (±1) → 1 bit per projection dimension
4. Store: sign matrix (binary) + L2 norms (float16)

### Dot Product Estimation (Attention)
Instead of dequantize → matmul, TurboQuant estimates the dot product directly:
```
dot = pq_decompress(pq_data) @ query + qjl_estimate_dot(qjl_signs, qjl_norms, query)
```
The QJL correction makes this an **unbiased** estimator of the true dot product, scaled by `(norm / num_features) * sqrt(π/2)`.

### Asymmetric Compression
- **Keys**: Full TurboQuant (PolarQuant + QJL residual) — needs accurate dot products for attention scores
- **Values**: PolarQuant only (lighter) — weighted sum is more tolerant of approximation

### Attention Sink
First `fp16_sink_size` tokens (default 128) stored uncompressed in float16. Preserves system prompt instruction-following capability. Critical for chat models.

### Dynamic Chunking
Tokens are buffered and compressed in chunks of 64 tokens. This amortizes the compression overhead and ensures the orthogonal rotation operates on reasonably-sized batches.

## Current exo KV Cache Architecture

### Files and Types

| File | Role |
|------|------|
| `src/exo/shared/types/mlx.py` | `KVCacheType = Sequence[KVCache \| RotatingKVCache \| QuantizedKVCache \| ArraysCache \| CacheList]` |
| `src/exo/worker/engines/mlx/cache.py` | `KVPrefixCache` (LRU prefix cache), `make_kv_cache()`, `trim_cache()`, `snapshot_ssm_states()` |
| `src/exo/worker/engines/mlx/constants.py` | `KV_CACHE_BITS`, `CACHE_GROUP_SIZE` (controls quantized cache creation) |
| `src/exo/worker/engines/mlx/generator/generate.py` | `prefill()`, `pipeline_parallel_prefill()`, `mlx_generate()` |
| `mlx-lm/mlx_lm/models/cache.py` | `KVCache`, `QuantizedKVCache`, `RotatingKVCache` class definitions |
| `mlx-lm/mlx_lm/generate.py` | `maybe_quantize_kv_cache()`, `stream_generate()`, `generate_step()` |

### How QuantizedKVCache Currently Works

`QuantizedKVCache` uses MLX's built-in affine quantization:

**Storage** (per layer): 3-tuple for keys, 3-tuple for values:
- `(data: uint32, scales: float16, biases: float16)` — packed ints + per-group scale/bias
- Shape: `(B, n_kv_heads, seq_len, dim // el_per_int)` for data, `(B, n_kv_heads, seq_len, dim // group_size)` for scales/biases

**`update_and_fetch(keys, values)`**:
1. Pre-allocates in steps of `self.step` (default 16384 in exo, set in `make_kv_cache()`)
2. Calls `mx.quantize(keys, group_size, bits)` → returns (data, scales, biases)
3. Writes into pre-allocated buffers at offset
4. Returns the full quantized tuple up to current offset

**Dequantization**: Happens implicitly inside `mx.fast.scaled_dot_product_attention()` — MLX's SDPA kernel accepts quantized KV tuples directly. The dequantization cost shows up inside the SDPA kernel, not as a separate step.

**`trim(n)`**: Simply decrements `self.offset` by n (lazy trim, data stays in buffer).

**`state` property**: Returns `(self.keys, self.values)` tuples (sliced to offset if needed). Used by `KVPrefixCache` for deepcopy/snapshot.

### Cache Creation Flow

In `make_kv_cache()` (`cache.py:374-408`):
1. Calls `model.make_cache()` to get default cache list
2. If `KV_CACHE_BITS` is set (via `EXO_KV_CACHE_BITS` env var):
   - Iterates cache entries
   - Replaces `KVCache` instances with `QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)`
   - Sets `qc.step = 16384` (large pre-allocation to reduce fragmentation)
   - Leaves `ArraysCache` (DeltaNet/SSM) and other types unchanged
3. If `KV_CACHE_BITS` is not set: uses standard `KVCache` with `step=16384`

### Prefill Quantization Flow

In `pipeline_parallel_prefill()` (`generate.py:119-262`):
1. Creates `quantize_cache_fn = functools.partial(maybe_quantize_kv_cache, ...)` with `kv_group_size` and `kv_bits` from constants
2. After each chunk's forward pass: calls `quantize_cache_fn(_prompt_cache)`
3. `maybe_quantize_kv_cache()` (`mlx-lm/generate.py:298-303`): iterates cache entries, calls `c.to_quantized()` on any `KVCache` that has the method
4. This converts FP16 KVCache → QuantizedKVCache mid-prefill

**Important**: When using `QuantizedKVCache` directly (via `make_kv_cache()` with `KV_CACHE_BITS`), `maybe_quantize_kv_cache` is a no-op because the cache is already quantized — quantization happens inside `update_and_fetch()`.

### Decode Flow

During decode (`stream_generate()` in mlx-lm), each token:
1. Model forward pass calls `cache[layer].update_and_fetch(new_keys, new_values)`
2. For `QuantizedKVCache`: new token's K/V are quantized and appended
3. Full quantized K/V returned to attention layer
4. `mx.fast.scaled_dot_product_attention()` handles dequantization internally

### KVPrefixCache Interaction

`KVPrefixCache` (`cache.py:83-305`):
- Stores `deepcopy(cache)` for each completed generation
- On cache hit: `deepcopy` the matched cache, trim to match point, return remaining tokens
- `trim_cache()`: calls `c.trim(num_tokens)` on KVCache/QuantizedKVCache, or restores snapshots for ArraysCache
- `snapshot_ssm_states()`: only snapshots ArraysCache entries (SSM state), skips KV caches

## Integration Plan

### Approach: New `TurboQuantKVCache` Class

Create a new cache class that implements the same interface as `QuantizedKVCache` but uses TurboQuant internally. This is cleaner than monkey-patching and fits exo's architecture.

### Required Interface

The new class must implement:

```python
class TurboQuantKVCache:
    offset: int          # Current number of tokens in cache
    step: int            # Pre-allocation step size

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple:
        """Store new K/V and return full cache for attention.

        Args:
            keys: (B, n_kv_heads, num_steps, k_head_dim)
            values: (B, n_kv_heads, num_steps, v_head_dim)

        Returns:
            (keys, values) — but format depends on attention integration
        """
        ...

    @property
    def state(self) -> tuple:
        """Return serializable state for KVPrefixCache deepcopy."""
        ...

    @state.setter
    def state(self, v: tuple) -> None:
        ...

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Trim n tokens from end of cache."""
        ...
```

### Critical Design Decision: Attention Integration

**The hard problem**: MLX's `mx.fast.scaled_dot_product_attention()` accepts either:
1. Full FP16 K/V tensors
2. Quantized tuples `(data, scales, biases)` — but only affine quantization

TurboQuant's compression format (PolarQuant angles + QJL signs) is **not compatible** with MLX's native SDPA kernel. There are two paths:

#### Option A: Decompress-then-SDPA (Simpler, Less Optimal)

`update_and_fetch()` stores compressed, but returns **decompressed FP16** tensors for SDPA:

```python
def update_and_fetch(self, keys, values):
    # Compress and store new tokens
    self._compress_and_store(keys, values)
    # Decompress everything for attention
    full_keys = self._decompress_all_keys()    # → (B, heads, seq, dim) fp16
    full_values = self._decompress_all_values() # → (B, heads, seq, dim) fp16
    return full_keys, full_values
```

**Pros**: Drop-in compatible with all attention implementations. No SDPA kernel changes needed.
**Cons**: Decompression cost on every token. Memory spike during decompress. May not beat QuantizedKVCache on speed.
**When this wins**: Memory savings — TurboQuant at 3-bit achieves 5.3x compression vs FP16 (vs ~4x for 4-bit affine). Allows larger KV caches in memory. Decompress cost may be less than affine dequantize because PolarQuant decompress is mostly trig ops (sin/cos) which are fast on Metal, vs the per-group scale+bias multiply in affine.

#### Option B: Custom Attention Kernel with Approximate Dot Product (Optimal, Complex)

Write a custom Metal kernel or MLX operation that computes attention scores directly from TurboQuant compressed format using `estimate_dot()`:

```python
# Pseudocode for compressed attention
scores = turboquant_estimate_dot(compressed_keys, queries)  # (B, heads, Q, KV_len)
scores = softmax(scores / sqrt(d_k))
output = polarquant_decompress_and_weight(compressed_values, scores)  # values only need PolarQuant decompress
```

**Pros**: Avoids full decompression. Potentially much faster for long contexts. The whole point of TurboQuant.
**Cons**: Requires custom Metal kernel or careful MLX op composition. QJL dot product estimation adds computation. Needs thorough numerical validation.

#### Recommended Approach: Option A First, Then B

1. Implement Option A to validate correctness and measure memory savings
2. Profile to understand if decompression is actually the bottleneck vs. the SDPA compute itself
3. If decompression dominates, implement Option B as a custom MLX kernel

### File Changes

#### New Files

**`src/exo/worker/engines/mlx/turboquant_cache.py`**

Core implementation. Contains:

```python
import mlx.core as mx
import numpy as np
import math

class PolarQuantCompressor:
    """MLX implementation of PolarQuant.

    Preconditions vectors with orthogonal rotation, converts to polar coordinates,
    quantizes angles to target bit depth.
    """
    def __init__(self, feature_dim: int, bits: int = 3, seed: int = 42):
        # Generate orthogonal rotation matrix R via QR decomposition
        # feature_dim must be power of 2
        rng = np.random.RandomState(seed)
        H = rng.randn(feature_dim, feature_dim).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        self.R = mx.array(Q)  # (feature_dim, feature_dim)
        self.bits = bits
        self.feature_dim = feature_dim

    def compress(self, x: mx.array) -> dict:
        """Compress vectors via polar quantization.

        Args:
            x: (..., feature_dim) input vectors
        Returns:
            dict with 'angles' (list of quantized angle arrays) and 'radius'
        """
        # 1. Rotate: x_rot = x @ R
        # 2. Recursive Cartesian-to-polar (pair even/odd dims, compute arctan2)
        # 3. Quantize angles to bits
        ...

    def decompress(self, compressed: dict) -> mx.array:
        """Reconstruct approximate vectors from compressed representation."""
        # 1. Dequantize angles
        # 2. Recursive polar-to-Cartesian
        # 3. Inverse rotation: x_approx = x_polar @ R.T
        ...


class QJLCompressor:
    """Quantized Johnson-Lindenstrauss for residual compression.

    Projects residuals onto random Gaussian matrix, stores only signs.
    Enables unbiased dot-product estimation.
    """
    def __init__(self, feature_dim: int, num_projections: int = 2048, seed: int = 43):
        key = mx.random.key(seed)
        self.R = mx.random.normal(key=key, shape=(feature_dim, num_projections))
        self.num_projections = num_projections
        self._scale = math.sqrt(math.pi / 2)

    def compress(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compress via random projection + sign quantization.

        Returns:
            (signs: mx.array of {-1, +1}, norms: mx.array of L2 norms)
        """
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)
        projection = x @ self.R  # (..., num_projections)
        signs = mx.sign(projection)
        signs = mx.where(signs == 0, 1.0, signs)
        return signs, norms

    def estimate_dot(self, signs: mx.array, norms: mx.array, query: mx.array) -> mx.array:
        """Estimate dot product between compressed vectors and query.

        Returns unbiased estimate of x @ query.
        """
        query_proj = query @ self.R  # (..., num_projections)
        dot = signs @ query_proj  # dot product in projection space
        return (norms / self.num_projections) * self._scale * dot


class TurboQuantKVCache:
    """KV cache using TurboQuant compression.

    Compatible with exo's cache interface (offset, state, update_and_fetch, trim).
    Uses PolarQuant for values, TurboQuant (PolarQuant + QJL) for keys.

    Config:
        head_dim: Attention head dimension (must be power of 2)
        bits: PolarQuant angle bit depth (default 3)
        qjl_projections: Number of QJL random projections (default 2048, or head_dim * 8)
        fp16_sink_size: Number of initial tokens to keep uncompressed (default 128)
        chunk_size: Tokens to buffer before compressing (default 64)
    """
    step = 16384  # Match exo's pre-allocation step

    def __init__(
        self,
        head_dim: int,
        bits: int = 3,
        qjl_projections: int | None = None,
        fp16_sink_size: int = 128,
        chunk_size: int = 64,
        seed: int = 42,
    ):
        self.offset = 0
        self.head_dim = head_dim
        self.bits = bits
        self.fp16_sink_size = fp16_sink_size
        self.chunk_size = chunk_size

        # Compressors (initialized lazily or here if head_dim known)
        self.key_pq = PolarQuantCompressor(head_dim, bits=bits, seed=seed)
        self.key_qjl = QJLCompressor(head_dim, num_projections=qjl_projections or head_dim * 8, seed=seed + 1)
        self.value_pq = PolarQuantCompressor(head_dim, bits=bits, seed=seed + 2)

        # Uncompressed sink buffers (first fp16_sink_size tokens)
        self.sink_keys: mx.array | None = None      # (B, heads, sink_len, dim)
        self.sink_values: mx.array | None = None
        self.sink_offset = 0

        # Compressed chunk storage
        self.compressed_key_chunks: list[dict] = []   # PolarQuant + QJL data per chunk
        self.compressed_value_chunks: list[dict] = []  # PolarQuant data per chunk

        # Buffer for tokens not yet forming a full chunk
        self.key_buffer: mx.array | None = None       # (B, heads, buffer_len, dim)
        self.value_buffer: mx.array | None = None
        self.buffer_len = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new K/V tokens and return full decompressed cache.

        Input shapes: (B, n_kv_heads, num_steps, head_dim)
        Returns: (full_keys, full_values) as fp16 tensors
        """
        B, n_kv_heads, num_steps, dim = keys.shape
        self.offset += num_steps

        # Route tokens to sink, buffer, or compression
        pos = self.offset - num_steps  # position of first new token

        for t in range(num_steps):
            token_pos = pos + t
            k_token = keys[:, :, t:t+1, :]  # (B, heads, 1, dim)
            v_token = values[:, :, t:t+1, :]

            if token_pos < self.fp16_sink_size:
                # Store in sink (uncompressed)
                self._append_to_sink(k_token, v_token)
            else:
                # Add to buffer
                self._append_to_buffer(k_token, v_token)

                # If buffer is full, compress it
                if self.buffer_len >= self.chunk_size:
                    self._compress_buffer()

        # Decompress and return full cache (Option A)
        return self._decompress_all()

    def _append_to_sink(self, k: mx.array, v: mx.array):
        if self.sink_keys is None:
            self.sink_keys = k
            self.sink_values = v
        else:
            self.sink_keys = mx.concatenate([self.sink_keys, k], axis=2)
            self.sink_values = mx.concatenate([self.sink_values, v], axis=2)
        self.sink_offset += 1

    def _append_to_buffer(self, k: mx.array, v: mx.array):
        if self.key_buffer is None:
            self.key_buffer = k
            self.value_buffer = v
        else:
            self.key_buffer = mx.concatenate([self.key_buffer, k], axis=2)
            self.value_buffer = mx.concatenate([self.value_buffer, v], axis=2)
        self.buffer_len += 1

    def _compress_buffer(self):
        """Compress buffered tokens into a chunk."""
        assert self.key_buffer is not None and self.value_buffer is not None
        B, heads, seq, dim = self.key_buffer.shape

        # Reshape to (B*heads*seq, dim) for compression
        k_flat = self.key_buffer.reshape(-1, dim)
        v_flat = self.value_buffer.reshape(-1, dim)

        # Keys: PolarQuant + QJL
        k_pq = self.key_pq.compress(k_flat)
        k_approx = self.key_pq.decompress(k_pq)
        k_residual = k_flat - k_approx
        k_qjl_signs, k_qjl_norms = self.key_qjl.compress(k_residual)

        self.compressed_key_chunks.append({
            'pq': k_pq,
            'qjl_signs': k_qjl_signs,
            'qjl_norms': k_qjl_norms,
            'shape': (B, heads, seq, dim),
        })

        # Values: PolarQuant only
        v_pq = self.value_pq.compress(v_flat)
        self.compressed_value_chunks.append({
            'pq': v_pq,
            'shape': (B, heads, seq, dim),
        })

        # Clear buffer
        self.key_buffer = None
        self.value_buffer = None
        self.buffer_len = 0

    def _decompress_all(self) -> tuple[mx.array, mx.array]:
        """Decompress all stored data and return full K/V tensors."""
        parts_k = []
        parts_v = []

        # 1. Sink (uncompressed)
        if self.sink_keys is not None:
            parts_k.append(self.sink_keys)
            parts_v.append(self.sink_values)

        # 2. Compressed chunks
        for chunk in self.compressed_key_chunks:
            shape = chunk['shape']
            k_approx = self.key_pq.decompress(chunk['pq'])
            parts_k.append(k_approx.reshape(shape))

        for chunk in self.compressed_value_chunks:
            shape = chunk['shape']
            v_approx = self.value_pq.decompress(chunk['pq'])
            parts_v.append(v_approx.reshape(shape))

        # 3. Buffer (uncompressed, not yet a full chunk)
        if self.key_buffer is not None:
            parts_k.append(self.key_buffer)
            parts_v.append(self.value_buffer)

        if not parts_k:
            return None, None

        full_keys = mx.concatenate(parts_k, axis=2)
        full_values = mx.concatenate(parts_v, axis=2)
        return full_keys, full_values

    @property
    def state(self):
        """Return state for KVPrefixCache serialization."""
        # Return enough to reconstruct the cache
        return (
            self.sink_keys, self.sink_values,
            self.compressed_key_chunks, self.compressed_value_chunks,
            self.key_buffer, self.value_buffer,
            self.sink_offset, self.buffer_len,
        )

    @state.setter
    def state(self, v):
        (
            self.sink_keys, self.sink_values,
            self.compressed_key_chunks, self.compressed_value_chunks,
            self.key_buffer, self.value_buffer,
            self.sink_offset, self.buffer_len,
        ) = v

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        """Trim n tokens from end of cache.

        Strategy: trim from buffer first, then from compressed chunks.
        """
        n = min(self.offset, n)
        remaining = n

        # Trim buffer first
        if self.key_buffer is not None and remaining > 0:
            trim_from_buffer = min(self.buffer_len, remaining)
            if trim_from_buffer >= self.buffer_len:
                self.key_buffer = None
                self.value_buffer = None
                self.buffer_len = 0
            else:
                self.key_buffer = self.key_buffer[:, :, :-trim_from_buffer, :]
                self.value_buffer = self.value_buffer[:, :, :-trim_from_buffer, :]
                self.buffer_len -= trim_from_buffer
            remaining -= trim_from_buffer

        # Trim compressed chunks (from end)
        while remaining > 0 and self.compressed_key_chunks:
            chunk_len = self.compressed_key_chunks[-1]['shape'][2]
            if remaining >= chunk_len:
                self.compressed_key_chunks.pop()
                self.compressed_value_chunks.pop()
                remaining -= chunk_len
            else:
                # Partial chunk trim: decompress, trim, re-buffer
                chunk_k = self.compressed_key_chunks.pop()
                chunk_v = self.compressed_value_chunks.pop()
                k_approx = self.key_pq.decompress(chunk_k['pq']).reshape(chunk_k['shape'])
                v_approx = self.value_pq.decompress(chunk_v['pq']).reshape(chunk_v['shape'])
                keep = chunk_len - remaining
                self.key_buffer = k_approx[:, :, :keep, :]
                self.value_buffer = v_approx[:, :, :keep, :]
                self.buffer_len = keep
                remaining = 0

        # Trim sink (last resort)
        if remaining > 0 and self.sink_keys is not None:
            trim_from_sink = min(self.sink_offset, remaining)
            self.sink_keys = self.sink_keys[:, :, :-trim_from_sink, :]
            self.sink_values = self.sink_values[:, :, :-trim_from_sink, :]
            self.sink_offset -= trim_from_sink
            remaining -= trim_from_sink

        self.offset -= n
        return n
```

#### Modified Files

**`src/exo/worker/engines/mlx/constants.py`**

Add new constants:

```python
# TurboQuant configuration
TURBOQUANT_ENABLED: bool = bool(os.environ.get("EXO_TURBOQUANT", ""))
TURBOQUANT_BITS: int = int(os.environ.get("EXO_TURBOQUANT_BITS", "3"))
TURBOQUANT_SINK_SIZE: int = int(os.environ.get("EXO_TURBOQUANT_SINK_SIZE", "128"))
TURBOQUANT_CHUNK_SIZE: int = int(os.environ.get("EXO_TURBOQUANT_CHUNK_SIZE", "64"))
TURBOQUANT_QJL_PROJECTIONS: int | None = (
    int(os.environ["EXO_TURBOQUANT_QJL_PROJECTIONS"])
    if os.environ.get("EXO_TURBOQUANT_QJL_PROJECTIONS")
    else None
)
```

**`src/exo/worker/engines/mlx/cache.py`**

In `make_kv_cache()`, add TurboQuant path:

```python
from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE, KV_CACHE_BITS,
    TURBOQUANT_ENABLED, TURBOQUANT_BITS, TURBOQUANT_SINK_SIZE,
    TURBOQUANT_CHUNK_SIZE, TURBOQUANT_QJL_PROJECTIONS,
)

def make_kv_cache(model: Model, max_kv_size: int | None = None, keep: int = 0) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        caches: KVCacheType = model.make_cache()

        if TURBOQUANT_ENABLED:
            from exo.worker.engines.mlx.turboquant_cache import TurboQuantKVCache
            replaced = 0
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    # Determine head_dim from model config
                    # This needs to be extracted from the model — see note below
                    tqc = TurboQuantKVCache(
                        head_dim=...,  # See "Extracting head_dim" section
                        bits=TURBOQUANT_BITS,
                        fp16_sink_size=TURBOQUANT_SINK_SIZE,
                        chunk_size=TURBOQUANT_CHUNK_SIZE,
                        qjl_projections=TURBOQUANT_QJL_PROJECTIONS,
                    )
                    caches[i] = tqc
                    replaced += 1
            logger.info(f"Using TurboQuant KV cache (bits={TURBOQUANT_BITS}, sink={TURBOQUANT_SINK_SIZE}) for {replaced}/{len(caches)} layers")
        elif KV_CACHE_BITS is not None:
            # ... existing QuantizedKVCache path ...
```

**`src/exo/shared/types/mlx.py`**

Add `TurboQuantKVCache` to the union type:

```python
from exo.worker.engines.mlx.turboquant_cache import TurboQuantKVCache

KVCacheType = Sequence[
    KVCache | RotatingKVCache | QuantizedKVCache | TurboQuantKVCache | ArraysCache | CacheList
]
```

Note: This creates a circular import. Better approach: use a Protocol or keep TurboQuantKVCache duck-typed (it already matches the interface). Just ensure `isinstance` checks in `cache.py` handle it. The key checks are:
- `isinstance(c, KVCache)` in `make_kv_cache()` — TurboQuantKVCache should NOT match this (it shouldn't subclass KVCache)
- `isinstance(c, (ArraysCache, RotatingKVCache))` in `has_non_kv_caches()` — TurboQuantKVCache should NOT match this
- `isinstance(c, KVCache)` in the QuantizedKVCache replacement loop — same, should not match
- `hasattr(c, "offset")` in `_entry_length()` — TurboQuantKVCache has `offset`, so this works
- `c.trim(n)` in `trim_cache()` — TurboQuantKVCache implements `trim()`

### Extracting head_dim

`head_dim` must be known at cache creation time. Options:

1. **From model config**: `model.args.head_dim` or `model.config.hidden_size // model.config.num_attention_heads`
2. **Lazy initialization**: Don't init compressors until first `update_and_fetch()` call when you see the actual tensor shapes
3. **From Qwen3.5 config**: `head_dim = 256` (known for target model)

Recommended: Lazy init. On first call to `update_and_fetch()`, read `keys.shape[-1]` and initialize compressors. This avoids coupling to model config format.

**Qwen3.5-397B confirmed dimensions** (from `patches/qwen3_5_moe/common.py`):
- `head_dim = 256`, `n_attn_heads = 32`, `n_kv_heads = 2`
- `linear_key_head_dim = 128`, `linear_value_head_dim = 128` (DeltaNet layers)
- Note: DeltaNet layers use `ArraysCache`, not KVCache, so TurboQuant only applies to the 8 standard attention layers per pipeline stage (not SSM layers)

### Interaction with `maybe_quantize_kv_cache()`

When `TURBOQUANT_ENABLED` is set and `KV_CACHE_BITS` is not:
- `maybe_quantize_kv_cache()` checks `hasattr(c, "to_quantized")` — TurboQuantKVCache won't have this method, so it's a no-op. Good.

When both are set:
- Don't do this. They're mutually exclusive. Add a check in constants or make_kv_cache.

### Interaction with `KVPrefixCache`

`KVPrefixCache` does `deepcopy(cache)` extensively. The `state` property and setter must support this. Since compressed chunks are stored as Python dicts of mx.arrays, `deepcopy` should work but verify that mx.array deepcopy does actual copies (it does — MLX arrays are reference-counted and deepcopy creates new arrays).

**Memory concern**: `deepcopy` of compressed cache is much cheaper than deepcopy of full FP16 cache. This is a bonus of TurboQuant.

### Interaction with Pipeline Parallel Prefill

In `pipeline_parallel_prefill()`:
- After each chunk: `quantize_cache_fn(_prompt_cache)` is called
- If using TurboQuantKVCache: this is a no-op (no `to_quantized` method)
- After each chunk: `mx.eval([c.state for c in _prompt_cache])` is called
- TurboQuantKVCache's `state` returns a tuple of arrays and lists — `mx.eval` needs to handle this
- **Potential issue**: `mx.eval` expects mlx arrays, but `state` returns mixed types (lists of dicts containing mx.arrays). `mx.eval([c.state for c in cache])` will fail if state contains non-array types.

**Fix**: Make `state` return a nested structure where all leaf values are mx.arrays. `mx.eval` can traverse nested lists/tuples of arrays. The compressed chunk dicts must be converted to tuples of arrays. Alternatively, implement a custom `eval_state()` method and modify the two call sites in `generate.py` (lines 211, 254) to use `mx.eval(flatten_arrays(c.state))` or similar.

Confirmed call sites:
- `generate.py:211`: `mx.eval([c.state for c in _prompt_cache])` — in prefill chunk loop
- `generate.py:254`: `mx.eval([c.state for c in _prompt_cache])` — post-prefill

### Interaction with ArraysCache Contiguous Fix

The contiguous fix in prefill (`generate.py:225-228`) only applies to `ArraysCache` entries, so no conflict.

## Known Limitations and Risks

### Qwen Compatibility

TurboQuant's reference implementation notes that Qwen 2.5 and Gemma 2 fail due to embedding attribute mismatches. This is a monkey-patching issue in the reference repo, not an algorithm issue. Since we're implementing a standalone cache class (not monkey-patching), this shouldn't apply. However, validate that Qwen3.5's attention layer correctly receives the decompressed FP16 tensors.

### Numerical Accuracy

- PolarQuant at 3-bit introduces quantization noise in attention scores
- QJL correction is unbiased but has variance proportional to `1/num_projections`
- Attention sink (128 uncompressed tokens) mitigates system prompt degradation
- **Must validate**: Run `bench/quant_compare.py` tasks with TurboQuant and compare output quality
- **Must validate**: Run `bench/sdpa_prefill_bench.py` equivalent to measure actual speedup

### Memory During Decompression

Option A (decompress-then-SDPA) creates a full FP16 copy of all K/V during every attention computation. For 65K context at Qwen3.5's config (2 KV heads, 256 dim), this is:
- Per layer: 2 * 65536 * 2 * 256 * 2 bytes = ~128 MB
- Temporary, freed after SDPA completes
- Still much less than storing FP16 permanently

### Prefill Performance

During prefill, TurboQuant adds compression overhead per chunk. This is compute (rotation, arctan2, QJL projection) that doesn't exist in the current affine quantization path. However, it removes the dequantization cost during subsequent SDPA calls on the growing cache, which is the 40% bottleneck. Net effect depends on context length — longer contexts benefit more.

## Implementation Status: EVALUATED — No Win on Qwen3.5

**Implemented and benchmarked (2026-04-04).** Adapted from flovflo/turboquant-mlx-qwen35-kv rather than full PolarQuant+QJL. Used simpler sign-flip + permutation rotation with MLX's native affine quantization.

### Benchmark Results (25K context, Qwen3.5-397B-A17B-4bit, 2-node PP RDMA)

| Approach | Cached Decode | vs Baseline |
|----------|--------------|-------------|
| Baseline (4-bit dequant + fused SDPA) | 40.9 tok/s | — |
| TurboQuant 4-bit + quantized_matmul | 37.0 tok/s | -10% |
| TurboQuant 3-bit + quantized_matmul | 38.3 tok/s | -6% |
| TurboQuant 3-bit + dequant + inverse-rotate + fused SDPA | 36.5 tok/s | -11% |

### Why It Doesn't Help

1. **Architecture mismatch**: Qwen3.5 has only 8/30 attention layers per PP rank (rest are DeltaNet/SSM). Savings on 8 layers can't overcome rotation overhead.
2. **Baseline already uses fused FlashAttention**: The existing code dequantizes then uses `mx.fast.scaled_dot_product_attention` (O(1) workspace). TurboQuant's `mx.quantized_matmul` is the slower unfused path.
3. **Rotation overhead ≈ dequant savings**: At 3-bit, dequant saves ~0.3ms/layer but rotation adds ~0.2ms/layer. Net gain is negligible on 8 layers.

### Key Micro-benchmark Finding

3-bit dequant is 29% faster than 4-bit per attention layer (0.78ms vs 1.10ms at 25K context). If quality at 3-bit could be preserved without rotation overhead, this would be a win — but plain 3-bit affine without rotation has quality degradation.

### When to Revisit

- Pure-attention models where ALL layers have KV caches
- If `mx.fast.scaled_dot_product_attention` adds native quantized KV tuple support
- Very long contexts (65K+) where dequant cost truly dominates

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXO_TURBOQUANT` | `""` (disabled) | Set to `1` to enable TurboQuant KV cache |
| `EXO_TURBOQUANT_BITS` | `3` | Key quantization bits |
| `EXO_TURBOQUANT_SKETCH_DIM` | `4` | Residual sketch dimensions |
| `EXO_TURBOQUANT_RESIDUAL` | `0` | Enable residual correction (`1` to enable) |

## Files

- `src/exo/worker/engines/mlx/turboquant_cache.py` — full implementation (kept, off by default)
- `src/exo/worker/engines/mlx/constants.py` — env var configuration
- `src/exo/worker/engines/mlx/cache.py` — TurboQuant path in `make_kv_cache()`
