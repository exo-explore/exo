# Prefill Optimization Spec

Implementation spec for reducing time-to-first-token (TTFT) in exo's pipeline-parallel prefill for Qwen3.5-397B-A17B on 2x Mac Studio M4 Max over TB5 RDMA. Intended for Claude Code to use during implementation.

## Current Performance

- TTFT: 1.5-2.6s (varies by prompt length and cache hit)
- Prefill chunk size: `EXO_PREFILL_STEP_SIZE=4096` (default)
- Pipeline: 2-stage, overlapping send/recv
- KV cache dequantization: ~40% of per-chunk SDPA time (from `bench/sdpa_prefill_bench.py`)
- DeltaNet contiguous fix: additional `mx.eval` per chunk for SSM layers

## Anatomy of One Prefill Chunk

Each chunk in `pipeline_parallel_prefill()` (`generate.py:194-236`) does the following sequentially:

```
1. model(prompt_chunk, cache=cache)          — forward pass (attention + MoE for all layers on this rank)
2. quantize_cache_fn(cache)                  — maybe_quantize_kv_cache (no-op if already QuantizedKVCache)
3. flush_prefill_sends()                     — async RDMA send of output to next rank
4. mx.eval([c.state for c in cache])         — SYNC BARRIER: materialize all cache states
5. for ArraysCache: mx.contiguous() + eval   — SYNC BARRIER: break shared buffer refs
6. progress_callback(processed, total)       — logging/UI
```

Steps 4 and 5 are synchronous GPU barriers — the CPU blocks until all Metal work completes. For a model with both attention (KVCache/QuantizedKVCache) and DeltaNet (ArraysCache) layers, there are effectively **two GPU sync points per chunk**.

## Optimization Opportunities

### 1. Reduce Sync Points Per Chunk

**Problem**: Two `mx.eval` calls per chunk (step 4 and step 5) each flush the entire GPU pipeline. On the M4 Max, a GPU flush costs ~50-100μs overhead plus the time to complete all pending work.

**Proposal**: Merge steps 4 and 5 into a single eval:

```python
# Current (two evals):
mx.eval([c.state for c in _prompt_cache])
for _c in _prompt_cache:
    if isinstance(_c, ArraysCache):
        _c.cache = [mx.contiguous(x) if x is not None else x for x in _c.cache]
        mx.eval(*[x for x in _c.cache if x is not None])

# Proposed (single eval):
for _c in _prompt_cache:
    if isinstance(_c, ArraysCache):
        _c.cache = [mx.contiguous(x) if x is not None else x for x in _c.cache]
# One combined eval for all cache states + contiguous arrays
all_states = []
for c in _prompt_cache:
    if isinstance(c, ArraysCache):
        all_states.extend([x for x in c.cache if x is not None])
    else:
        s = c.state
        if isinstance(s, tuple):
            all_states.extend([x for x in s if isinstance(x, mx.array)])
        elif isinstance(s, mx.array):
            all_states.append(s)
mx.eval(*all_states)
```

**Risk**: The contiguous fix was added specifically because eval alone wasn't sufficient — slices still shared parent buffer references. The `mx.contiguous` call creates independent copies, but it needs the source arrays to be materialized first. Test carefully that a single eval after contiguous works correctly.

**Files**: `src/exo/worker/engines/mlx/generator/generate.py:211-228`

**Expected impact**: Save ~50-100μs per chunk × number of chunks. For a 4096-token prompt with 1 chunk, negligible. For a 32K prompt with 8 chunks, saves ~400-800μs.

### 2. Overlap Forward Pass with Previous Chunk's Eval

**Problem**: The current loop is strictly sequential — chunk N's forward pass doesn't start until chunk N-1's eval completes. But the forward pass and the cache eval operate on different data (new chunk vs. already-cached data).

**Proposal**: Use two MLX streams to overlap computation:

```python
eval_stream = mx.new_stream(mx.default_device())

for i in range(n_real):
    # Forward pass on generation_stream (current)
    with mx.stream(generation_stream):
        model(prompt[processed:processed + chunk_size][None], cache=_prompt_cache)
        quantize_cache_fn(_prompt_cache)

    flush_prefill_sends()

    # Cache eval on separate stream — can overlap with next chunk's forward dispatch
    with mx.stream(eval_stream):
        mx.eval([c.state for c in _prompt_cache])
        # contiguous fix...

    processed += chunk_size
```

**Risk**: This is tricky. The next chunk's forward pass reads from the cache that the eval stream is materializing. MLX's stream dependency system should handle this (arrays carry stream dependencies), but pipeline communication (send/recv) may not respect cross-stream ordering. Needs careful validation.

**Files**: `src/exo/worker/engines/mlx/generator/generate.py:189-236`

**Expected impact**: Could overlap 30-50% of eval time with next chunk's kernel dispatch, saving ~0.5-1ms per chunk on long prompts. Higher impact at longer contexts where eval takes longer.

### 3. Prefill Chunk Size Tuning

**Problem**: The default 4096 was chosen empirically but may not be optimal for the 2-node M4 Max setup with Qwen3.5.

**Analysis**:
- Smaller chunks: more pipeline overlap (less bubble), more sync points, more overhead per chunk
- Larger chunks: fewer sync points, better GPU utilization per chunk, but more pipeline bubble time (rank 1 waits longer for first real chunk)
- The `sdpa_prefill_bench.py` uses QUERY_LEN=2048, suggesting that's the actual per-node chunk length being tested

**Proposal**: Benchmark sweep with `EXO_PREFILL_STEP_SIZE` at 1024, 2048, 4096, 8192, 16384. Measure both total prefill time and per-chunk breakdown (forward, eval, send).

**Implementation**: Add a `bench/prefill_chunk_sweep.py` that:
1. Loads the model on 2 nodes
2. Runs prefill with varying chunk sizes on a fixed prompt (e.g., 32K tokens)
3. Logs per-chunk timings using `time.perf_counter()` around each step
4. Reports total TTFT and breakdown

**Key config**: Note that `pipeline_parallel_prefill` adjusts step size: `prefill_step_size = prefill_step_size // min(4, group.size())`. For world_size=2, this is a no-op (2 < 4). For world_size=3, also no-op. Only kicks in at world_size >= 4.

**Files**: `bench/` (new file), `src/exo/worker/engines/mlx/generator/generate.py:150`

**Expected impact**: Potentially 10-30% TTFT improvement if current chunk size is suboptimal for this hardware.

### 4. Pipeline Stage Overlap During Prefill

**Problem**: The current pipeline overlap scheme has rank 0 doing real work while rank 1 does a dummy iteration, then they sync. This means rank 1 is idle for 1 chunk duration at the start, and rank 0 is idle for 1 chunk at the end.

**Current timing** (2 ranks, 3 real chunks):
```
Rank 0: [real_0] [real_1] [real_2] [dummy ]
Rank 1: [dummy ] [real_0] [real_1] [real_2]
```

**Observation**: The dummy iterations just call `distributed_prompt_progress_callback()` — they don't do any computation. During rank 0's first real chunk, rank 1 is doing nothing useful.

**Proposal**: Use the dummy iteration time for useful work:
- **Warm up Metal**: Run a small dummy forward pass to warm GPU caches and shader pipelines
- **Pre-fetch/tokenize**: If there are additional prompts in a batch, start processing them
- **KV cache pre-allocation**: Allocate cache memory during dummy slots instead of lazily during first real chunk

**Risk**: Low — dummy iterations are currently no-ops beyond the callback.

**Files**: `src/exo/worker/engines/mlx/generator/generate.py:190-192` (leading dummies), `238-240` (trailing dummies)

**Expected impact**: Small (50-200μs savings), but compounds with longer pipelines (3+ stages).

### 5. DeltaNet Kernel Selection for Prefill

**Problem**: Qwen3.5 has hybrid attention + DeltaNet (GatedDeltaNet) layers. The DeltaNet layers use chunkwise parallel algorithms during prefill, and `bench/chunkwise_benchmark.py` shows the optimal kernel varies with sequence length.

**From the benchmark** (Qwen3.5 config: Hk=16, Hv=64, Dk=192, Dv=128):
- Sequential kernel: best for T < 512
- Chunkwise (C=64): preferred for T > 2048
- Trade-off: memory vs compute efficiency

**Proposal**: Auto-select the DeltaNet kernel based on the prefill chunk size. If `EXO_PREFILL_STEP_SIZE` is 4096, use chunkwise with C=64. If a smaller chunk size is used (e.g., 1024), consider C=32 or sequential.

**Implementation**: Look for where the DeltaNet kernel is dispatched. The kernel selection may already be adaptive — check the `gated_delta_kernel` vs `gated_delta_chunkwise` dispatch logic. If it's hardcoded, make it configurable or auto-tuned based on input sequence length.

**Files**: Look in `src/exo/worker/engines/mlx/patches/qwen3_5_moe/` for DeltaNet forward pass code. Also `bench/chunkwise_benchmark.py` for the benchmark methodology.

**Expected impact**: 5-15% improvement in DeltaNet layer prefill time, which is a fraction of total prefill. Moderate impact.

### 6. Reduce Attention Dequantization Cost (TurboQuant)

**Problem**: 40% of per-chunk SDPA time is dequantizing the 4-bit KV cache. As the cache grows during prefill (later chunks attend over earlier ones), this cost increases.

**Solution**: TurboQuant integration (see `docs/turboquant-integration.md`). This is documented separately and addresses the dequant cost specifically.

**Expected impact**: Up to 40% reduction in per-chunk attention time for later chunks. Early chunks (small cache) see less benefit. Net TTFT improvement depends on prompt length — longer prompts benefit more.

### 7. Fused SDPA Kernel Optimization

**Problem**: The `sdpa_decomposed_bench.py` and `sdpa_decomposed_v2_bench.py` benchmarks suggest the decomposed (matmul + online softmax) approach can sometimes beat the fused SDPA, particularly for GQA configurations.

**Analysis from bench code**:
- `sdpa_decomposed_v2_bench.py` avoids the 16x GQA expansion by reshaping Q to `(B, Hkv, gqa_ratio, qL, D)` and doing per-KV-head matmuls
- `sdpa_gqa_profile.py` profiles the L2 cache fit hypothesis: at 2 KV heads + fp16 + 256 dims, ~49K tokens fit in L2 (~48 MB)

**Proposal**: If the decomposed approach is faster for Qwen3.5's 16:1 GQA ratio, replace the fused SDPA call during prefill with the decomposed version. This could be done conditionally (only during prefill when chunk sizes are large).

**Files**: `bench/sdpa_decomposed_v2_bench.py` for reference implementation, attention code in `src/exo/worker/engines/mlx/patches/qwen3_5_moe/`

**Expected impact**: 10-20% improvement in SDPA time if decomposed is faster for this configuration. Benchmark first.

### 8. Aggressive Prefix Caching

**Problem**: `KVPrefixCache` already exists and works, but its effectiveness depends on workload patterns. For interactive use (chat), the system prompt is often identical across requests.

**Current behavior** (`cache.py:159-232`):
- LRU eviction when memory exceeds threshold (85% on 128GB)
- Token-level prefix matching
- Media region validation
- SSM snapshot restoration for partial matches

**Proposal**: Exploit the attention sink concept from TurboQuant at the prefix cache level. For chat workloads:
1. **Pin system prompt cache**: Never evict the system prompt's KV cache entry. It's shared across all requests.
2. **Hierarchical caching**: L1 = last N conversations (hot), L2 = system prompt only (permanent)
3. **Warm prefill**: On startup, automatically prefill the default system prompt so the first real request skips that portion entirely.

**Implementation**:
- Add `KVPrefixCache.pin(index)` to mark entries as non-evictable
- In `warmup_inference()` (`generate.py:390-452`), optionally prefill with the configured system prompt
- Add `EXO_PIN_SYSTEM_PROMPT=1` env var to enable

**Files**: `src/exo/worker/engines/mlx/cache.py` (KVPrefixCache), `src/exo/worker/engines/mlx/generator/generate.py` (warmup_inference)

**Expected impact**: Eliminates system prompt prefill for repeated requests. If system prompt is 500 tokens, saves ~200-400ms per request after first. High impact for chat workloads.

### 9. Early Exit for Short Prompts

**Problem**: `pipeline_parallel_prefill()` is used when `num_tokens >= prefill_step_size`. For prompts just above this threshold (e.g., 4097 tokens), the pipeline overhead (dummy iterations, send/recv, sync) may exceed the benefit.

**Proposal**: Increase the threshold for pipeline prefill, or add a heuristic:
- If `num_tokens < 2 * prefill_step_size`, use `stream_generate` (simpler, no pipeline overhead)
- Pipeline prefill only when there are enough chunks to amortize the setup cost

**Files**: `src/exo/worker/engines/mlx/generator/generate.py:330`

**Expected impact**: Small — only affects prompts near the threshold boundary.

### 10. Memory Pre-allocation for KV Cache

**Problem**: `QuantizedKVCache` pre-allocates in steps of 16384 (`step = 16384`). But the first allocation happens lazily on the first `update_and_fetch` call, which is inside the first prefill chunk's forward pass. This adds allocation latency to the first chunk.

**Proposal**: Pre-allocate the KV cache to the expected prompt length before starting prefill. If the prompt is 32K tokens, allocate cache buffers for 32K upfront rather than growing in 16K steps.

**Implementation**:
```python
# Before prefill loop:
estimated_cache_size = len(prompt_tokens) + MAX_EXPECTED_GENERATION
for c in cache:
    if hasattr(c, 'pre_allocate'):
        c.pre_allocate(estimated_cache_size)
```

Add `pre_allocate(n)` method to `QuantizedKVCache` (and `TurboQuantKVCache`).

**Files**: `mlx-lm/mlx_lm/models/cache.py` (QuantizedKVCache), `src/exo/worker/engines/mlx/generator/generate.py` (before prefill call)

**Expected impact**: Eliminates re-allocation stutters during prefill. Most impactful for long prompts that trigger multiple step expansions.

## Priority Order

Based on expected impact and implementation complexity:

| Priority | Optimization | Expected Impact | Complexity | Status |
|----------|-------------|-----------------|------------|--------|
| 1 | Prefill chunk size sweep (benchmark) | 10-30% TTFT | Low | **DONE** — 4096 is optimal |
| 2 | Aggressive prefix caching (pin system prompt) | 200-400ms per repeated request | Low | **DONE** — `KVPrefixCache.pin()` added |
| 3 | TurboQuant KV cache (separate doc) | Up to 40% of attention time | Medium | **EVALUATED** — no win on Qwen3.5 (8/30 layers are attention) |
| 4 | Merge sync points per chunk | 50-100μs × N chunks | Low | **BLOCKED** — contiguous fix depends on first eval |
| 5 | Fused vs decomposed SDPA benchmark | 10-20% of SDPA time | Low (benchmark first) | Not started |
| 6 | DeltaNet kernel auto-selection | 5-15% of DeltaNet time | Medium | Not started |
| 7 | Overlap forward + eval (dual stream) | 0.5-1ms per chunk | High (correctness risk) | Not started |
| 8 | KV cache pre-allocation | Eliminates alloc stutter | Low | Not started |
| 9 | Pipeline dummy slot utilization | 50-200μs | Low | Not started |
| 10 | Early exit for short prompts | Edge case | Low | Not started |

### Key Finding

The real bottleneck on Qwen3.5-397B is **DeltaNet/SSM layers** (22/30 per PP rank), not attention or KV cache dequant. Optimizing the 8 attention layers (chunk size, KV cache, SDPA) yields diminishing returns. The highest-impact next step is optimizing the GatedDeltaNet chunkwise parallel kernel.

## Benchmarking Infrastructure

All prefill benchmarks should follow the pattern established in `bench/sdpa_prefill_bench.py`:

```python
mx.synchronize()
t0 = time.perf_counter()
# ... operation ...
mx.eval(result)
mx.synchronize()
elapsed = time.perf_counter() - t0
```

For end-to-end prefill timing, instrument `pipeline_parallel_prefill()` with per-step timers:

```python
# Add inside the chunk loop (generate.py:194):
t_forward = time.perf_counter()
model(prompt[processed:processed + chunk_size][None], cache=_prompt_cache)
t_quantize = time.perf_counter()
quantize_cache_fn(_prompt_cache)
t_send = time.perf_counter()
flush_prefill_sends()
t_eval = time.perf_counter()
mx.eval([c.state for c in _prompt_cache])
t_contiguous = time.perf_counter()
# ... contiguous fix ...
t_done = time.perf_counter()

logger.info(
    f"Chunk {i}: forward={t_quantize-t_forward:.3f}s "
    f"quant={t_send-t_quantize:.3f}s "
    f"send={t_eval-t_send:.3f}s "
    f"eval={t_contiguous-t_eval:.3f}s "
    f"contiguous={t_done-t_contiguous:.3f}s"
)
```

This instrumentation should be gated behind `EXO_PROFILE_PREFILL=1` env var so it doesn't affect production.

## Environment Variables (New)

| Variable | Default | Description |
|----------|---------|-------------|
| `EXO_PROFILE_PREFILL` | `""` | Enable per-chunk timing logs |
| `EXO_PIN_SYSTEM_PROMPT` | `""` | Pin system prompt KV cache (never evict) |

## Dependencies

- **TurboQuant**: See `docs/turboquant-integration.md` — addresses optimization #3
- **Speculative decoding (MTP/DFlash)**: Independent work, decode-only — no conflict with prefill changes
- **Chunk size tuning**: Should be done first (benchmark only) to establish baseline before other changes
