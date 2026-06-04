# Plan: TurboQuant-style KV cache compression for MLX and PDD

## Summary

This plan describes how to add TurboQuant-style KV cache compression to exo's
MLX inference path, with Apple Silicon decode as the primary target and
Spark-to-Mac prefill/decode disaggregation (PDD) as the motivating deployment.

The goal is not to quantize model weights. It is to compress runtime cache state
after prefill so long-context decode uses less memory and, eventually, can move
cache state between prefill and decode placements more cheaply.

For `mlx-community/Qwen3-Coder-Next-bf16`, only some layers use ordinary
attention KV cache. The remaining Qwen3-Next Gated DeltaNet layers use
`ArraysCache` state. TurboQuant should therefore target `KVCache` entries first
and preserve `ArraysCache` state losslessly until a model-specific compression
scheme is proven.

## Current state

- exo already has MLX cache abstractions in the generation path:
  - `KVCache`, `QuantizedKVCache`, `RotatingKVCache`, `ArraysCache`, and
    `CacheList` are accepted cache entry types.
  - `maybe_quantize_kv_cache()` is already called during prefill/generation, but
    `KV_BITS` and `KV_CACHE_BITS` are disabled by default.
- MLX already has a basic affine `QuantizedKVCache`.
- TurboQuant for MLX exists only as experimental external work:
  - upstream `mlx-lm` PR `#1059` adds an experimental `TurboQuantKVCache`.
  - `rachittshah/mlx-turboquant` provides a proof-of-concept package.
  - `TheTom/turboquant_plus` explores a faster delegated cache design, but with
    a larger dependency/fork surface.
- Qwen3-Next style models are hybrid. For Qwen3-Coder-Next, full attention is
  every fourth layer, so cache compression will mostly affect those full
  attention layers. The DeltaNet state is much smaller and should remain raw in
  v1.

## Implementation approach

### Phase 1: benchmark existing MLX cache quantization

Add an opt-in benchmark harness before adding new cache code.

- Add a small benchmark script or test utility that runs prefill plus decode for:
  - raw MLX cache,
  - existing `QuantizedKVCache` at 8-bit and 4-bit,
  - later TurboQuant cache candidates.
- Measure:
  - prefill tokens/sec,
  - decode tokens/sec,
  - peak process/GPU memory where available,
  - cache bytes by layer and cache type,
  - next-token logit drift against raw cache.
- Use small, fast models first:
  - one Llama/Qwen dense attention model,
  - one Qwen3-Next style model if available locally.
- Do not enable cache quantization by default.

Success criteria:

- Basic KV cache quantization can be toggled and measured without changing
  default behavior.
- Qwen3-Next `KVCache` and `ArraysCache` entries are separately accounted for.

### Phase 2: introduce a TurboQuant cache adapter

Add a local adapter layer instead of hard-forking exo's generation logic.

- Define an internal cache policy enum:
  - `raw`
  - `mlx_quantized`
  - `turboquant`
- Add an environment-gated cache factory path for MLX runners:
  - default remains `raw`,
  - `EXO_MLX_KV_CACHE_POLICY=mlx_quantized` uses existing MLX cache
    quantization,
  - `EXO_MLX_KV_CACHE_POLICY=turboquant` uses the TurboQuant implementation
    when installed/available.
- Prefer adapting the upstream `mlx-lm` `TurboQuantKVCache` shape once it
  stabilizes. If using an external package first, isolate imports behind one
  module so it can be removed later.
- Only replace ordinary `KVCache` entries. Leave `ArraysCache`, `RotatingKVCache`,
  and unknown cache entries untouched unless explicitly supported.
- Keep recent tokens raw if the chosen TurboQuant implementation supports it;
  otherwise use TurboQuant only as a memory experiment, not as a decode-speed
  default.

Success criteria:

- A dense attention model can prefill and decode with TurboQuant cache.
- Qwen3-Next models can run with full-attention `KVCache` entries compressed and
  DeltaNet `ArraysCache` entries preserved raw.
- Prefix cache, trimming, cancellation, and batching still work for raw cache.
  TurboQuant may initially be single-request only if the external implementation
  lacks trim/merge support.

### Phase 3: Apple Silicon fast path

Optimize for Macs after correctness is proven.

- Avoid a dequantize-entire-cache-then-attend design. That saves memory but can
  cut decode throughput.
- Target a fused/tiled attention path:
  - store old K/V blocks in TurboQuant form,
  - keep a small recent window raw,
  - dequantize K/V tiles inside the attention kernel or immediately before the
    attention tile is consumed,
  - avoid writing full BF16 K/V back to memory.
- Start with 4-bit. Treat 3-bit as experimental until quality is validated.
- Keep this behind a separate feature flag until decode speed is within an
  acceptable range of raw cache on Apple Silicon.

Success criteria:

- On Apple Silicon, long-context decode with TurboQuant cache is no worse than
  raw cache by an agreed threshold while substantially reducing cache memory.
- The implementation does not regress raw-cache throughput.

### Phase 4: PDD cache handoff prototype

Use cache compression to make Spark-to-Mac PDD practical.

- Add explicit cache serialization boundaries only after cache correctness is
  proven in-process.
- Serialize:
  - token ids and prompt metadata,
  - per-layer cache type and shape,
  - compressed TurboQuant payloads for `KVCache` layers,
  - raw `ArraysCache` payloads for Qwen3-Next linear/DeltaNet layers,
  - position offsets and cache metadata.
- Restrict v1 PDD to same model revision, same tokenizer, same MLX cache layout,
  and same layer ownership on the decode placement.
- Do not attempt layer-resharding of cache in v1.

Success criteria:

- Prefill on one MLX runner, serialize cache, load cache into another compatible
  MLX runner, and continue decode with matching output within tolerance.
- For Qwen3-Coder-Next, the transfer size is reduced for long prompts compared
  with raw BF16 KV cache.

## Test plan

- Unit tests:
  - cache policy selection keeps default `raw`,
  - unsupported cache entries remain untouched,
  - TurboQuant imports fail closed with a clear error,
  - cache byte accounting distinguishes `KVCache` and `ArraysCache`.
- Correctness tests:
  - dense model raw vs quantized/TurboQuant logits for short prompts,
  - Qwen3-Next raw vs compressed full-attention cache with raw `ArraysCache`,
  - prefix cache hit, trim, and continuation behavior where supported.
- Performance tests:
  - 4k, 16k, and 64k prompt prefill/decode benchmarks,
  - Apple Silicon decode throughput comparison,
  - Spark/Mac transfer-size estimate from actual serialized cache.
- Integration tests:
  - single-node MLX generation with cache policy enabled,
  - multi-node pipeline generation remains unchanged with raw policy,
  - PDD prototype round-trip only after serialization exists.

## Risks and defaults

- Default behavior must remain raw cache until TurboQuant correctness and speed
  are proven.
- Qwen3-Coder-Next is not a simple KV-only model; v1 must preserve `ArraysCache`
  raw.
- A naive TurboQuant implementation may reduce memory but slow decode. It should
  not be considered successful for Apple Silicon unless decode speed is measured.
- External TurboQuant implementations are experimental. Keep dependencies
  optional and isolated.
- PDD should be treated as a follow-on feature. Cache compression is useful on
  its own, but cache handoff requires additional serialization and placement
  work.
