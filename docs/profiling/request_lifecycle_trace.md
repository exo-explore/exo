# Request Lifecycle Trace — Qwen3.5-397B-A17B-4bit on 2x M4 Max (PP/RDMA)

**Date:** 2026-04-05
**Commit:** `19d8081d`
**Test:** 16,010 prompt tokens → 257 decode tokens, 42.5s wall clock
**Config:** `EXO_PREFILL_STEP_SIZE=4096`, `EXO_KV_CACHE_BITS=4`, `EXO_SPECULATIVE=1`, PP speculation with MTP drafting

All times relative to request arrival. Measured with `time.perf_counter()` (µs resolution).

---

## Phase 1: Request Startup (0–1.09s)

| Offset | Duration | Span | Notes |
|--------|----------|------|-------|
| 2.8ms | 201µs | `runner.update_status_running` | MP queue send to main process |
| 3.3ms | 93µs | `runner.acknowledge_task` | MP queue send |
| 3.4ms | 19µs | `runner.submit_text_generation` | Enqueue to batch generator |
| 3.4ms | **1.063s** | `batch_gen.agree_on_tasks` | **First-request task agreement — 2x `all_gather` over RDMA. Both ranks must reach this point.** |
| 1.066s | 267µs | `start_task.apply_chat_template` | Jinja2 template rendering |
| 1.066s | 12.7ms | `submit.encode_prompt` | Tokenization (16K tokens) |
| 1.079s | 2µs | `submit.vision` | No-op (text only) |
| 1.079s | 208µs | `submit.kv_prefix_cache_lookup` | KV cache miss (first request) |
| 1.079s | 3µs | `submit.make_sampler` | Trivial |

**Startup total: 1.079s** — dominated by `agree_on_tasks` (1.063s). This is the ranks synchronizing for the first time after idle. Subsequent requests with warm ranks would be much faster.

---

## Phase 2: Prefill (1.079s–35.381s) — 34.302s total

### Prefill Setup

| Offset | Duration | Span | Notes |
|--------|----------|------|-------|
| 1.079s | 1.050ms | `prefill.clear_cache` | Release Metal buffers for headroom |
| 1.080s | 11.685ms | `prefill.barrier` | `mx.distributed.all_sum` barrier (R1 saw 61µs — R0 arrived 11ms later) |
| 1.092s | 673µs | `prefill.mem_checkpoint` | `mx.eval(zeros(1))` + memory stats |

### Prefill Chunks (8 chunks × 2048 tokens, last chunk 1672 tokens)

**Rank 0 (layers 0–29):**

| Chunk | Forward | Distributed CB | Flush Sends | Eval Cache | Contiguous |
|-------|---------|----------------|-------------|------------|------------|
| 0 | 3.021s | 433µs | 9µs | 164µs | 662µs |
| 1 | 3.010s | **1.186s** | 14µs | 218µs | 774µs |
| 2 | 3.093s | 431ms | 10µs | 174µs | 654µs |
| 3 | 3.191s | 427ms | 12µs | 214µs | 740µs |
| 4 | 3.257s | 418ms | 13µs | 206µs | 675µs |
| 5 | 3.339s | 420ms | 9µs | 182µs | 656µs |
| 6 | 3.425s | 436ms | 17µs | 264µs | 692µs |
| 7 | 2.942s | 1.000s | 15µs | 217µs | 753µs |

**Rank 1 (layers 30–59):**

| Chunk | Forward | Distributed CB | Flush Sends | Eval Cache | Contiguous |
|-------|---------|----------------|-------------|------------|------------|
| 0 | 4.197s | 434µs | 1µs | **331ms** | 656µs |
| 1 | 3.192s | 290µs | 1µs | **313ms** | 629µs |
| 2 | 3.305s | 380µs | 0µs | **313ms** | 627µs |
| 3 | 3.361s | 402µs | 0µs | **313ms** | 663µs |
| 4 | 3.446s | 312µs | 0µs | **314ms** | 771µs |
| 5 | 3.547s | 490µs | 0µs | **313ms** | 611µs |
| 6 | 3.628s | 402µs | 0µs | **313ms** | 618µs |
| 7 | 3.106s | 337µs | 0µs | **277ms** | 611µs |

**Key observations:**
- **R0 `forward` includes blocking GPU eval + PP send** (via `PipelineLastLayer`). The forward time IS the GPU compute time for layers 0–29.
- **R0 `eval_cache` ≈ 200µs** — cache already materialized as side effect of `PipelineLastLayer`'s eval.
- **R1 `forward` includes blocking `recv_like` wait** for R0's hidden states, then GPU compute for layers 30–59.
- **R1 `eval_cache` ≈ 313ms** — actual GPU compute time for R1's layers, materializing cache.
- **R0 `distributed_cb` ≈ 420-436ms (steady state)** — R0 waiting at the PP barrier for R1 to finish its eval_cache + contiguous. This is NOT wasted work — it's the pipeline bubble.
- **Chunks 1 and 7 have ~1s `distributed_cb`** on R0 — these are the first/last chunks of the pipeline where the stagger causes extra wait.
- **Forward time grows** from 3.0s → 3.4s per chunk as KV cache grows (more attention computation per token).
- **`contiguous` ≈ 650-770µs** — breaking DeltaNet ArraysCache shared buffer references.
- **`flush_sends` ≈ 0-17µs** — negligible (async sends already flushed during forward).

### Prefill Post-Processing

| Offset | Duration | Span | Notes |
|--------|----------|------|-------|
| 30.699s / 34.448s | 3.419s / 1.009s | `prefill.distributed_callback` | Trailing PP sync (R0 waits for R1's last chunk) |
| 34.118s / 34.448s | 1.262s / 1.009s | `prefill.post_loop_tokens` | 2x single-token forward passes (for stream_generate compatibility) |
| 35.381s / 35.459s | 153µs / 149µs | `prefill.cache_trim_and_rollback` | Trim extra tokens + SSM state deepcopy rollback |

**Prefill total: 34.3s** at **467 tok/s** (16,010 tokens). Forward compute dominates. The pipeline bubble (R0 waiting for R1 at distributed callbacks) is inherent to 2-rank PP.

---

## Phase 3: PP Speculation Setup (35.381s–35.624s) — 322ms

| Offset | Duration | Span | Notes |
|--------|----------|------|-------|
| 35.381s | 4µs | `pp_spec.get_pipeline_info` | |
| 35.381s | 52µs | `pp_spec.install_spec_layers` | Monkey-patch model layers for speculation |
| 35.382s | **242ms** | `pp_spec.first_token` | First decode token via standard PP (no speculation yet) |
| 35.624s | --- | `pp_spec.decode_loop_start` | Marker |

**PP spec setup total: 322ms.** The `first_token` (242ms) includes a full PP round-trip: R0 forward (17ms) + R1 forward (16ms) + collectives + warmup overhead.

---

## Phase 4: Decode (35.624s–~40.7s) — ~5.1s for 257 tokens

### Steady-State Decode Step Timing

**Normal step (~17-20ms, ~88% of steps):**
- R0: `r0_compute=1.1-1.5ms` (send pre-computed hidden) + `r0_draft=16-19ms` (MTP predict + speculative forward)
- R1: `r1_compute=15.7-16.2ms` (recv + forward + sample) + `tok_xchg=1.5-1.9ms`
- Both ranks work **in parallel** — R0 drafts while R1 computes

**Outlier step (~37ms, ~12% of steps — draft rejection):**
- R0: `r0_compute=17.3ms` (full forward, no pre-computed hidden) + `r0_draft=19ms`
- R1: `r1_compute=35.5ms` (waits 17ms for R0, then 16ms compute) + `tok_xchg=1.5ms`
- **Sequential** — R1 blocked waiting for R0's full forward pass

### Decode Overhead (per token)

| Component | R0 | R1 | Notes |
|-----------|-----|-----|-------|
| `callback` (on_generation_token) | 0µs | 0µs | Only fires every 50 tokens |
| `detokenizer` | 10µs | 10µs | |
| `stop_check` | 10µs | 10µs | |
| `logprobs` | 0µs | 0µs | Not requested |
| `response_build` | 10µs | 10µs | |
| MP queue send | 40µs | 0µs | Only R0 sends to API |
| `recv_poll` | 50µs | 30µs | Non-blocking queue check |
| **Total CPU overhead** | **~120µs** | **~60µs** | **<1% of step time** |

### Decode Agreement Callbacks (every 50 tokens)

| Component | R0 | R1 | Notes |
|-----------|-----|-----|-------|
| `agree_on_cancel_and_tasks` | 1.696ms (first), then skipped | 163µs | 5 distributed collectives |
| `agree_on_tasks` (per step) | 29-54µs | 28-262µs | Fast when no tasks pending |

**Decode rate: ~50 tok/s** (257 tokens in ~5.1s). MTP speculation acceptance rate ~88%.

---

## Phase 5: Cleanup (<1ms)

| Offset | Duration | Span | Notes |
|--------|----------|------|-------|
| ~40.7s | 2µs | `submit.clamp_rotating_caches` | No-op |
| ~40.7s | 174µs | `submit.save_prefix_cache` | Save KV cache for future prefix matching |
| ~40.7s | 3µs | `submit.make_logits_processors` | No-op |

---

## Summary: Where Every Second Goes

### 16K Prefill (34.3s)

| Component | Time | % | Notes |
|-----------|------|---|-------|
| GPU forward compute (R0 layers 0-29) | ~25.3s | 73.8% | Grows with KV cache size |
| GPU forward compute (R1 layers 30-59) | ~25.3s | — | Overlapped with R0 (pipeline parallel) |
| R1 eval_cache (materialize R1's KV cache) | ~2.5s | 7.3% | ~313ms/chunk × 8 |
| PP pipeline bubble (R0 waiting for R1) | ~4.3s | 12.5% | Inherent to 2-rank PP |
| Contiguous (DeltaNet buffer fix) | ~5.4ms | <0.1% | |
| Post-loop tokens | ~1.3s | 3.8% | stream_generate compatibility |
| Other (barrier, clear_cache, mem check) | ~13ms | <0.1% | |
| First-request agree_on_tasks | ~1.1s | 3.2% | One-time sync at request start |

### Decode (257 tokens in ~5.1s)

| Component | Time | % | Notes |
|-----------|------|---|-------|
| GPU forward + MTP draft (normal steps) | ~3.9s | 76.5% | ~17ms × 226 steps |
| GPU forward (outlier/reject steps) | ~1.1s | 21.6% | ~37ms × 31 steps |
| PP spec setup + first token | 322ms | 6.3% | One-time |
| CPU overhead (detok, send, etc.) | ~31ms | 0.6% | ~120µs × 257 |
| Agreement callbacks | ~2ms | <0.1% | Every 50 tokens |

### Total Request (42.5s wall clock)

| Phase | Time | % |
|-------|------|---|
| Startup (agree_on_tasks) | 1.1s | 2.6% |
| Prefill | 34.3s | 80.7% |
| PP spec setup | 322ms | 0.8% |
| Decode | ~5.1s | 12.0% |
| TTFT overhead (encode, template, etc.) | ~80ms | 0.2% |
| Unaccounted | ~1.6s | 3.8% |
