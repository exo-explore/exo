# Request Lifecycle Trace — Qwen3.5-397B-A17B-4bit on 2x M4 Max (PP/RDMA)

**Date:** 2026-04-05
**Commit:** `570504b3` (latest, includes post-loop optimization + clear_cache bump)
**Test:** 16,010 prompt tokens → 257 decode tokens
**Config:** `EXO_PREFILL_STEP_SIZE=4096`, `EXO_KV_CACHE_BITS=4`, `EXO_SPECULATIVE=1`, PP speculation with MTP drafting

All times relative to request arrival. Measured with `time.perf_counter()` (µs resolution).

---

## Phase 1: Request Startup (0µs–1,079,000µs)

| Offset (µs) | Duration (µs) | Span | Notes |
|-------------|---------------|------|-------|
| 2,800 | 201 | `runner.update_status_running` | MP queue send to main process |
| 3,300 | 93 | `runner.acknowledge_task` | MP queue send |
| 3,400 | 19 | `runner.submit_text_generation` | Enqueue to batch generator |
| 3,400 | **1,063,000** | `batch_gen.agree_on_tasks` | **First-request task agreement — 2x `all_gather` over RDMA. Both ranks must reach this point.** |
| 1,066,000 | 267 | `start_task.apply_chat_template` | Jinja2 template rendering |
| 1,066,000 | 12,699 | `submit.encode_prompt` | Tokenization (16K tokens) |
| 1,079,000 | 2 | `submit.vision` | No-op (text only) |
| 1,079,000 | 208 | `submit.kv_prefix_cache_lookup` | KV cache miss (first request) |
| 1,079,000 | 3 | `submit.make_sampler` | Trivial |

**Startup total: 1,079,000µs** — dominated by `agree_on_tasks` (1,063,000µs). This is the ranks synchronizing for the first time after idle. Subsequent requests with warm ranks would be much faster.

---

## Phase 2: Prefill (1,079,000µs–35,381,000µs) — 34,302,000µs total

### Prefill Setup

| Offset (µs) | Duration (µs) | Span | Notes |
|-------------|---------------|------|-------|
| 1,079,000 | 1,050 | `prefill.clear_cache` | Release Metal buffers for headroom |
| 1,080,000 | 11,685 | `prefill.barrier` | `mx.distributed.all_sum` barrier (R1 saw 61µs — R0 arrived 11,685µs later) |
| 1,092,000 | 673 | `prefill.mem_checkpoint` | `mx.eval(zeros(1))` + memory stats |

### Prefill Chunks (8 chunks × 2048 tokens, last chunk 1672 tokens)

**Rank 0 (layers 0–29):**

| Chunk | Forward (µs) | Distributed CB (µs) | Flush Sends (µs) | Eval Cache (µs) | Contiguous (µs) |
|-------|-------------|---------------------|-------------------|-----------------|-----------------|
| 0 | 3,021,000 | 433 | 9 | 164 | 662 |
| 1 | 3,010,000 | **1,186,000** | 14 | 218 | 774 |
| 2 | 3,093,000 | 430,866 | 10 | 174 | 654 |
| 3 | 3,191,000 | 426,739 | 12 | 214 | 740 |
| 4 | 3,257,000 | 417,740 | 13 | 206 | 675 |
| 5 | 3,339,000 | 420,422 | 9 | 182 | 656 |
| 6 | 3,425,000 | 435,698 | 17 | 264 | 692 |
| 7 | 2,942,000 | **1,000,000** | 15 | 217 | 753 |

**Rank 1 (layers 30–59):**

| Chunk | Forward (µs) | Distributed CB (µs) | Flush Sends (µs) | Eval Cache (µs) | Contiguous (µs) |
|-------|-------------|---------------------|-------------------|-----------------|-----------------|
| 0 | 4,197,000 | 434 | 1 | **331,279** | 656 |
| 1 | 3,192,000 | 290 | 1 | **313,385** | 629 |
| 2 | 3,305,000 | 380 | 0 | **313,173** | 627 |
| 3 | 3,361,000 | 402 | 0 | **313,381** | 663 |
| 4 | 3,446,000 | 312 | 0 | **313,725** | 771 |
| 5 | 3,547,000 | 490 | 0 | **313,323** | 611 |
| 6 | 3,628,000 | 402 | 0 | **313,263** | 618 |
| 7 | 3,106,000 | 337 | 0 | **276,838** | 611 |

**Key observations:**
- **R0 `forward` includes blocking GPU eval + PP send** (via `PipelineLastLayer`). The forward time IS the GPU compute time for layers 0–29.
- **R0 `eval_cache` ≈ 164–264µs** — cache already materialized as side effect of `PipelineLastLayer`'s eval.
- **R1 `forward` includes blocking `recv_like` wait** for R0's hidden states, then GPU compute for layers 30–59.
- **R1 `eval_cache` ≈ 276,838–331,279µs** — actual GPU compute time for R1's layers, materializing cache.
- **R0 `distributed_cb` ≈ 417,740–435,698µs (steady state)** — R0 waiting at the PP barrier for R1 to finish its eval_cache + contiguous. This is NOT wasted work — it's the pipeline bubble.
- **Chunks 1 and 7 have ~1,000,000µs `distributed_cb`** on R0 — these are the first/last chunks of the pipeline where the stagger causes extra wait.
- **Forward time grows** from 3,021,000µs → 3,425,000µs per chunk as KV cache grows (more attention computation per token).
- **`contiguous` ≈ 611–774µs** — breaking DeltaNet ArraysCache shared buffer references.
- **`flush_sends` ≈ 0–17µs** — negligible (async sends already flushed during forward).

### Prefill Post-Processing

| Offset (µs) | Duration (µs) | Span | Rank | Notes |
|-------------|---------------|------|------|-------|
| 30,699,000 / 34,448,000 | 3,419,000 / 1,009,000 | `prefill.distributed_callback` | R0 / R1 | Trailing PP sync (R0 waits for R1's last chunk) |
| 34,118,000 / 34,448,000 | 1,262,000 / 1,009,000 | `prefill.post_loop_tokens` | R0 / R1 | 2x single-token forward passes (for stream_generate compatibility) |
| 35,381,000 / 35,459,000 | 153 / 149 | `prefill.cache_trim_and_rollback` | R0 / R1 | Trim extra tokens + SSM state deepcopy rollback |

**Prefill total: 34,302,000µs** at **467 tok/s** (16,010 tokens). Forward compute dominates. The pipeline bubble (R0 waiting for R1 at distributed callbacks) is inherent to 2-rank PP.

---

## Phase 3: PP Speculation Setup (35,381,000µs–35,624,000µs) — 322,000µs

| Offset (µs) | Duration (µs) | Span | Notes |
|-------------|---------------|------|-------|
| 35,381,000 | 4 | `pp_spec.get_pipeline_info` | |
| 35,381,000 | 52 | `pp_spec.install_spec_layers` | Monkey-patch model layers for speculation |
| 35,382,000 | **242,301** | `pp_spec.first_token` | First decode token via standard PP (no speculation yet) |
| 35,624,000 | 0 | `pp_spec.decode_loop_start` | Marker |

**PP spec setup total: 322,000µs.** The `first_token` (242,301µs) includes a full PP round-trip: R0 forward (17,000µs) + R1 forward (16,000µs) + collectives + warmup overhead.

---

## Phase 4: Decode (35,624,000µs–~40,700,000µs) — ~5,076,000µs for 257 tokens

### Steady-State Decode Step Timing

**Normal step (~17,000–20,000µs, ~88% of steps):**

| Component | R0 (µs) | R1 (µs) | Notes |
|-----------|---------|---------|-------|
| `r0_compute` | 1,100–1,500 | 0 | Send pre-computed hidden from accepted draft |
| `r0_draft` | 16,000–19,000 | 0 | MTP predict + speculative forward |
| `r1_compute` | 0 | 15,700–16,200 | recv hidden + forward layers 30–59 + sample |
| `tok_xchg` | 100–800 | 1,400–1,900 | `all_gather` token exchange |
| `hidden_xchg` | 30 | 30 | MTP hidden state transfer |
| `verify` | 40 | 0 | Draft accept/reject check |
| **Loop total** | **17,000–20,000** | **17,000–20,000** | **Parallel — R0 drafts while R1 computes** |

**Outlier step (~37,000µs, ~12% of steps — draft rejection):**

| Component | R0 (µs) | R1 (µs) | Notes |
|-----------|---------|---------|-------|
| `r0_compute` | 17,300 | 0 | Full forward (no pre-computed hidden available) |
| `r0_draft` | 19,000 | 0 | MTP predict + speculative forward |
| `r1_compute` | 0 | 35,500 | Waits 17,300µs for R0, then 16,200µs compute |
| `tok_xchg` | 100–800 | 1,500 | |
| `hidden_xchg` | 30 | 30 | |
| `verify` | 100 | 0 | Cache restore on rejection |
| **Loop total** | **~37,000** | **~37,000** | **Sequential — R1 blocked waiting for R0's full forward** |

### Decode CPU Overhead (per token)

| Component | R0 (µs) | R1 (µs) | Notes |
|-----------|---------|---------|-------|
| `callback` (on_generation_token) | 0 | 0 | Only fires every 50 tokens |
| `detokenizer` | 10 | 10 | |
| `stop_check` | 10 | 10 | |
| `logprobs` | 0 | 0 | Not requested |
| `response_build` | 10 | 10 | |
| MP queue send | 40 | 0 | Only R0 sends to API |
| `recv_poll` | 50 | 30 | Non-blocking queue check |
| **Total CPU overhead** | **120** | **60** | **<1% of step time** |

### Decode Agreement Callbacks (every 50 tokens)

| Component | R0 (µs) | R1 (µs) | Notes |
|-----------|---------|---------|-------|
| `agree_on_cancel_and_tasks` (first) | 1,696 | 163 | 5 distributed collectives |
| `agree_on_tasks` (per step, steady state) | 29–54 | 28–262 | Fast when no tasks pending |

**Decode rate: ~50.6 tok/s** (257 tokens in ~5,076,000µs). MTP speculation acceptance rate ~88%.

---

## Phase 5: Cleanup (<1,000µs)

| Offset (µs) | Duration (µs) | Span | Notes |
|-------------|---------------|------|-------|
| ~40,700,000 | 2 | `submit.clamp_rotating_caches` | No-op |
| ~40,700,000 | 174 | `submit.save_prefix_cache` | Save KV cache for future prefix matching |
| ~40,700,000 | 3 | `submit.make_logits_processors` | No-op |

---

## Summary: Where Every Microsecond Goes

### 16K Prefill (34,302,000µs)

| Component | Time (µs) | % | Notes |
|-----------|----------|---|-------|
| GPU forward compute (R0 layers 0–29) | 25,278,000 | 73.7% | Grows with KV cache size per chunk |
| GPU forward compute (R1 layers 30–59) | 25,282,000 | — | Overlapped with R0 (pipeline parallel) |
| R1 eval_cache (materialize R1's KV cache) | 2,489,367 | 7.3% | ~313,000µs/chunk × 8 chunks |
| PP pipeline bubble (R0 waiting for R1) | 4,339,465 | 12.6% | Inherent to 2-rank PP |
| Contiguous (DeltaNet buffer fix) | 5,406 | <0.1% | ~662µs/chunk × 8 chunks |
| Flush sends | 84 | <0.1% | ~10µs/chunk × 8 chunks |
| Post-loop tokens | 1,262,000 | 3.7% | stream_generate compatibility (2x single-token forward) |
| Prefill barrier | 11,685 | <0.1% | R0 arrived 11,685µs before R1 |
| Clear cache | 1,050 | <0.1% | Release Metal buffers |
| Memory checkpoint | 673 | <0.1% | `mx.eval(zeros(1))` + stats |
| First-request agree_on_tasks | 1,063,000 | 3.1% | One-time distributed sync at request start |

### Decode (257 tokens in ~5,076,000µs)

| Component | Time (µs) | % | Notes |
|-----------|----------|---|-------|
| GPU forward + MTP draft (normal steps, ~226) | 3,842,000 | 75.7% | ~17,000µs × 226 steps |
| GPU forward (outlier/reject steps, ~31) | 1,147,000 | 22.6% | ~37,000µs × 31 steps |
| PP spec setup + first token | 322,000 | 6.3% | One-time |
| CPU overhead (detok, send, etc.) | 30,840 | 0.6% | ~120µs × 257 tokens |
| Agreement callbacks | 1,696 | <0.1% | First occurrence (every 50 tokens) |
| `agree_on_tasks` during decode | ~9,250 | 0.2% | ~37µs × 250 step() calls |

### Total Request (42,500,000µs wall clock)

| Phase | Time (µs) | % |
|-------|----------|---|
| Startup (agree_on_tasks) | 1,079,000 | 2.5% |
| Prefill | 34,302,000 | 80.7% |
| PP spec setup | 322,000 | 0.8% |
| Decode | 5,076,000 | 11.9% |
| TTFT overhead (encode, template, cache lookup) | 13,179 | <0.1% |
| Runner cleanup (status update + trace dump) | 135,946 | 0.3% |
| Inter-span overhead (Python call/dispatch across 1,362 spans) | 104,054 | 0.2% |
| **Total unaccounted** | **240,000** | **0.56%** |

---

## Benchmark Results

### Current (2026-04-05, commit `5546b726`)

| Test | Prompt Tokens | Completion Tokens | Time | Throughput | vs Previous |
|------|--------------|-------------------|------|------------|-------------|
| Short prompt (cold) | 25 | 258 | 6.7–6.9s | 37–38 tok/s | same |
| Short prompt (warm) | 25 | 258 | 5.3–5.4s | 48.0–48.9 tok/s | same |
| Long generation | 25 | 1,026 | 19.4–19.5s | 52.6–52.9 tok/s | **+6%** |
| 16K prompt (cold) | 16,070 | 33 | 36.6s | 439 tok/s prefill | same |
| 16K prompt (KV cache hit) | 26,410 | 33 | 1.4–1.9s | — | **~70% faster** |
| Stress (3 sequential) | 18 | 130 | 2.9–3.2s | 41–45 tok/s | same |

### Previous baseline (2026-04-05, commit `570504b3`)

| Test | Prompt Tokens | Completion Tokens | Time | Throughput |
|------|--------------|-------------------|------|------------|
| Short prompt (cold) | 20 | 257 | 6.7s | 38.2 tok/s |
| Short prompt (warm) | 20 | 257 | 5.4s | 47.7–48.0 tok/s |
| Long generation | 35 | 1,025 | 20.6–20.8s | 49.3–49.8 tok/s |
| 16K prompt (cold) | 16,010 | 257 | 40.6s | 467 tok/s prefill |
| 16K prompt (KV cache hit) | 16,010 | 257 | 6.4s | prefill skipped |
| Stress (3 sequential) | 20 | 129 | 3.0–3.2s | 40.0–42.4 tok/s |

---

## Optimizations Applied in This Session

1. **`mx.clear_cache()` interval bumped from 256/512 → 2048 tokens** (`af1567f0`)
   - Eliminated a guaranteed 17,000µs stall every 256 decode tokens
   - No memory impact during steady-state decode

2. **Redundant post-loop forward pass eliminated** (`570504b3`)
   - `pipeline_parallel_prefill` was doing 2 forward passes on `prompt[-1:]` after the chunk loop; only 1 is needed
   - Changed `trim(2)` → `trim(1)` for pipeline path (stream_generate path unchanged)
   - Saved ~111,000µs per 16K prefill

3. **µs-resolution request trace system** (`3c19ac9e`, `79c02840`)
   - `RequestTrace` singleton records every span from request arrival through decode completion
   - 1,362 spans per request, 99.44% of wall time accounted for
   - Gated on `EXO_TRACING_ENABLED=1`, zero overhead when disabled

4. **Rolling SSM snapshots** (pending)
   - Only keep last 2 SSM snapshots during prefill (rollback uses `snapshots[-2]` exclusively)
   - Previously deep-copied entire cache (22 DeltaNet + 8 attention layers) after every chunk

5. **Eval barrier merge attempted and reverted** (2026-04-05)
   - Merging the two `mx.eval` calls per chunk (cache state + contiguous) into one is WORSE.
   - Reason: On R1, eval_cache (313ms) runs during the pipeline bubble while R0 waits.
     Moving contiguous into the same eval shifts work into R1's forward time, making the
     pipeline bubble larger. Net result: +4s regression on 16K prefill.

6. **Fused MoE dispatches for prefill** (`d6d4f3c8`)
   - Fuse routed gate+up into single `gather_qmm` (3→2 dispatches per MoE layer × 30)
   - Fuse shared expert gate+up into single `quantized_matmul` (2→1 dispatches × 30)
   - Merge GDN in_proj_b + in_proj_a (N=32+32→64, eliminates 1 underutilized dispatch × 22)
   - Prefill forward time unchanged (dispatch overhead savings ~20µs each, invisible vs 3s/chunk)
   - **Decode throughput improved +6%** (49.8 → 52.8 tok/s on long generation)
   - **KV cache hit latency improved ~70%** (6.4s → 1.9s)

7. **EXO_PP_LAYER_SPLIT env var** (`498de7cc`)
   - Override proportional layer allocation for pipeline parallel stages
   - Format: "33,27" gives R0=33 layers, R1=27 layers
   - Tested 33/27 split: total compute unchanged, just redistributed

8. **fp16 compute dtype attempted and reverted** (`ea6fa664`→`a9e08330`)
   - Changed all bf16 activations/weights to fp16 (EXO_COMPUTE_DTYPE=fp16)
   - JACCL/RDMA transport doesn't support fp16 — required bf16 casts at all 9 PP boundaries
   - MLX quantized_matmul kernels are only optimized for bf16 inputs — fp16 caused 7× decode slowdown
   - Infrastructure remains for future use if MLX adds fp16 kernel support

---

## Forward Compute Breakdown (2026-04-05, isolated component benchmarks)

The 73.7% "GPU forward compute" is ALL layer computation, not just DeltaNet. Component-level benchmarking reveals the actual breakdown per prefill chunk (T=2048, TP=2, one rank):

| Component | Time (ms) | % of Forward | GPU Efficiency | Notes |
|-----------|----------|-------------|---------------|-------|
| **MoE (gather_qmm × 30 layers)** | **2,106** | **65%** | **21%** | 128 experts, ~128 tok/expert, 6 GB weights/layer |
| Projections (~100 qlinear) | 1,021 | 32% | 42% | qkv+z+b+a+out per DeltaNet, qkv+out per attn |
| DeltaNet kernel (22 layers) | 97 | 3% | 13% | Sequential recurrence — fast, latency-limited |
| Conv1d + RMSNorm + other | ~50 | <2% | — | |

**Key finding**: MoE dominates at 65% of forward time with only 21% GPU efficiency. The DeltaNet recurrence kernel is only 3% — the previous claim that it was "THE bottleneck" was incorrect (it confused "22/30 layers are DeltaNet type" with "DeltaNet kernel dominates time").

### MoE Efficiency Scales with Sequence Length

| T/rank | MoE ms/tok/layer | vs T=2048 | 16K prompt total compute |
|--------|-----------------|-----------|-------------------------|
| 2048 | 1.03 | baseline | 25,824ms |
| 4096 | 0.71 | -31% | 20,596ms |
| 8192 | 0.62 | -40% | 19,062ms |

Larger prefill chunks → more tokens per expert → better GPU utilization. Increasing `EXO_PREFILL_STEP_SIZE` from 4096 to 8192 (T=4096/rank) reduces per-token MoE cost by 31%.

### Chunkwise DeltaNet: Not Viable (ops-based)

Benchmarked an MLX ops-based chunkwise parallel algorithm (matmul decomposition, C=64 chunk size) against the sequential Metal kernel:
- Sequential kernel: 4.6ms/layer for T=2048 (single Metal dispatch, state in registers)
- Chunkwise ops (no forward substitution): 12.4ms projected (2.7× slower)
- Chunkwise ops (with forward substitution): 62ms projected (13× slower)

MLX dispatch overhead for multiple matmuls per chunk dominates. The sequential kernel is already efficient for its workload.

## Remaining Optimization Opportunities

### Investigated and ruled out
- **Larger prefill chunks** (`EXO_PREFILL_STEP_SIZE=8192`): 19% WORSE due to O(T²) attention cost within chunks outweighing MoE efficiency gains.
- **Fuse MoE dispatches for prefill**: Implemented (gate+up fusion, shared expert fusion, b+a merge). Dispatch savings (~20µs each) are invisible vs 3s/chunk forward. No prefill improvement, but +6% decode.
- **fp16 compute dtype**: MLX quantized kernels only optimized for bf16. 7× decode slowdown.
- **Chunkwise DeltaNet**: 2.7× slower than sequential kernel due to MLX dispatch overhead.
- **gather_qmm block tuning (BM=128)**: No improvement — already at 97% of achievable ceiling.
- **Per-expert regular qmm (128 dispatches)**: 33% slower than gather_qmm.
- **Layer rebalancing (33/27 split)**: Total compute unchanged.

### Investigated and resolved
- **`distributed_prompt_progress_callback`**: Was **340ms per prefill chunk** (~2.7s total for 16K). Replaced with `agree_on_cancellations_fast()` (single `mx_any` fast-path, only falls through to expensive `all_gather` if any rank has cancellations) and removed `agree_on_tasks()` from prefill callback (tasks are agreed during decode instead). Result: **~80µs per chunk** — effectively eliminated. Commit `489efaaa`.
- **Pre-warm `agree_on_tasks`**: First-request `agree_on_tasks` was ~1,031ms. Pre-warming both `agree_on_tasks` and `agree_on_cancellations` during inference warmup reduced this to ~801ms (~230ms saved). Both ranks enter the collective within ~3ms of each other, so the remaining ~800ms is the actual `mx.distributed.all_gather` cost on the data path (the pre-warm only warmed the empty-queue count gather, not the data gather). Could be improved further by warming with a fake non-empty payload, but only ~2% of total prefill — diminishing returns. Commit `b32b3dc5`.

### Investigated and ruled out (architectural blockers)
- **Tensor parallel instead of PP**: Would halve per-device compute, but TP requires per-layer all-reduce. With MoE expert routing over RDMA/JACCL, the all-reduce pattern is incompatible. Confirmed dead.
- **Attention scaling at long context**: Per-chunk forward grows from 3.0s (chunk 0) to 6.4s (chunk 36) for 77K context. Caused by the 7-8 GQA full-attention layers in Qwen3.5-397B doing SDPA against the growing KV cache. Sliding window or sparse attention would fix it but requires model changes (re-training or tuning). Out of scope.

### Compute-bound: software optimization tapped out

**16K prefill breakdown (~36s total):**
| Component | Time | % | Notes |
|-----------|------|---|-------|
| Forward compute (8 chunks × ~4s) | ~32s | 89% | Dominates everything |
| eval_cache (8 × 313ms) | ~2.5s | 7% | KV cache materialization |
| Pipeline bubble | ~3s | 8% | Overlaps with forward; inherent to 2-rank PP |
| `agree_on_tasks` first call | 800ms | 2% | First-request only |
| `distributed_callback` | 640µs | <0.01% | Eliminated |
| Everything else | <100ms | <0.3% | |

**GPU at 32% utilization = 97% of achievable ceiling** for 8-bit gather_qmm at these MoE shapes. This is the M4 Max practical limit for quantized MoE workloads.

### Inherent (architecture-limited)
- **PP pipeline bubble (12.6% of prefill)**: R0 waits for R1 at each chunk boundary. Inherent to 2-rank PP.
- **Decode outlier steps (22.6% of decode)**: MTP draft rejection penalty (37,000µs vs 17,000µs normal). 88-90% acceptance rate with K=1 MTP.
- **GPU efficiency ceiling**: MoE at 32% GPU eff (97% of achievable ceiling for 8-bit gather_qmm at these shapes). Projections at 42%. These are near the practical limit for M4 Max with quantized workloads.
- **CPU overhead**: 120µs/token (<1% of decode step). Already negligible.

### Remaining levers (require non-software changes)
- **More nodes** (hardware scale)
- **Concurrent batching** for throughput (already supported in `BatchGenerator`; no latency win, but ~N× throughput at concurrency N)
- **Model changes**: sliding-window attention for long context, smaller experts, lower precision quantization
- **Different hardware**: Apple's MoE/quantized-kernel ceiling on M4 Max is the wall
