# Request Lifecycle Trace — Qwen3.5-397B-A17B-4bit on 2x M4 Max (PP/RDMA)

**Date:** 2026-04-05
**Commit:** `19d8081d`
**Test:** 16,010 prompt tokens → 257 decode tokens, 42,500ms wall clock
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
| Unaccounted | 1,721,821 | 4.1% |

### Unaccounted Time Analysis (1,721,821µs / 4.1%)

The "unaccounted" time is distributed across:
- Python function call overhead between traced spans (~50µs per span × ~300 spans ≈ 15,000µs)
- `batch_gen.step()` framework (task queue management, output parsing) per decode token (~30µs × 257 ≈ 7,710µs)
- Runner loop overhead (match/dispatch, finished list management) per decode token (~50µs × 257 ≈ 12,850µs)
- Trailing distributed callback after last prefill chunk: 3,419,000µs on R0 (pipeline drain)
- The trailing callback is counted in prefill but the wall-clock stagger between R0 finishing prefill and R1 finishing prefill contributes to the gap
- Time between `pp_spec.decode_loop_start` marker and first actual decode step (~80,000µs warmup)
