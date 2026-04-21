# DFlash Speculative Decoding on Qwen3.5 Attn/MoE Split — Pipeline Reference

Navigation aid for reading and debugging this code path. Every step has a `file:line` so it can be opened directly. Written for Qwen3.5 (both dense 27B and MoE variants) running with `world_size==2` on JACCL.

All paths below are relative to `exo/src/exo/worker/engines/mlx/` unless noted otherwise.

---

## 1. Executive summary

- **Two ranks.** `ATTN_RANK=0` runs attention (GQA `self_attn` and linear `linear_attn`). `MOE_RANK=1` runs the MLP / MoE block. Ranks are defined in `patches/qwen3_5_moe_split/decoder.py:16`.
- **Two independent layers of patches:**
  - **Layer A (structural).** Replaces class methods so different ranks do different work. All installs in `patches/qwen3_5_moe_split/apply.py:220–294`.
  - **Layer B (kernel).** Instance-level swaps of each projection's `__call__` with a dynamic LpB kernel picker. Target: `patches/qwen3_5/lpb_patch.py:102`. Drafter: `speculative/bf16_lpb_patch.py:99`.
- **Three phases per request.** Prefill (S > 1) → speculative cycles (S = V+1, one-shot per cycle) → termination (stop token or length cap).
- **Drafter lives on both ranks.** Both ranks run the drafter in parallel; drafts are synced via `all_gather` and MOE_RANK's drafts win.
- **Rollback runs on both ranks.** Same `n_accepted` on both (deterministic at temp=0; broadcast at temp>0). ATTN_RANK uses the real speculative GDN states; MOE_RANK's `cache[1]` stays `None` because it never runs `linear_attn`.

---

## 2. Architecture overview

```
                 JACCL group (world_size=2)
                  ▲                    ▲
                  │ all_gather         │ all_gather
                  │                    │
  ┌───────────────┴─────────┐  ┌───────┴────────────────┐
  │  ATTN_RANK (rank 0)     │  │  MOE_RANK (rank 1)     │
  │  ─────────────────────  │  │  ─────────────────────  │
  │  Target model:          │  │  Target model:          │
  │   - self_attn (GQA)     │  │   - mlp (dense/MoE)     │
  │   - linear_attn (GDN)   │  │   - post_attention_ln   │
  │   - input_layernorm     │  │   - input_layernorm†    │
  │   - KV cache real       │  │   - KV cache empty*     │
  │   - GDN state real      │  │   - GDN state empty*    │
  │                         │  │                         │
  │  Drafter (full copy)    │  │  Drafter (full copy)    │
  │   - 5 layers, KVCache   │  │   - 5 layers, KVCache   │
  └─────────────────────────┘  └─────────────────────────┘

  † kept under EXO_SPECULATIVE=1 (used for GDN conv_input reconstruction)
  * rank-specific cache shims in apply.py:33 make empty-cache ops safe
```

**Draft-position authority.** ATTN_RANK's `cache.offset` is the truth — it runs attention and knows the real position. MOE_RANK's `cache.offset` is always 0. Every speculative cycle opens with an `all_gather` of `_draft_position` that overwrites MOE_RANK's copy (`dflash_split.py:65–70`).

---

## 3. Patch inventory

### 3a. Layer A — structural patches

Installed in order by `apply_qwen35_attn_moe_split_patches` (`apply.py:220`). Invoked from `auto_parallel.py:474` under `attn_moe_split_auto_parallel`, gated by `isinstance(inner, Qwen3_5TextModelInner)` and `world_size==2`.

| # | Patched symbol | Replacement | Install site | Replacement body | Purpose |
|---|----------------|-------------|--------------|------------------|---------|
| 1 | `DecoderLayer.__call__` | `_split_call` | `apply.py:250` | `decoder.py:36` | Serial S==1 split, one all_gather per sub-step |
| 2 | `Qwen3_5TextModel.__call__` | `_pipelined_call` | `apply.py:253` | `model_forward.py:430` | S>1 routes through `pipelined_layer_loop`; S==1 falls through to stock loop → `_split_call` |
| 3 | `mtp_module.speculative_forward` | pipelined variant | `apply.py:259` | `model_forward.py:491` (factory), `:499` (body) | MTP speculative path (not used under `SPECULATIVE_MODE=dflash`) |
| 4 | `dflash_speculative.dflash_speculative_forward` | `_pipelined_dflash_forward` | `apply.py:270` | `model_forward.py:615` (factory), `:623` (body) | DFlash verify forward over the pipeline; captures target-layer hiddens + GDN speculative states |
| 5 | `DFlashBatchGenerator._speculative_next` | `_split_speculative_next` | `apply.py:279` | `dflash_split.py:29` | Draft/verify/accept cycle with cross-rank syncs |
| 6 | Cache shims on MOE_RANK | many | `apply.py:284` | `apply.py:33–211` | Make `state` / `extract` / `filter` / `extend` safe on unpopulated caches |
| 7 | Weight dropping | `_drop_unused_weights` | `apply.py:288` | `apply.py:297` | Only runs when `EXO_SPECULATIVE!=1`; speculative keeps both sides' weights |

### 3b. Layer B — kernel patches

| # | Target | Function | Body | Notes |
|---|--------|----------|------|-------|
| 1 | Target model projections | `apply_lpb_patches(model)` | `patches/qwen3_5/lpb_patch.py:102` | Called from `apply.py:245` under `EXO_LPB_PATCHES=1`. Patches per-layer `mlp.{gate,up,down}_proj`, attn `{q,k,v,o}_proj` or `{in_proj_qkv,in_proj_z,out_proj}`, and `lm_head`. Instance-level wrapping via `setattr(parent, proj_name, _LpB*Linear(...))`. |
| 2 | Drafter projections | `apply_bf16_lpb_patches(drafter)` | `speculative/bf16_lpb_patch.py:99` | Called from `generator/batch_generate.py:117`. Patches per-layer `mlp_{gate,up,down}` + `self_attn.{q,k,v,o}_proj` + `drafter.fc` + `drafter.lm_head`. `DFLASH_LPB_ONLY=q_proj,k_proj,...` bisect knob at `bf16_lpb_patch.py:106`. |
| 3 | `gated_delta_update` | `_make_speculative_gdu(spec_all_states)` | `speculative/mtp_module.py:126` | Temporary swap on `mlx_lm.models.qwen3_5.gated_delta_update`. Installed at verify entry (`model_forward.py:667`), restored post-loop (`model_forward.py:732`). Captures per-step recurrent states for rollback. |

The LpB wrappers memoize `M → kernel_fn` per projection. Kernel selection logic lives in `matmul/patches/kernel_picker.py:33` (`pick_bf16_kernel`) and `:69` (`pick_int8_kernel`). `MAX_M=16` in both LpB patches; M>16 falls back to the original `__call__`.

### 3c. Target-layer capture (stock DFlash, no bypass)

Stock `DFlashBatchGenerator._setup_hidden_capture` (`speculative/dflash_batch_generator.py:53`) wraps target-layer indices in a `_CapturingLayer`. Its `__call__` is what populates `self._captured['{layer,prefill}_hiddens']`. Our pipelined path **never calls `layer.__call__`** — it dispatches `layer.self_attn` / `layer.linear_attn` / `layer.mlp` directly. Compensation: `_populate_dflash_captured` (`model_forward.py:387`) reaches into `_CapturingLayer`'s closure at the end of `_pipelined_call` to write the dict DFlash expects. See commit `b1a5dccf`.

---

## 4. Execution flow by phase

### 4a. Startup & warmup

1. Model loads via `auto_parallel.attn_moe_split_auto_parallel` (`auto_parallel.py:436`), which calls `apply_qwen35_attn_moe_split_patches(model, group)` (`auto_parallel.py:474`). Both ranks run patch install; differences are gated by `group.rank()`.
   > *Prints:* `Patched X target projections with dynamic LpB` (`lpb_patch.py:130`) if `EXO_LPB_PATCHES!=0`; `Qwen3.5 attn/moe split patch applied on rank N/2` (`apply.py:291`).
2. `BatchGenerator` init reads `EXO_SPECULATIVE=1 EXO_SPECULATIVE_MODE=dflash` (`batch_generate.py:100`).
3. Drafter is instantiated: `DFlashDrafter(self.model, dflash_path)` (`batch_generate.py:116`). See `dflash_module.py:107` — weights loaded via `huggingface_hub.snapshot_download` if not cached.
   > *Prints:* `DFlash config: N layers, hidden=..., heads=.../..., block=...` (`dflash_module.py:133`); `Target layers: [...], mask_token=...` (`:136`) — **this is the authoritative list of `target_layer_ids`**; `DFlash loaded: X tensors, Y.YM params` (`:164`).
4. Drafter is LpB-patched: `apply_bf16_lpb_patches(drafter)` (`batch_generate.py:117`).
   > *Prints:* `DFLASH_LPB_ONLY=...` only if that env var is set (`bf16_lpb_patch.py:109`); `Patched X DFlash drafter projections with dynamic LpB` (`:143`).
5. `DFlashBatchGenerator.__init__` (`dflash_batch_generator.py:23`) runs `_setup_hidden_capture` (`dflash_batch_generator.py:53`) — wraps each index in `drafter.target_layer_ids` with `_CapturingLayer`.
6. **Warmup** — `warmup_dflash` (`batch_generate.py:272`) sweeps drafter projections across every M they'll see: `S_ctx ∈ [1, V+1]` × `block_size`. Then one target verify at `M=V+1` to compile every target projection. Without this, Metal kernel compilation stalls the first real batch. Warmup also runs one non-speculative `dflash_speculative_forward` (`batch_generate.py:311`) for target-only kernel priming, then rolls back the cache.
   > *Prints:* bookends `Warming up DFlash speculative decoding kernels...` (`batch_generate.py:298`) and `DFlash warmup complete` (`:357`). Each pipelined forward inside warmup emits the full `[rank N] pipeline begin ...` + per-stage stream from `model_forward.py:212, 225, 268, 306, 331`.
   > *Evals:* `mx.eval(target_hidden_full, logits)` (`:313`), `mx.eval(dl)` per drafter sweep iteration (`:332`), `mx.eval(target_hidden, vl)` after the verify forward (`:341`).

### 4b. Prefill (first step for a uid)

Entry: `BatchGenerator.step` → `DFlashBatchGenerator._next` (`dflash_batch_generator.py:92`).

1. `_next` sees uid ∉ `_prefilled`, routes to `_first_step_capture` (`dflash_batch_generator.py:119`).
2. `super()._next()` → stock `BatchGenerator._next` processes the prompt, calls `self.model(prompt_tokens, cache)`.
3. That reaches our patched `Qwen3_5TextModel.__call__` = `_pipelined_call` (`model_forward.py:430`). S = prompt_len > 1, so it takes the pipelined branch (`:456–481`):
   1. `_dflash_capturing_target_ids(self)` (`model_forward.py:353`) detects the `_CapturingLayer` wrappers.
   2. `pipelined_layer_loop(..., capture_layers=target_ids)` (`model_forward.py:146`) runs the 2N+1 stage pipeline — see section 4c step 7.5 for the full stage-by-stage print/eval inventory. Same structure applies here at `S = prompt_len`; whether the single-gather fast path or the two-gather slow path fires depends on `prompt_len % 2`.
   3. `_populate_dflash_captured(self, layer_hiddens, S)` (`model_forward.py:387`) writes `_captured['layer_hiddens']` and `_captured['prefill_hiddens']` via closure introspection.
4. Back in `super()._next()`: `inner.norm` → `lm_head` (LpB wrapper, M = prompt_len, typically > 16 → fallback to stock GEMM). First token sampled.
5. Back in `_first_step_capture`: reads `_captured['prefill_hiddens']` → `_last_target_hidden[uid]` shape `(1, prompt_len, D·|target_layers|)`. Seeds `_draft_position[uid]` from `cache.offset` on one of the caches.
   > *Evals:* `mx.eval(target_hidden)` (`dflash_batch_generator.py:125`) before caching — forces the concatenated prefill hiddens to materialize.
6. Falls into `_speculative_next` (first real cycle).

### 4c. Speculative cycle (per step)

Entry: `DFlashBatchGenerator._speculative_next` — patched to `_split_speculative_next` in `dflash_split.py:29`. Both ranks run identical code unless noted.

Numbered per the source, cross-referenced with `dflash_split.py` line numbers.

1. **Preamble** (`:34–55`). Record `tic`, append `y_val` to `batch.tokens[0]`, read `_last_target_hidden[uid]`. If missing, fall back to `super()._next()` — this hits the S==1 decode path (section 4d). Logs `DFlash speculative cycle (...)` and any fallback.
   > *Prints:* either `[rank N] DFlash: NO target_hidden -> fallback ... (y_val=...)` (`dflash_split.py:46`) OR `[rank N] DFlash speculative cycle (y_val=..., target_hidden.shape=...)` (`:51`) — one per cycle per rank.
2. **Draft-position sync** (`:65–70`). `all_gather(_draft_position)` → take `ATTN_RANK`'s. ATTN_RANK's cache offset is authoritative; MOE_RANK's stays at 0 because it never runs attention.
3. **Draft** (`:73–76`). Both ranks: `drafter.draft(last_target_hidden, block_ids, start)` → `draft_logits` of shape `(B, block_size-1, vocab)`. `block_ids` is `[y_val, MASK, MASK, ...]` of length `block_size`. The drafter implementation is `dflash_module.py:226` — see notes at the bottom of section 4c.
4. **Sample** (`:79–91`). temp=0: `argmax` of `draft_logits`. temp>0: per-position `mx.random.categorical` with softmax. Both ranks sample **independently** (RNG seeds may diverge).
   > *Evals:* `mx.eval(all_drafts_arr)` (`dflash_split.py:81`) before `.tolist()` on temp=0 path; temp>0 uses `.item()` per position which evals inline.
5. **Draft sync** (`:94–98`). `all_gather(drafts_local)` → take MOE_RANK's `drafts_arr`. After this, both ranks have identical `drafts` of length `verify_len`.
   > *Evals:* `mx.eval(drafts_arr)` (`dflash_split.py:97`) before `.tolist()` — required because the list is used Python-side to build `verify_input`.
6. **Build verify input** (`:101–102`). `verify_input = concat([[y_val]], drafts_arr)`, shape `(1, V+1)`.
7. **Pipelined verify forward** (`:106–112`). `dflash_speculative_forward(model, verify_input, cache, target_layer_ids, speculative=True)` → our `_pipelined_dflash_forward` (`model_forward.py:623`):
   1. Wrap GDN caches in `SpeculativeArraysCache` (`speculative_cache.py:15`).
   2. Monkey-patch `gated_delta_update` → `_make_speculative_gdu(spec_all_states)` (`mtp_module.py:126`, installed at `model_forward.py:667`). The speculative kernel returns `(y, state_out, all_states)` and appends `all_states` to the closure list. Shape: `(B, T, H_v, D_v, D_k)`.
   3. Collect GDN pre-loop data: for each `is_linear` layer, take `spec_cache[0]` (existing conv state) or zeros, store `(pre_conv, c, layer, idx)`.
   4. Compute `capture_set = target_layer_ids ∪ {L-1 : L is GDN, L ≥ 1}` (`model_forward.py:707`). The extra `L-1` outputs are needed to reconstruct conv_input post-loop.
   5. Run `pipelined_layer_loop(..., capture_layers=capture_set)`. Stages inside the loop:
      - **Stage 0 (startup)** — ATTN_RANK: `attn_0(H0)`; MOE_RANK: idle placeholder. 1 all_gather. (`model_forward.py:214–228`)
      - **Stages 1..2N-1 (main)** — alternating B/A:
        - B (odd stage, `T = stage//2`): ATTN runs `attn_T(H1)`, MOE runs `moe_T(h_T_H0)`. Even S → 1 all_gather, odd S → 2. (`:236–275`)
        - A (even stage, T≥1): ATTN runs `attn_T(H0)`, MOE runs `moe_{T-1}(h_{T-1}_H1)`. Capture of layer T-1 at A-stage end: `capture[T-1] = concat(x_H0, x_H1)` (`:318–319`). 1 or 2 all_gathers.
      - **Stage 2N (drain)** — ATTN idle, MOE `moe_{N-1}(h_{N-1}_H1)`. 1 all_gather. (`:322–334`)
      - Capture of layer N-1 happens post-drain (`:342–343`).
      - Total collectives: even S → 2N+1; odd S → 4N-1.
      - Per-stage `mx.eval` calls (`:221`, `:249`, `:262` etc.) break MLX's graph accumulation and keep JACCL's queue drained.

      > *Prints per stage:* `[rank N] pipeline begin S=V+1 N=64 even=<bool>` at loop entry (`model_forward.py:212`), then `[rank N] stage 0 (T=0 startup) h_H0.mean=...` (`:225`), per B-stage `[rank N] stage K (T=... B) h_H1.mean=... out_H0.mean=...` (`:268`), per A-stage `[rank N] stage K (T=... A) h_H0.mean=... out_H1.mean=...` (`:306`), and `[rank N] drain out_H1.mean=...` (`:331`). Total = 1 + 1 + (2N−1) + 1 = 2N+1 print lines per rank per pipelined forward.
      > *Evals per stage (graph-break + JACCL queue bound):* startup — `mx.eval(contribution)` (`:221`), `mx.eval(gathered)` (`:223`). Main-loop B stage — even S: `mx.eval(my_out)` (`:249`), `mx.eval(gathered)` (`:251`). Odd S: `mx.eval(attn_side)` (`:261`), `mx.eval(moe_side)` (`:262`), `mx.eval(attn_contrib)` (`:266`), `mx.eval(moe_contrib)` (`:267`). Main-loop A stage — even S: `mx.eval(my_out)` (`:287`), `mx.eval(gathered)` (`:289`). Odd S: `mx.eval(attn_side)` (`:299`), `mx.eval(moe_side)` (`:300`), `mx.eval(attn_contrib)` (`:304`), `mx.eval(moe_contrib)` (`:305`). Drain — `mx.eval(contribution)` (`:327`), `mx.eval(gathered)` (`:329`).
      > *Async / no eval:* captured tensors written to the `capture` dict (`:319`, `:343`) and the final `concat([out_H0, out_H1])` (`:339`) are **not** eval'd here — they materialize implicitly when post-loop reconstruction reads them (step 7.7 / 7.8) and when `lm_head` runs in step 7.10.
   6. Restore stock `gated_delta_update` (`:732`).
   7. **Merge H0+H1 speculative states** (`:741–752`). `spec_all_states` has `2 × len(gdn_spec_data)` entries (one per half per layer). Concat consecutive pairs along step dim → per-layer `all_states`. On MOE_RANK `spec_all_states` is empty because the monkey-patched GDU never ran there → `merged_states == []`.
   8. **Reconstruct conv_input per GDN layer** (`:754–788`). `layer_input = initial_embed` (for layer 0) or `layer_hiddens[layer_idx - 1]` (captured by the pipeline). Then `normed = input_layernorm(layer_input)`, `qkv = in_proj_qkv(normed)`, and `spec_cache.conv_input = concat([pre_conv, qkv], axis=1)`. On MOE_RANK we keep `input_layernorm` and `linear_attn` under `EXO_SPECULATIVE=1`, so this runs there too.
   9. Concat captured target hiddens → `target_hidden` of shape `(1, V+1, D·|target_layers|)` (`:791`).
   10. Final norm + lm_head (LpB wrapper, M = V+1 — LpB fires) → `verify_logits` of shape `(1, V+1, vocab)`.
8. **Acceptance** (`:116–162`). temp=0: `argmax(verify_logits[:, :V, :]) == drafts_arr` → `matches`, count leading True → `n_accepted`. temp>0: MOE_RANK computes acceptance ratios + uniforms; `all_gather([n_accepted_local])` broadcasts MOE's value.
   > *Async evals:* temp=0 — `mx.async_eval(matches, all_next, target_hidden)` (`dflash_split.py:120`) kicks off acceptance + next-step state; the Python `for i in range(V): if matches[i].item(): ...` loop blocks per-index on the already-scheduled compute. temp>0 — `mx.async_eval(accept_ratios, uniforms, corrections, bonus_token, target_hidden)` (`:146–148`).
   > *Prints:* `[DFlash] n_accepted=X/V` on MOE_RANK only (`:165`) — one line per cycle.
9. **Rollback** (`:171–177`). `rollback = V - n_accepted`. If > 0: GQA `BatchKVCache` → `c.offset -= rollback`. GDN `SpeculativeArraysCache` → `c.rollback(n_accepted)` (`speculative_cache.py:81`): sets `base.cache[1] = all_states[0, n_accepted]` if `all_states` is set (ATTN_RANK only), and `base.cache[0] = conv_input[:, n_accepted+1 : n_accepted+1+3, :]` (both ranks).
10. **Unwrap speculative caches** (`:179–181`). Replace each `SpeculativeArraysCache` with its `.base` for the next cycle's stock layer dispatch.
11. **Emit tokens** (`:184–212`). Bonus/correction token (all accepted: `all_next[V]` or `bonus_token`; partial: `all_next[n_accepted]` or `corrections[n_accepted]`). Update `_last_target_hidden[uid] = target_hidden[:, :n_accepted+1, :]`, advance `_draft_position += n_accepted+1`. Buffer accepted draft tokens into `_token_buffer[uid]` (`:234` or `:248`).
   > *Async evals:* `mx.async_eval(batch.y)` at `dflash_split.py:239` (stop-token path) or `:250` (normal return) — schedules the next-step bonus token without blocking the `Response` yield.

**Notes on the drafter.** `DFlashDrafter.draft` (`dflash_module.py:226`):
- `target_hidden` shape `(B, accepted_len, n_layers · hidden_size)` — compressed via `self.fc` (`dflash_module.py:243`) to `(B, accepted_len, hidden_size)`.
- `start` = `_draft_position[uid]` = prompt_len on first cycle, then `start + n_accepted + 1` each cycle.
- K/V are formed from `concat(target_hidden, draft_input)` (`dflash_module.py:50`), so `k_proj/v_proj` see `M = accepted_len + block_size`. On the first cycle `accepted_len = prompt_len`, so these projections usually fall back to stock GEMM. After the first cycle `accepted_len ≤ V+1`, so LpB fires at `M = V+1 + block_size` (if ≤ 16).

### 4d. Buffered drain and S==1 fallback decode

- `DFlashBatchGenerator._next` (`dflash_batch_generator.py:92`) yields from `_token_buffer[uid]` first via `_yield_buffered` (`:341`) — one token per call.
- Once the buffer is empty and a new forward is needed, `_speculative_next` fires again.
- If `_last_target_hidden[uid]` is missing (edge case — e.g. dropped between requests), `_split_speculative_next` calls `super()._next()`. That reaches `self.model(y, cache)` with `S=1` → `_pipelined_call` S==1 branch (`model_forward.py:446–454`) → stock layer loop → `DecoderLayer.__call__` → `_split_call` (`decoder.py:36`):
  - Step 1: ATTN runs `self_attn` or `linear_attn` on input; MOE burns `DUMMY_LN_ITERS=50` layernorms to keep JACCL balanced, evaled every 2 layers (`decoder.py:52`).
  - 1st all_gather picks ATTN's output.
  - Step 2: MOE runs MLP; ATTN burns layernorms.
  - 2nd all_gather picks MOE's output.
  - Two `all_gather`s per layer, N layers → `2N` total collectives for an S==1 decode.
  > *Prints:* per layer per rank — `[rank N] L=lc after gather-1 h.mean=...` (`decoder.py:56`) and `[rank N] L=lc after gather-2 out.mean=...` (`:69`). 2·N lines per decode step per rank.
  > *Evals:* per layer — `mx.eval(h)` (`:53`) on the idle rank every 2nd `lc` (JACCL queue bound); `mx.eval(h)` post-gather-1 (`:55`); `mx.eval(out)` (`:66`) idle-rank equivalent; `mx.eval(result)` post-gather-2 (`:68`).
- At M=1 every LpB wrapper fires its fast-path kernel.

### 4e. Termination

- `_yield_buffered` (`dflash_batch_generator.py:341`) drains `_token_buffer[uid]` one token at a time.
- Detects `finish_reason = "stop"` (`:348`) or `"length"` (`:350`).
- On finish: `cache = batch.extract_cache(0)` (`:356`) — this iterates `[c.extract(idx) for c in batch.cache]`.
  - `BatchKVCache.extract` → our `_bkv_extract` (`apply.py:165`): returns empty `KVCache` if `self.keys is None`.
  - `ArraysCache.extract` → our `_arrays_extract` (`apply.py:130`): builds a new `ArraysCache` preserving `None` per-element (`c[idx:idx+1] if c is not None else None`). Required because MOE_RANK's GDN `base.cache[1]` stays `None` after rollback (see section 6).

---

## 5. Kernel dispatch matrix

Rows = dispatch site. Columns = phase / M seen at call time. Cell = kernel actually dispatched. "stock" means MLX built-in GEMM via the original `nn.Linear.__call__`.

| Site | Prefill (M = prompt_len) | Verify (M = V+1) | Decode (M = 1) | Drafter first cycle | Drafter steady state |
|------|--------------------------|------------------|----------------|---------------------|----------------------|
| target `q/k/v/o_proj` | stock | LpB | LpB | — | — |
| target `mlp.{gate,up,down}_proj` | stock | LpB | LpB | — | — |
| target `linear_attn.{in_proj_*, out_proj}` | stock | LpB | LpB | — | — |
| target `gated_delta_update` | stock | **speculative** (captures all_states) | stock | — | — |
| target `lm_head` | stock | LpB | LpB | — | — |
| drafter `q_proj` | — | — | — | LpB (M=BS) | LpB (M=BS) |
| drafter `k_proj, v_proj` | — | — | — | stock (M=prompt_len+BS, typically >16) | LpB (M=accepted_len+BS) |
| drafter `o_proj` | — | — | — | LpB (M=BS) | LpB (M=BS) |
| drafter `mlp_{gate,up,down}` | — | — | — | LpB (M=BS) | LpB (M=BS) |
| drafter `fc` | — | — | — | stock (M=prompt_len, >16) | LpB (M=accepted_len ≤ V+1) |
| drafter `lm_head` | — | — | — | LpB (M=BS-1) | LpB (M=BS-1) |

**LpB kernel picker.** `pick_bf16_kernel` (`matmul/patches/kernel_picker.py:33`) / `pick_int8_kernel` (`:69`). Rounds M to `{1, 2, 4, 8, 12, 16, 32, 64}`. Decision summary:

- bf16, N > 50000 (lm_head): `lpb` (M ≤ 8) or `lpb_twice`.
- bf16, M_rnd ≤ 8: `lpb`.
- bf16, M_rnd = 12: `lpb_twice` if max(N, K) ≤ 4096 else `sk_steel16`.
- bf16, M_rnd = 16: `sk_steel16`.
- bf16, M_rnd ∈ {32, 64}: `sk_steel32`.
- int8, M_rnd ≤ 8: `lpb`.
- int8, M_rnd = 12 or 16: `qsk16`.
- int8, M_rnd ∈ {32, 64}: `qsk32`.

Note: `MAX_M=16` in both LpB patches means branches for `M_rnd ∈ {32, 64}` are never hit on this path — the picker supports them, the wrappers don't call at those M.

---

## 6. Invariants

### Shapes

- `_last_target_hidden[uid]`: `(1, k, D · |target_layer_ids|)`.
  - `k = prompt_len` right after `_first_step_capture`.
  - `k = n_accepted + 1` after each `_speculative_next` (`dflash_split.py:197`).
- `block_ids`: `(1, block_size)` with `block_ids[0, 0] = y_val`, rest `mask_token_id`.
- `verify_input`: `(1, V+1)`.
- `target_hidden` returned from verify: `(1, V+1, D · |target_layer_ids|)`.
- `verify_logits`: `(1, V+1, vocab)`.
- GDN `SpeculativeArraysCache.all_states`: `(B, V+1, H_v, D_v, D_k)` after the H0/H1 merge. `SpeculativeArraysCache.conv_input`: `(B, 3 + V+1, conv_dim)`.

### Cache offsets

- `BatchKVCache.offset == BatchKVCache._idx - left_padding`. Masks must slice against `_idx`, not `offset` (the pipelined loop uses `_idx` at `model_forward.py:185`).
- Rollback: `BatchKVCache.offset -= (V - n_accepted)` decrements both `offset` and `_idx` via the cache API. `ArraysCache` rollback rewrites `cache[0]` and `cache[1]` from speculative state.
- GDN base cache entries on MOE_RANK: `cache[0]` gets reconstructed by `_pipelined_dflash_forward`, `cache[1]` stays `None` because the speculative GDU never ran there. Downstream code must tolerate this — `_arrays_extract` does (`apply.py:130`).

### Stage indexing in `pipelined_layer_loop` (`model_forward.py:146`)

- Stage range: `0` (startup) + `1..2N-1` (main loop) + `2N` (drain) = `2N+1` stages.
- `T = stage // 2`. B-stage = odd stage (runs `attn_T(H1)` + `moe_T(h_T_H0)`). A-stage = even stage (runs `attn_T(H0)` + `moe_{T-1}(h_{T-1}_H1)`).
- Capture of layer L happens at the start of the A-stage for T = L+1 (i.e. stage `2(L+1)`), so `capture[L]` is set when we enter A-stage for layer L+1. Layer N-1 is captured post-drain (`model_forward.py:342`).
- Even S: 1 all_gather per stage. Odd S: 2 all_gathers per stage (`_gather_two` at `model_forward.py:110`).

### Draft-position authority

- ATTN_RANK's `cache.offset` is authoritative throughout.
- `_split_speculative_next` begins with `all_gather([_draft_position])` and takes index `ATTN_RANK` (`dflash_split.py:69`).

---

## 7. Environment variables

| Variable | Default | File:line | Effect |
|----------|---------|-----------|--------|
| `EXO_SPECULATIVE` | `"0"` | `batch_generate.py:100`, `apply.py:288` | Enable speculative decode. Also gates `_drop_unused_weights` — speculative keeps all weights on both ranks. |
| `EXO_SPECULATIVE_MODE` | `"mtp"` | `batch_generate.py:101` | `"dflash"` for this path, `"mtp"` for MTP. |
| `EXO_SPECULATIVE_TEMP` | `"0.7"` | `batch_generate.py:103` | Sampling temperature for the drafter; 0 means greedy/deterministic. |
| `EXO_SPECULATIVE_ALPHA` | `"1.0"` | `batch_generate.py:104` | Acceptance ratio exponent α (temp>0 only). |
| `EXO_SPECULATIVE_GAMMA` | `"2"` | `batch_generate.py:141` | MTP draft chain length. Not used on DFlash path. |
| `EXO_DFLASH_MODEL` | `"z-lab/Qwen3.5-27B-DFlash"` | `batch_generate.py:112` | HF repo for drafter weights. Pre-pull with `huggingface-cli download ...` to avoid first-run download. |
| `EXO_DFLASH_VERIFY` | `"5"` | `batch_generate.py:113` | V — number of drafts per cycle. `verify_input.shape = (1, V+1)`. Even `V+1` → single-all_gather fast path in `pipelined_layer_loop`. |
| `EXO_DFLASH_BLOCK_SIZE` | `"6"` | `batch_generate.py:114` | Drafter block size BS. Draft produces `BS-1` logits. |
| `EXO_MTP_WEIGHTS` | `""` | `batch_generate.py:169` | Explicit MTP weights path. Not used on DFlash path. |
| `EXO_MTP_MODEL` | `""` | `batch_generate.py:173` | HF repo for MTP extraction. Not used on DFlash path. |
| `EXO_LPB_PATCHES` | `"1"` | `apply.py:243` | Toggle target-side LpB kernel swaps. `"0"` disables. |
| `EXO_DISABLE_LOGPROBS` | `"0"` | `batch_generate.py:618` | Skip logprob computation (some speculative paths require this). |
| `DFLASH_LPB_ONLY` | `""` | `bf16_lpb_patch.py:106` | Comma-separated projection names (`mlp_gate,mlp_up,...,lm_head`) to restrict drafter LpB patching. Debugging bisect knob. |

---

## 8. Debugging guide

### 8a. Existing prints

Init-time (once per process):

| Print | Source | Notes |
|-------|--------|-------|
| `DFlash config: ...` | `dflash_module.py:133` | Drafter arch summary |
| `Target layers: [...], mask_token=...` | `dflash_module.py:136` | **Exact `target_layer_ids` the drafter expects** |
| `DFlash loaded: N tensors, M params` | `dflash_module.py:164` | Drafter weight load confirmation |
| `Patched X target projections with dynamic LpB` | `lpb_patch.py:130` | Target-side LpB swap count |
| `Patched X DFlash drafter projections with dynamic LpB` | `bf16_lpb_patch.py:143` | Drafter-side LpB swap count |
| `DFLASH_LPB_ONLY=...` | `bf16_lpb_patch.py:109` | Only if the bisect env var is set |
| `Qwen3.5 attn/moe split patch applied on rank N/2` | `apply.py:291` | Structural patches done |
| `Warming up DFlash speculative decoding kernels...` / `DFlash warmup complete` | `batch_generate.py:298, 357` | Kernel warmup bookends |

Per forward / per cycle (hot path — noisy, disable once stable):

| Print | Source | When |
|-------|--------|------|
| `[rank N] pipeline begin S=... N=... even=...` | `model_forward.py:212` | Once per pipelined forward (prefill + verify + warmup verify) |
| `[rank N] stage 0 (T=0 startup) ...` | `model_forward.py:225` | Startup stage of pipelined loop |
| `[rank N] stage K (T=... B) h_H1.mean=... out_H0.mean=...` | `model_forward.py:268` | Per B stage in main loop |
| `[rank N] stage K (T=... A) h_H0.mean=... out_H1.mean=...` | `model_forward.py:306` | Per A stage in main loop |
| `[rank N] drain out_H1.mean=...` | `model_forward.py:331` | Drain stage of pipelined loop |
| `[rank N] L=lc after gather-1 h.mean=...` | `decoder.py:56` | Per layer, mid-`_split_call` (S==1 decode) |
| `[rank N] L=lc after gather-2 out.mean=...` | `decoder.py:69` | Per layer, end of `_split_call` |
| `[rank N] DFlash speculative cycle (y_val=..., target_hidden.shape=...)` | `dflash_split.py:51` | Once per speculative cycle |
| `[rank N] DFlash: NO target_hidden -> fallback ...` | `dflash_split.py:46` | When `_last_target_hidden[uid]` is missing |
| `[DFlash] n_accepted=X/V` | `dflash_split.py:165` | MOE_RANK only, end of cycle |

### 8b. `mx.eval` / `mx.async_eval` inventory

Evals serve three distinct purposes in this pipeline, and removing the wrong one
can silently break correctness or JACCL flow control. Classification:

**(i) Pipeline graph-break evals.** Force the MLX graph to materialize between
stages of `pipelined_layer_loop` and between gathers of the S==1 `_split_call`.
Without these, MLX would fuse arbitrarily many stages into one graph, blowing
up memory and (more importantly) desynchronizing `all_gather` order with
JACCL's internal queue (`MAX_SEND_WR=32`). Keep these unless you're
intentionally testing graph fusion.

| File:line | Target | Location in schedule |
|-----------|--------|----------------------|
| `model_forward.py:221` | `contribution` | Before startup gather |
| `model_forward.py:223` | `gathered` | After startup gather |
| `model_forward.py:249, 251` | `my_out`, `gathered` | B stage, even-S (single gather) |
| `model_forward.py:261, 262` | `attn_side`, `moe_side` | B stage, odd-S (pre two-gather) |
| `model_forward.py:266, 267` | `attn_contrib`, `moe_contrib` | B stage, odd-S (post two-gather) |
| `model_forward.py:287, 289` | `my_out`, `gathered` | A stage, even-S |
| `model_forward.py:299, 300` | `attn_side`, `moe_side` | A stage, odd-S (pre) |
| `model_forward.py:304, 305` | `attn_contrib`, `moe_contrib` | A stage, odd-S (post) |
| `model_forward.py:327, 329` | `contribution`, `gathered` | Drain stage (pre / post gather) |
| `decoder.py:53` | `h` | Idle rank's dummy layernorm, every 2nd layer — bounds JACCL queue under S==1 |
| `decoder.py:55` | `h` | After per-layer gather-1 |
| `decoder.py:66` | `out` | Idle rank's dummy layernorm, every 2nd layer |
| `decoder.py:68` | `result` | After per-layer gather-2 |

**(ii) Sampling evals (pre-`.item()` / `.tolist()`).** Required because Python-side
control flow depends on concrete values (draft tokens, acceptance bits, `n_accepted`).

| File:line | Target | Used for |
|-----------|--------|----------|
| `dflash_split.py:81` | `all_drafts_arr` | `.tolist()` to build `drafts` (temp=0) |
| `dflash_split.py:97` | `drafts_arr` | `.tolist()` after draft-sync `all_gather` |
| `dflash_batch_generator.py:125` | `target_hidden` | Before caching as `_last_target_hidden[uid]` (stock capture path) |
| `dflash_batch_generator.py:145, 151` | `prompt_toks`, `(target_hidden, logits)` | "direct" prefill mode only (unused under our patched path) |
| `dflash_batch_generator.py:199` | `all_drafts_arr` | Same role as `dflash_split.py:81`, stock path |

**(iii) Async evals (kick off compute, keep the Python loop moving).** These don't
block; MLX schedules the compute and the next `.item()` will wait. Used to
overlap acceptance math with the next forward's dispatch.

| File:line | Targets | Purpose |
|-----------|---------|---------|
| `dflash_split.py:120` | `matches`, `all_next`, `target_hidden` | temp=0 acceptance loop reads `matches[i].item()`; kicked off early |
| `dflash_split.py:146–148` | `accept_ratios`, `uniforms`, `corrections`, `bonus_token`, `target_hidden` | temp>0 acceptance |
| `dflash_split.py:239, 250` | `batch.y` | Emit next-token without blocking the `Response` return |
| `dflash_batch_generator.py:225, 243, 327, 338` | same four roles, stock path | Stock DFlash mirror — only fires if `_speculative_next` isn't patched |
| `mtp_batch_generator.py:115, 162, 182, 278, 291` | MTP equivalents | Not on DFlash path, here for cross-reference |

Note: `mtp_module.py:340, 359` also call `mx.eval` during MTP weight load; not on our hot path.

### 8c. Suggested instrumentation points

- Before/after each `all_gather` in `_split_speculative_next` (`dflash_split.py:66, 95, 159`) — shape + mean.
- `_populate_dflash_captured` (`model_forward.py:387`) — dump the set of keys written on each call.
- `SpeculativeArraysCache.rollback` (`speculative_cache.py:81`) — print `n_accepted`, whether `all_states` / `conv_input` were non-None, and the resulting `base.cache[*]` shapes.
- Entry to `_pipelined_dflash_forward` (`model_forward.py:623`) — print `inputs.shape`, length of `cache_list`, and whether any `SpeculativeArraysCache` already exists.
- Warmup completion — pair with `model_forward.py:212` to confirm one pipelined forward happens during warmup at `M=V+1`.

### 8d. Gotchas (bug history, one-liner each)

- Mask from `create_attention_mask(..., return_array=True)` is 2D `(S, offset+S)`, not 4D — `slice_fa_mask` handles both (`model_forward.py:63`).
- `BatchKVCache.offset` is sometimes an `mx.array`, sometimes `int`. Use `.max().item()` when unsure (`model_forward.py:188`).
- `_idx ≠ offset` on `BatchKVCache` — mask slicing uses `_idx` (actual K buffer length), positional encodings use `offset` (net of left_padding).
- `spec_all_states` has `2 × len(gdn_spec_data)` entries after our pipelined loop, not `len(gdn_spec_data)` — H0 and H1 each push. Merge post-loop (`model_forward.py:654`).
- Conv rollback only works if `conv_input` is reconstructed post-loop — the pipelined path destroys the pre-layer state that stock `dflash_speculative_forward` implicitly relies on.
- `_CapturingLayer.__call__` never fires on the pipelined path (we bypass `layer.__call__`). Compensated by `_populate_dflash_captured`.
- `ArraysCache.extract` stock crashes on mixed None / non-None `cache[i]` entries. MOE_RANK rollback produces exactly that — `[conv_input, None]`. Fixed by per-element `_arrays_extract` (`apply.py:130`).
- Drafter first cycle: `k_proj`, `v_proj`, `fc` see `M > MAX_M` and fall back to stock GEMM. Not a bug, but worth knowing when profiling cold-start latency.
- `_draft_position` on MOE_RANK stays at 0 because it never runs attention. Sync via `all_gather` at the top of every cycle (`dflash_split.py:66`).
- Drafter shares `embed_tokens` and `lm_head` with the target (`dflash_module.py:151, 153`) — weight changes to the target propagate automatically.

---

## 9. File index

Grouped by layer. Each entry: path, purpose, key symbols.

### 9a. Layer A (structural, `patches/qwen3_5_moe_split/`)

| File | Purpose | Key symbols |
|------|---------|-------------|
| `apply.py` | Install all structural patches | `apply_qwen35_attn_moe_split_patches:220`, `_patch_caches_for_moe_rank:33`, `_arrays_extract:130`, `_drop_unused_weights:297` |
| `decoder.py` | S==1 serial split `DecoderLayer.__call__` | `make_split_decoder_call:21`, `_split_call:36`, `ATTN_RANK:16`, `MOE_RANK:17`, `DUMMY_LN_ITERS:18` |
| `model_forward.py` | S>1 pipelined forward + DFlash capture plumbing | `attention:38`, `moe:50`, `slice_fa_mask:63`, `pipelined_layer_loop:146`, `_dflash_capturing_target_ids:353`, `_dflash_captured_dict:366`, `_populate_dflash_captured:387`, `make_pipelined_model_call:416`, `make_pipelined_speculative_forward:491` (MTP), `make_pipelined_dflash_speculative_forward:615` |
| `dflash_split.py` | Rank-aware `_speculative_next` for DFlash | `make_split_speculative_next:25`, `_split_speculative_next:29` |
| `PIPELINE.md` | This document | — |

### 9b. Layer B (kernel)

| File | Purpose | Key symbols |
|------|---------|-------------|
| `patches/qwen3_5/lpb_patch.py` | Target-side LpB projection patcher | `apply_lpb_patches:102`, `_make_bf16_forward:30`, `_make_int8_forward:51`, `_patch_proj:73`, `MAX_M:27` |
| `speculative/bf16_lpb_patch.py` | Drafter-side LpB patcher | `apply_bf16_lpb_patches:99`, `_BF16LpBLinear:27`, `_QuantizedLpBLinear:55`, `MAX_M:24` |
| `matmul/patches/kernel_picker.py` | (N, K, M) → kernel selection | `pick_bf16_kernel:33`, `pick_int8_kernel:69`, `_round_up_to_bench_col:26` |
| `matmul/kernels/bf16/*.py` | bf16 Metal kernels (lpb, lpb_twice, sk_steel, bm8) | `custom_bf16_qmv_loop_over_b`, `custom_bf16_qmv_loop_over_b_twice`, `custom_bf16_gemm_splitk_steel` |
| `matmul/kernels/quantized/*.py` | int8 Metal kernels (lpb, bm8, bm16, qsk) | `custom_qmv_loop_over_b`, `custom_qmm_splitk` |

### 9c. Stock DFlash (`speculative/`, unmodified)

| File | Purpose | Key symbols |
|------|---------|-------------|
| `dflash_batch_generator.py` | BatchGenerator subclass with DFlash logic | `DFlashBatchGenerator:20`, `_setup_hidden_capture:53`, `_next:92`, `_first_step_capture:119`, `_speculative_next:170`, `_yield_buffered:341` |
| `dflash_module.py` | DFlash drafter model | `DFlashAttention:19`, `DFlashDecoderLayer:73`, `DFlashDrafter:97`, `DFlashDrafter.draft:226`, `reset_draft_cache:186`, `crop_draft_cache:189` |
| `dflash_speculative.py` | Stock DFlash forward (replaced by our pipelined variant) | `dflash_speculative_forward:14` |
| `speculative_cache.py` | GDN rollback wrapper | `SpeculativeArraysCache:15`, `SpeculativeArraysCache.rollback:81` |
| `mtp_module.py` | MTP speculative machinery + shared speculative GDU factory | `speculative_forward:27`, `_make_speculative_gdu:126`, `MTPPredictor:154` |
| `speculative_gdn_kernel.py` | Speculative GDN Metal kernel | `speculative_gated_delta_kernel` |

### 9d. Driver / entry points

| File | Purpose | Key symbols |
|------|---------|-------------|
| `generator/batch_generate.py` | Orchestrator; drafter instantiation; warmup | DFlash init block at `:106–130`, `warmup_dflash:272`, MTP init block at `:135–170` |
| `auto_parallel.py` | Per-strategy model setup | `attn_moe_split_auto_parallel:436`, call site `:474` |
| `patches/__init__.py` | Single-device fused patch dispatch (not this path) | `apply_mlx_patches:14`, `maybe_apply_patches:26` |

### 9e. Stock mlx_lm touch points (read-only reference)

| File | Purpose |
|------|---------|
| `mlx_lm/models/qwen3_5.py` | `DecoderLayer`, `Qwen3_5TextModel`, `gated_delta_update`, attention modules |
| `mlx_lm/models/cache.py` | `ArraysCache:592`, `BatchKVCache`, `KVCache`, `ArraysCache.extract:630` |
| `mlx_lm/generate.py` | `BatchGenerator`, `Batch.extract_cache:882` |

---

_Document maintained on `david/attn-moe-split`. Last updated when `_arrays_extract` was made None-tolerant (commit `e8276c11`) and `_populate_dflash_captured` was added (commit `b1a5dccf`)._
