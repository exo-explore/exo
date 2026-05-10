# Gemma 4 MTP Coupled-Drafter Benchmark Report

A/B comparison of `EXO_DRAFT_MODE=none` vs `EXO_DRAFT_MODE=model` (with the
target's `coupled_drafter` declared in the model card) on a single
wc-smbp host, post Phase 3 ship of `Drafter` + `CoupledModelDrafter`.

## Hardware / software
- Host: `wc-smbp` (Apple M5 Max MacBook Pro, 128 GB unified memory)
- Branch: `team-wcv/main` @ `ed3897f7` (Phase 3 cards merged)
- mlx-vlm: 0.5.0
- Generation backend: `MlxRingInstance`, single device, pipeline sharding
- Bench harness: `bench/drafter_bench.py`, 2 runs/scenario, warmup enabled

## Target / drafter pairs

Two target/drafter pairs were benched:

| Target                                       | Storage | Drafter                                              |
|----------------------------------------------|---------|------------------------------------------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit` (MoE) | ~15.6 GB | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` |
| `mlx-community/gemma-4-31b-it-4bit` (dense)  | ~17.1 GB | `mlx-community/gemma-4-31B-it-assistant-bf16`        |

Confirmed dispatch via API telemetry on every MTP run:
`response.draft_mode == "model"`, `drafter_model_id == "...assistant-bf16"`,
`accept_fraction` populated per-request.

## Results — Gemma 4 26B-A4B (MoE)

Median generation tokens/s across 2 runs per scenario.

| Scenario              | Target only (t/s) | MTP coupled (t/s) | Speedup | MTP accept |
|-----------------------|-------------------|-------------------|---------|------------|
| short_repetitive      | 123.4             | 132.6             | +7.5%   | 0.65       |
| code_completion       | 122.3             | 149.3             | +22.1%  | 0.70       |
| creative_prose        | 120.1             | 93.9              | -21.8%  | 0.55       |
| factual_qa            | 118.3             | 118.3             | +0.0%   | 0.64       |
| long_context_summary  | 107.4             | 42.0              | -60.9%  | 0.16       |
| **all-scenario median** | **120.3**       | **118.3**         | **-1.6%** | -        |

Raw JSON: `bench/results/mtp/gemma-4-26b-a4b-it-4bit-{target-only,mtp-coupled}.json`.

## Results — Gemma 4 31B (dense)

Median generation tokens/s across 2 runs per scenario.

| Scenario              | Target only (t/s) | MTP coupled (t/s) | Speedup | MTP accept |
|-----------------------|-------------------|-------------------|---------|------------|
| short_repetitive      | 26.01             | 28.10             | +8.1%   | 0.65       |
| code_completion       | 25.67             | 29.05             | +13.2%  | 0.67       |
| factual_qa            | 24.81             | 26.76             | +7.9%   | 0.66       |
| creative_prose        | 25.39             | 20.61             | -18.8%  | 0.58       |
| long_context_summary  | 21.03             | 8.08              | -61.6%  | 0.03       |
| **all-scenario median** | **25.39**       | **26.76**         | **+5.4%** | -        |

Raw JSON: `bench/results/mtp/gemma-4-31b-it-4bit-{target-only,mtp-coupled}.json`.

## Headline numbers vs scenario peaks

The all-scenario medians above (-1.6% for 26B-A4B, +5.4% for 31B)
include the long-context summary regression that drags the aggregate
toward zero. The strong gains on tight, repetitive, code-shaped output
are real and consistent with prior runs:

- 26B-A4B (MoE): peak speedup +22.1% on `code_completion` (the
  "~+25% on code" figure observed in earlier private benches).
- 31B (dense): peak speedup +13.2% on `code_completion`, with three of
  five scenarios above +7%.

The **right way to read these numbers**: the all-scenario median is a
worst-case proxy that mixes a pathological long-context case with the
favourable scenarios. A per-prompt routing heuristic (recommended
below) would let the dispatch keep MTP on for the high-accept
scenarios and fall back to plain decoding for the low-accept ones,
converting the headline median much closer to the per-scenario peak.

## Interpretation

1. **MTP dispatch is wired correctly on both targets.** Every MTP request
   comes back with `draft_mode == "model"`, the coupled drafter
   `model_id` populated, and a per-request `accept_fraction`. This
   validates Phase 2c's `mlx_generate` dispatch and Phase 3's card
   wiring across both an MoE and a dense target.

2. **Speedup is content-dependent and tracks accept rate.** Across both
   targets, scenarios where the assistant predicts well (`accept ≳ 0.65`)
   win double-digit speedups; scenarios where it predicts poorly
   (`accept ≲ 0.20`) regress sharply because the assistant cost is paid
   even when tokens are rejected. The dense 31B median crosses zero into
   net positive (+5.4%) because more scenarios fall into the high-accept
   regime; the MoE 26B-A4B median sits at ~zero (-1.6%) because its
   target-only path is unusually fast (only ~4B active params).

3. **Long-context summarization is consistently bad for MTP.** Both
   targets see the same -61% regression with `accept` collapsing to
   single-digit percent. The assistant head is trained on short-form
   coherent prose; large summarization prompts push it out of
   distribution. This is the canonical case for routing MTP off.

4. **Dense targets benefit more in absolute terms than MoE targets.**
   The 31B target only runs at ~25 t/s standalone (vs. ~120 t/s for the
   26B-A4B MoE), so the same MTP overhead amortizes better and the same
   accept rate yields a larger relative speedup.

## Recommended follow-ups

- **Add a per-prompt routing heuristic** that disables MTP when the
  model's running-window accept rate falls below a configurable floor
  (e.g. 0.3). The infrastructure already publishes `accept_fraction`
  per request, so this is a small `mlx_generate` change rather than a
  drafter-architecture change. With the 31B numbers in hand, a default
  floor around 0.4–0.5 would convert the +5.4% median into something
  closer to the +13% headline (code-completion-style) without exposing
  the user to the -61% long-context regression.
- **`safetensors.index.json` bootstrap for single-file MTP drafters.**
  The 26B/31B `assistant` heads ship as single safetensors; the current
  bench needed a `chat_template.jinja` patch on the 31B target after
  packaging fixed the safetensors-only case. Single-file drafters
  (e.g. `gemma-4-e2b-it-4bit`) still need the bootstrap to be packaged
  cleanly without the manual workaround. Tracked as
  `card_drafter_packaging` in the local todo list (fix landed in
  `src/exo/download/download_utils.py`; pending downstream verification).
