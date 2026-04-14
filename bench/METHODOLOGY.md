# Exo-Bench — Methodology

exo bench measures inference throughput and resource consumption of an exo cluster under controlled conditions. It sends prompts to the `/bench/chat/completions` endpoint, collects server-reported timing statistics, and records system-level metrics (power, GPU utilisation, temperature) throughout each run.

The goal is to have accurate, transparent and reproducible numbers to compare speed and scaling across different models and different setups, and to be able to track these results as optimizations and features are added to EXO.

Below is a technical summary of how Exo-Bench works. While the methodology and benchmark may change over time, this document will be kept up to date whenever this happens. If you find an issue with the methodology, or would like a  feature to be added, please open a GitHub issue!

---

## Prompt Construction

Benchmarks need prompts of an exact token length. Unfortunately, we do not have direct access to the model but just the chat completion endpoint. To get around this fact, we create a request that will tokenise to a certain prompt length.

This is achieved by:

1. Tokenising a sample message through the model's `apply_chat_template()` to measure overhead (system tokens, special tokens, chat formatting).
2. Binary-searching over a repeated atom string (default `"a "`) to find the content length that produces exactly the target number of tokens after template expansion.
3. Returning both the content string and the verified token count.

The actual token count is recorded in every result row as `pp_tokens`, so downstream analysis can confirm the prompt hit its target.

Chat template formatting means that it may be impossible to attain very small pp benchmarks. e.g. pp=32 may not work. This tradeoff was made because the result of such a small prompt does not seem very interesting or useful for any real-world use cases.

---

## Bench Endpoint

When a request reaches the server via the `/bench/chat/completions` endpoint, three things change compared to a normal chat completion:

- **KV prefix cache is disabled**. Every request starts from a cold cache, ensuring prefill timing is not affected by prior requests.
- **EOS tokens are banned**. A logits processor suppresses all end-of-sequence tokens, forcing the model to generate exactly `max_tokens` tokens. This guarantees consistent generation length for fair TPS comparison — the model cannot short-circuit a run by stopping early.
- **No model output parsing**. The bench collection path concatenates raw token text without any model-specific post-processing (thinking tag extraction, structured output handling, etc.). This is to avoid model outputs such as tool parsing or any structural mistakes from breaking the benchmark - we are testing for speed; see Exo-Eval for performance metrics.

---

## Timing

### Prefill TPS

Measured server-side per task.

```
prefill_tps = num_prompt_tokens / prefill_wall_seconds
```

### Generation TPS

Measured server-side per task. Each task records wall-clock timestamps as tokens arrive:

- First generated token: timestamp recorded
- Every subsequent token: timestamp updated

When generation completes:

```
gen_span = last_token_time - first_token_time
generation_tps = (completion_tokens - 1) / gen_span
```

The first token is excluded from the numerator because the rate measures inter-token throughput — the time between the first and last token divided by the number of intervals.

This does mean that tg=1 will not work.

---

## Concurrency

### Single Request

The client records wall-clock `elapsed_s` around the HTTP round-trip (network latency + server prefill + generation + response serialisation). This is a convenience metric for end-to-end latency. The authoritative TPS numbers come from the server-side per-task timing in the `generation_stats` response.

### Concurrent Requests

When `--concurrency N` is set with N > 1, all N requests must hit the server at the same instant. The mechanism:

1. The prompt is built once and shared across all threads.
2. Each thread gets its own HTTP connection.
3. A thread barrier blocks all threads until every thread is ready.
4. The first thread past the barrier records the batch start time and signals the others.
5. All threads use the same start time as their reference, then fire their HTTP request.
6. Each thread's `elapsed_s` is measured from the shared start time to its own response completion.

**Batch wall time** is the maximum `elapsed_s` across all N requests — the time until the last request finishes.

### Aggregate TPS

```
per_req_tps = max(generation_tps across N concurrent requests)
agg_gen_tps = per_req_tps * concurrency
```

`max` is used instead of `mean` because all requests run in parallel against the same model. The fastest request's generation rate represents the system's per-stream throughput capacity; multiplying by concurrency gives aggregate throughput.

---

## Warmup

Before measurement begins, `--warmup N` (default: 0) discarded requests are sent using the first pp/tg pair. Warmup results are not included in the output.

---

## System Metrics

A background thread polls each node at 1 Hz, collecting:

- GPU utilisation (%)
- Temperature (C)
- System power draw (W)
- CPU cluster usage (performance and efficiency cores)

**Energy** is computed via trapezoidal integration of the power samples over each inference window (the wall-clock span of each benchmark request or concurrent batch). Average power is `total_joules / total_inference_seconds`.

---

## Output Format

Results are written as JSON with three top-level keys:

- **`runs`**: Array of per-request result objects, each containing:
  - `elapsed_s`, `output_text_preview` (first 200 chars)
  - `stats`: `{ prompt_tps, generation_tps, prompt_tokens, generation_tokens, peak_memory_usage }`
  - Placement metadata: `model_id`, `placement_sharding`, `placement_instance_meta`, `placement_nodes`
  - Run metadata: `pp_tokens`, `tg`, `repeat_index`, `concurrency`, `concurrent_index`
  - `download_duration_s` (if model was freshly downloaded)
- **`cluster`**: Cluster state snapshot at time of benchmarking.
- **`system_metrics`**: Per-node time-series samples (GPU, power, temperature).

---

## Reproducing Results

```bash
cd bench && uv run python exo_bench.py \
  --model "mlx-community/Qwen3.5-27B-4bit" \
  --instance-meta jaccl \
  --sharding tensor \
  --min-nodes 2 --max-nodes 2 \
  --pp 512 4096 --tg 128 \
  --repeat 3 \
  --warmup 1
```

Run --help for all the available flags.
