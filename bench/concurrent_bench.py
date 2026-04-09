# type: ignore
#!/usr/bin/env python3
"""Concurrent inference benchmark against an existing exo deployment.

Unlike `exo_bench.py`, this script does NOT place or tear down instances.
It fires N parallel chat completions against whatever instances of `--model`
are already running on the cluster, repeats for `--iterations`, and reports
per-request and aggregate tok/s along with the bimodal-tail diagnostic
(max_wall / median_wall ratio per iteration).

Used to characterize sibling-runner contention on a single machine for the
"Metal scheduling" investigation. Run before/after a fix to compare.

Examples:
    # Single-shot, 6-way concurrent
    uv run python bench/concurrent_bench.py \\
        --model mlx-community/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit \\
        --concurrency 6 --iterations 10

    # Sweep concurrency
    uv run python bench/concurrent_bench.py \\
        --model mlx-community/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit \\
        --concurrency 1,2,4,6,8 --iterations 5 \\
        --json-out bench/results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from harness import ExoClient, ExoHttpError
from loguru import logger


def _parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        for piece in v.split(","):
            piece = piece.strip()
            if piece:
                out.append(int(piece))
    return out


def _fixed_prompt(approx_words: int) -> str:
    """A fixed prompt of approximately N words. We don't aim for exact token
    counts here — concurrent_bench is about contention, not token-accounting."""
    base = (
        "Write a concise technical explanation of how a CPU pipeline handles "
        "branch prediction, including speculative execution and how mispredicts "
        "are recovered. Be precise and avoid filler. "
    )
    repeats = max(1, approx_words // len(base.split()))
    return (base * repeats).strip()


def _run_one(
    client: ExoClient, model: str, prompt: str, max_tokens: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    out = client.post_bench_chat_completions(payload)
    wall = time.perf_counter() - t0

    stats = out.get("generation_stats") or {}
    return {
        "wall_s": wall,
        "prompt_tps": float(stats.get("prompt_tps") or 0.0),
        "generation_tps": float(stats.get("generation_tps") or 0.0),
        "prompt_tokens": int(stats.get("prompt_tokens") or 0),
        "generation_tokens": int(stats.get("generation_tokens") or 0),
    }


def _run_concurrent_iteration(
    host: str,
    port: int,
    timeout_s: float,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    """One iteration: fire `concurrency` requests in parallel, return stats."""
    iter_t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    def _task(_idx: int) -> dict[str, Any]:
        c = ExoClient(host, port, timeout_s=timeout_s)
        return _run_one(c, model, prompt, max_tokens)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_task, i): i for i in range(concurrency)}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(repr(e))

    iter_wall = time.perf_counter() - iter_t0

    if not results:
        return {
            "wall_s": iter_wall,
            "errors": errors,
            "ok": False,
        }

    walls = sorted(r["wall_s"] for r in results)
    gen_tps_per_req = [r["generation_tps"] for r in results if r["generation_tps"] > 0]
    gen_tokens = sum(r["generation_tokens"] for r in results)

    median_wall = statistics.median(walls)
    max_wall = max(walls)
    min_wall = min(walls)
    # bimodal indicator: how much the slowest request lagged the median
    tail_ratio = (max_wall / median_wall) if median_wall > 0 else float("inf")

    per_req_gen_tps_mean = statistics.mean(gen_tps_per_req) if gen_tps_per_req else 0.0
    # Steady-state aggregate generation throughput. Matches exo_bench.py's
    # `agg_gen_tps = per_req_tps * concurrency` — independent of prefill cost.
    aggregate_gen_tps = per_req_gen_tps_mean * len(results)
    # Realized wall throughput including prefill — relevant for short
    # max_tokens runs where prefill dominates.
    wall_throughput_tps = (gen_tokens / iter_wall) if iter_wall > 0 else 0.0

    return {
        "wall_s": iter_wall,
        "min_wall_s": min_wall,
        "median_wall_s": median_wall,
        "max_wall_s": max_wall,
        "tail_ratio": tail_ratio,
        "per_req_gen_tps_mean": per_req_gen_tps_mean,
        "aggregate_gen_tps": aggregate_gen_tps,
        "wall_throughput_tps": wall_throughput_tps,
        "gen_tokens_total": gen_tokens,
        "errors": errors,
        "per_request": results,
        "ok": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="concurrent-bench",
        description=(
            "Fire N concurrent chat-completion requests against an existing "
            "exo deployment and measure aggregate / per-request throughput "
            "and tail latency. Does NOT place or delete instances."
        ),
    )
    ap.add_argument("--host", default="192.168.86.201")
    ap.add_argument("--port", type=int, default=52415)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--model",
        required=True,
        help="Model id (must already be deployed on the cluster).",
    )
    ap.add_argument(
        "--concurrency",
        nargs="+",
        default=["6"],
        help="Concurrency levels (ints, comma-separated). E.g. 1,2,4,6,8.",
    )
    ap.add_argument(
        "--iterations", type=int, default=10, help="Iterations per concurrency level."
    )
    ap.add_argument(
        "--max-tokens", type=int, default=256, help="max_tokens per request."
    )
    ap.add_argument(
        "--prompt-words",
        type=int,
        default=120,
        help="Approximate words in the fixed prompt (controls prompt-token count).",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations per concurrency level (excluded from stats).",
    )
    ap.add_argument(
        "--json-out",
        default=None,
        help="Path to write per-iteration JSON results.",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="Free-form label embedded in the JSON (e.g. 'pre-fix', 'post-fix').",
    )
    args = ap.parse_args()

    concurrency_levels = _parse_int_list(args.concurrency)
    if not concurrency_levels:
        logger.error("--concurrency must contain at least one positive integer")
        return 2
    if any(c <= 0 for c in concurrency_levels):
        logger.error("--concurrency values must be >= 1")
        return 2

    prompt = _fixed_prompt(args.prompt_words)

    # Sanity check: server reachable, model present.
    probe = ExoClient(args.host, args.port, timeout_s=15.0)
    try:
        state = probe.request_json("GET", "/state")
    except ExoHttpError as e:
        logger.error(f"Cannot reach exo at {args.host}:{args.port}: {e}")
        return 1

    matching_instances: list[str] = []
    for inst_id, inst in (state.get("instances") or {}).items():
        ring = inst.get("MlxRingInstance") or {}
        sa = ring.get("shardAssignments") or {}
        if sa.get("modelId") == args.model:
            matching_instances.append(inst_id)
    if not matching_instances:
        logger.error(
            f"No running instances found for model {args.model}. "
            f"Place at least one before running this bench."
        )
        return 1
    logger.info(f"Found {len(matching_instances)} running instances of {args.model}.")

    output: dict[str, Any] = {
        "model": args.model,
        "host": args.host,
        "port": args.port,
        "label": args.label,
        "instance_count_at_start": len(matching_instances),
        "iterations_per_level": args.iterations,
        "warmup_per_level": args.warmup,
        "max_tokens": args.max_tokens,
        "prompt_words": args.prompt_words,
        "by_concurrency": {},
    }

    for concurrency in concurrency_levels:
        logger.info(
            f"=== concurrency={concurrency}  iterations={args.iterations}  "
            f"warmup={args.warmup}  max_tokens={args.max_tokens} ==="
        )

        all_iters: list[dict[str, Any]] = []
        for i in range(args.warmup + args.iterations):
            is_warmup = i < args.warmup
            result = _run_concurrent_iteration(
                args.host,
                args.port,
                args.timeout,
                args.model,
                prompt,
                args.max_tokens,
                concurrency,
            )
            result["iteration_index"] = i
            result["warmup"] = is_warmup
            all_iters.append(result)

            if not result.get("ok"):
                logger.error(
                    f"  iter={i:>2}{' (warmup)' if is_warmup else ''}  "
                    f"FAILED  errors={result.get('errors')}"
                )
                continue

            tag = "warmup " if is_warmup else "       "
            logger.info(
                f"  iter={i:>2} {tag} wall={result['wall_s']:.2f}s  "
                f"agg_tps={result['aggregate_gen_tps']:.1f}  "
                f"per_req={result['per_req_gen_tps_mean']:.1f}  "
                f"tail_ratio={result['tail_ratio']:.2f}  "
                f"max_wall={result['max_wall_s']:.2f}s  "
                f"errors={len(result.get('errors') or [])}"
            )

        scored = [r for r in all_iters if not r.get("warmup") and r.get("ok")]
        if scored:
            agg_tps_vals = [r["aggregate_gen_tps"] for r in scored]
            per_req_tps_vals = [r["per_req_gen_tps_mean"] for r in scored]
            tail_ratios = [r["tail_ratio"] for r in scored]
            walls = [r["wall_s"] for r in scored]

            # bimodal flagging: an iteration is "bad" if its tail_ratio > 1.5,
            # i.e. the slowest request was >50% slower than the median.
            bad_iters = [r for r in scored if r["tail_ratio"] > 1.5]

            summary = {
                "iterations_scored": len(scored),
                "agg_tps_mean": statistics.mean(agg_tps_vals),
                "agg_tps_median": statistics.median(agg_tps_vals),
                "agg_tps_min": min(agg_tps_vals),
                "agg_tps_max": max(agg_tps_vals),
                "per_req_tps_mean": statistics.mean(per_req_tps_vals),
                "tail_ratio_mean": statistics.mean(tail_ratios),
                "tail_ratio_max": max(tail_ratios),
                "wall_mean_s": statistics.mean(walls),
                "wall_max_s": max(walls),
                "bad_iter_count": len(bad_iters),
                "bad_iter_rate": len(bad_iters) / len(scored),
            }
            logger.info(
                f"  SUMMARY  agg_tps={summary['agg_tps_mean']:.1f} (med {summary['agg_tps_median']:.1f}, "
                f"min {summary['agg_tps_min']:.1f}, max {summary['agg_tps_max']:.1f})  "
                f"per_req={summary['per_req_tps_mean']:.1f}  "
                f"tail_ratio_mean={summary['tail_ratio_mean']:.2f}  "
                f"bad_rate={summary['bad_iter_rate'] * 100:.0f}% "
                f"({summary['bad_iter_count']}/{len(scored)})"
            )
        else:
            summary = {"iterations_scored": 0}
            logger.warning("  SUMMARY  no successful scored iterations")

        output["by_concurrency"][str(concurrency)] = {
            "summary": summary,
            "iterations": all_iters,
        }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        logger.info(f"wrote results to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
