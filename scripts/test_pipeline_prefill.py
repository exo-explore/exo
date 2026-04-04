#!/usr/bin/env python3
"""Test prefill throughput through the full exo pipeline.

Connects to a running exo instance, creates a vLLM placement for the given model,
sends a prompt of the specified token count, and reports prompt_tps.

Usage:
    uv run python scripts/test_pipeline_prefill.py --model openai/gpt-oss-20b --pp 8192 --host gx10-de89
    uv run python scripts/test_pipeline_prefill.py --model openai/gpt-oss-20b --pp 4096,8192 --host gx10-de89 --repeat 3
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from typing import Any

sys.path.insert(0, "bench")

from harness import (
    ExoClient,
    ExoHttpError,
    instance_id_from_instance,
    resolve_model_short_id,
    run_planning_phase,
    settle_and_fetch_placements,
    wait_for_instance_gone,
    wait_for_instance_ready,
)

from exo_bench import PromptSizer, load_tokenizer_for_bench


def run_one(client: ExoClient, model_id: str, pp: int, tg: int, sizer: PromptSizer) -> dict[str, Any]:
    content, actual_pp = sizer.build(pp)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "max_tokens": tg,
    }
    t0 = time.perf_counter()
    out = client.post_bench_chat_completions(payload)
    elapsed = time.perf_counter() - t0
    stats = out.get("generation_stats", {})
    return {
        "elapsed_s": elapsed,
        "prompt_tps": stats.get("prompt_tps", 0),
        "prompt_tokens": stats.get("prompt_tokens", 0),
        "generation_tps": stats.get("generation_tps", 0),
        "generation_tokens": stats.get("generation_tokens", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Test prefill throughput through exo pipeline")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=52415)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pp", required=True, help="Prompt token counts, comma-separated")
    ap.add_argument("--tg", type=int, default=2, help="Generation tokens (default 2)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    pp_list = [int(x.strip()) for x in args.pp.split(",")]
    client = ExoClient(args.host, args.port, timeout_s=args.timeout)

    short_id, full_model_id = resolve_model_short_id(client, args.model, force_download=args.force_download)
    print(f"Model: {short_id} ({full_model_id})")

    tokenizer = load_tokenizer_for_bench(full_model_id)
    sizer = PromptSizer(tokenizer)

    ns = argparse.Namespace(
        instance_meta="vllm",
        sharding="both",
        max_nodes=1,
        min_nodes=1,
        skip_pipeline_jaccl=False,
        skip_tensor_ring=False,
        settle_timeout=60,
        danger_delete_downloads=False,
        timeout=args.timeout,
    )

    selected = settle_and_fetch_placements(client, full_model_id, ns, settle_timeout=60)
    if not selected:
        print("ERROR: No vLLM placements available")
        return 1

    preview = selected[0]
    instance = preview["instance"]
    instance_id = instance_id_from_instance(instance)
    print(f"Placement: {preview.get('sharding')} / {preview.get('instance_meta')} / id={instance_id}")

    run_planning_phase(client, full_model_id, preview, False, args.timeout, None)

    client.request_json("POST", "/instance", body={"instance": instance})
    try:
        wait_for_instance_ready(client, instance_id)
    except (RuntimeError, TimeoutError) as e:
        print(f"ERROR: Failed to initialize: {e}")
        with contextlib.suppress(ExoHttpError):
            client.request_json("DELETE", f"/instance/{instance_id}")
        return 1

    print("Instance ready. Running prefill tests...")
    time.sleep(2)

    try:
        for pp in pp_list:
            for r in range(args.repeat):
                time.sleep(1)
                try:
                    result = run_one(client, full_model_id, pp, args.tg, sizer)
                except Exception as e:
                    print(f"  pp={pp} repeat={r+1}: FAILED - {e}")
                    continue
                prompt_tps = result["prompt_tps"]
                marker = "OK" if prompt_tps >= 3500 else "BELOW TARGET"
                print(
                    f"  pp={pp} repeat={r+1}: "
                    f"prompt_tps={prompt_tps:.1f} "
                    f"prompt_tokens={result['prompt_tokens']} "
                    f"elapsed={result['elapsed_s']:.2f}s "
                    f"[{marker}]"
                )
    finally:
        try:
            client.request_json("DELETE", f"/instance/{instance_id}")
        except ExoHttpError as e:
            if e.status != 404:
                raise
        wait_for_instance_gone(client, instance_id)
        print("Instance cleaned up.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
