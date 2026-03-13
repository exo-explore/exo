#!/usr/bin/env python3
"""Verify TorchKVCache round-trip produces identical model outputs.

Run on Mac:
    uv run python scripts/investigate_roundtrip_mlx.py --model-path ~/.exo/models/mlx-community--gpt-oss-20b-MXFP4-Q8
"""
import argparse
from copy import deepcopy

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from exo.worker.engines.kv_cache import TorchKVCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model, tokenizer = load(args.model_path)

    prompt = "The capital of France is"
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    print(f"Prompt: {prompt!r} ({len(prompt_tokens)} tokens)")

    # Prefill: generate 1 token to populate cache
    cache_orig = model.make_cache()
    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=1,
        sampler=make_sampler(temp=0.0),
        prompt_cache=cache_orig,
    ):
        first_token = out.text
        break
    print(f"First generated token: {first_token!r}")

    # Snapshot the original cache (deep copy for comparison after model forward)
    cache_snapshot = deepcopy(cache_orig)

    # Convert original cache → TorchKVCache → back to MLX
    print("\nConverting: MLX -> TorchKVCache (NHD) -> MLX")
    torch_cache = TorchKVCache.from_mlx_cache(cache_snapshot)
    print(torch_cache)
    cache_roundtrip = torch_cache.to_mlx_cache()

    # Run model forward with original cache
    next_token = mx.array([[tokenizer.encode(first_token)[-1]]])
    logits_orig = model(next_token, cache=cache_orig)
    mx.eval(logits_orig)

    # Run model forward with round-tripped cache
    logits_roundtrip = model(next_token, cache=cache_roundtrip)
    mx.eval(logits_roundtrip)

    # Compare
    diff = mx.abs(logits_orig - logits_roundtrip)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    print("\n=== Logit Comparison ===")
    print(f"  logits_orig shape: {logits_orig.shape}")
    print(f"  max  abs diff: {max_diff:.2e}")
    print(f"  mean abs diff: {mean_diff:.2e}")

    # Check if top token is the same
    top_orig = mx.argmax(logits_orig[0, -1]).item()
    top_roundtrip = mx.argmax(logits_roundtrip[0, -1]).item()
    print(f"  top token orig:      {top_orig} ({tokenizer.decode([top_orig])!r})")
    print(f"  top token roundtrip: {top_roundtrip} ({tokenizer.decode([top_roundtrip])!r})")
    print(f"  top tokens match: {top_orig == top_roundtrip}")

    if max_diff == 0.0:
        print("\nBIT-EXACT round-trip confirmed.")
    elif max_diff < 1e-5:
        print(f"\nNear-exact round-trip (max diff {max_diff:.2e}, likely bfloat16 precision).")
    else:
        print(f"\nWARNING: significant round-trip diff {max_diff:.2e}")


if __name__ == "__main__":
    main()
