#!/usr/bin/env python3
"""Investigate MLX KV cache shapes and test NHD conversion.

Run on Mac:
    uv run python scripts/investigate_mlx_kv.py --model-path ~/.exo/models/mlx-community--gpt-oss-20b-MXFP4-Q8
"""
import argparse

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler

from exo.worker.engines.kv_cache import TorchKVCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model, tokenizer = load(args.model_path)
    print(f"Model loaded. {len(model.layers)} layers")

    # Create cache and run generation to populate it
    prompt = "Hello, how are you?"
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    print(f"Prompt: {prompt!r} ({len(prompt_tokens)} tokens)")

    cache = model.make_cache()
    print(f"\n=== Raw MLX Cache ({len(cache)} layers) ===")
    for i, c in enumerate(cache):
        print(f"  [{i}] {type(c).__name__}", end="")
        if isinstance(c, RotatingKVCache):
            print(f" max_size={c.max_size} keep={c.keep}")
        elif isinstance(c, ArraysCache):
            print(f" size={len(c.state)}")
        else:
            print()

    # Generate tokens to fill the cache
    print(f"\nGenerating {args.max_tokens} tokens...")
    generated = []
    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.0),
        prompt_cache=cache,
    ):
        generated.append(out.text)
        if out.finish_reason:
            break
    print(f"Generated: {''.join(generated)!r}")

    # Inspect populated cache
    print("\n=== Populated MLX Cache ===")
    for i, c in enumerate(cache):
        if isinstance(c, (KVCache, RotatingKVCache)):
            if c.keys is not None:
                k, v = c.state
                mx.eval(k)
                mx.eval(v)
                print(f"  [{i}] {type(c).__name__}: keys={k.shape} values={v.shape} dtype={k.dtype} offset={c.offset}", end="")
                if isinstance(c, RotatingKVCache):
                    print(f" _idx={c._idx} keep={c.keep} max_size={c.max_size} meta_state={c.meta_state}")
                else:
                    print()
            else:
                print(f"  [{i}] {type(c).__name__}: empty")
        elif isinstance(c, ArraysCache):
            shapes = [list(a.shape) if a is not None else None for a in c.state]
            print(f"  [{i}] ArraysCache: {shapes}")

    # Convert to TorchKVCache
    print("\n=== Converting to TorchKVCache (NHD format) ===")
    torch_cache = TorchKVCache.from_mlx_cache(cache)
    print(torch_cache)

    # Value statistics for KV layers
    print("\n=== Value Statistics ===")
    for idx, layer in torch_cache.kv_layers():
        k, v = layer.keys, layer.values
        print(f"  [{idx}] keys: min={k.min():.6f} max={k.max():.6f} mean={k.mean():.6f} std={k.std():.6f}")
        print(f"  [{idx}] vals: min={v.min():.6f} max={v.max():.6f} mean={v.mean():.6f} std={v.std():.6f}")

    # Round-trip test
    print("\n=== Round-trip: TorchKVCache -> MLX Cache ===")
    restored_cache = torch_cache.to_mlx_cache()

    for i, (orig, restored) in enumerate(zip(cache, restored_cache, strict=True)):
        if isinstance(orig, (KVCache, RotatingKVCache)) and orig.keys is not None:
            ok, ov = orig.state
            rk, rv = restored.state
            mx.eval(ok, ov, rk, rv)
            k_diff = mx.max(mx.abs(ok - rk)).item()
            v_diff = mx.max(mx.abs(ov - rv)).item()
            print(f"  [{i}] {type(orig).__name__}: key_diff={k_diff:.2e} val_diff={v_diff:.2e}", end="")
            if isinstance(orig, RotatingKVCache):
                meta_match = orig.meta_state == restored.meta_state
                print(f" meta_match={meta_match} orig={orig.meta_state} restored={restored.meta_state}")
            else:
                offset_match = orig.offset == restored.offset
                print(f" offset_match={offset_match} ({orig.offset} vs {restored.offset})")
        elif isinstance(orig, ArraysCache):
            diffs = []
            for _j, (oa, ra) in enumerate(zip(orig.state, restored.state, strict=True)):
                if oa is not None and ra is not None:
                    mx.eval(oa, ra)
                    diffs.append(mx.max(mx.abs(oa - ra)).item())
            print(f"  [{i}] ArraysCache: max_diffs={[f'{d:.2e}' for d in diffs]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
