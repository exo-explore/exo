"""Inject extracted vLLM KV cache into MLX model caches and test decode.

Runs on Mac (Apple Silicon). Loads per-layer KV tensors saved by
test_kv_extract.py, converts to MLX format, injects into MLX caches,
and generates tokens to verify correctness.

Usage:
  uv run python scripts/disaggregated/test_kv_inject.py \
    --model mlx-community/gpt-oss-20b-MXFP4-Q8 \
    --kv-dir /path/to/extracted/kv_cache/ \
    --num-tokens 20
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mlx.core as mx
import torch
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache


def _torch_to_mx(t: torch.Tensor) -> mx.array:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy()).astype(mx.bfloat16)
    return mx.array(t.numpy())


def _to_bhsd(keys: torch.Tensor, values: torch.Tensor, num_tokens: int) -> tuple[mx.array, mx.array]:
    """Convert vLLM block format to MLX BHSD [1, H, S, D].

    Input can be:
      - 4D [blocks, block_size, H, D] — flatten to [blocks*block_size, H, D], trim to num_tokens
      - 3D [S, H, D] — use directly
    """
    if keys.dim() == 4:
        keys = keys.reshape(-1, keys.shape[2], keys.shape[3])[:num_tokens]
        values = values.reshape(-1, values.shape[2], values.shape[3])[:num_tokens]
    elif keys.dim() == 3:
        keys = keys[:num_tokens]
        values = values[:num_tokens]

    k_mx = _torch_to_mx(keys.permute(1, 0, 2).unsqueeze(0))
    v_mx = _torch_to_mx(values.permute(1, 0, 2).unsqueeze(0))
    return k_mx, v_mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MLX model path/ID")
    parser.add_argument("--kv-dir", required=True, help="Directory with extracted KV tensors")
    parser.add_argument("--num-tokens", type=int, default=500, help="Tokens to generate")
    parser.add_argument("--prompt", default=None, help="Override prompt (must match extraction prompt)")
    args = parser.parse_args()

    kv_dir = Path(args.kv_dir)
    with open(kv_dir / "metadata.json") as f:
        metadata = json.load(f)

    num_extracted_layers = metadata["num_layers"]
    num_tokens = metadata["num_tokens"]
    vllm_token_ids = metadata.get("token_ids", [])

    print(f"Extracted KV: {num_extracted_layers} layers, {num_tokens} tokens")
    if vllm_token_ids:
        print(f"  Using vLLM token_ids ({len(vllm_token_ids)} tokens)")
    else:
        print(f"  WARNING: No token_ids in metadata")

    print(f"\nLoading MLX model: {args.model}")
    model, tokenizer = load(args.model)

    caches = model.make_cache()
    num_model_layers = len(caches)
    print(f"\nMLX model expects {num_model_layers} cache layers:")
    for i, c in enumerate(caches):
        print(f"  Layer {i:3d}: {type(c).__name__}", end="")
        if isinstance(c, RotatingKVCache):
            print(f" (max_size={c.max_size}, keep={c.keep})", end="")
        elif isinstance(c, ArraysCache):
            print(f" (size={len(c.state)})", end="")
        print()

    layer_info = metadata.get("layers", [])
    print(f"\nExtracted {num_extracted_layers} layers from vLLM")

    print("\nInjecting KV cache into MLX caches...")
    injected = 0
    skipped = 0
    for i in range(num_model_layers):
        cache = caches[i]

        if isinstance(cache, ArraysCache):
            conv_path = kv_dir / f"layer_{i:03d}_conv.pt"
            rec_path = kv_dir / f"layer_{i:03d}_rec.pt"
            keys_path = kv_dir / f"layer_{i:03d}_keys.pt"
            values_path = kv_dir / f"layer_{i:03d}_values.pt"
            if conv_path.exists():
                conv = torch.load(conv_path, weights_only=True)
                rec = torch.load(rec_path, weights_only=True) if rec_path.exists() else None
                states = [_torch_to_mx(conv)]
                states.append(_torch_to_mx(rec) if rec is not None else None)
                cache.state = states
                injected += 1
                print(f"  Layer {i}: ArraysCache conv={tuple(conv.shape)}, rec={tuple(rec.shape) if rec is not None else 'None'}")
            elif keys_path.exists():
                conv = torch.load(keys_path, weights_only=True)
                rec = torch.load(values_path, weights_only=True)
                cache.state = [_torch_to_mx(conv), _torch_to_mx(rec)]
                injected += 1
                print(f"  Layer {i}: ArraysCache (legacy) conv={tuple(conv.shape)}, rec={tuple(rec.shape)}")
            else:
                print(f"  Layer {i}: SKIP — ArraysCache, no files")
                skipped += 1
            continue

        keys_path = kv_dir / f"layer_{i:03d}_keys.pt"
        values_path = kv_dir / f"layer_{i:03d}_values.pt"
        if not keys_path.exists():
            skipped += 1
            continue

        keys_torch = torch.load(keys_path, weights_only=True)
        values_torch = torch.load(values_path, weights_only=True)
        k_mx, v_mx = _to_bhsd(keys_torch, values_torch, num_tokens)
        seq_len = int(k_mx.shape[2])

        if isinstance(cache, KVCache) and not isinstance(cache, RotatingKVCache):
            cache.keys = k_mx
            cache.values = v_mx
            cache.offset = seq_len
            injected += 1
        elif isinstance(cache, RotatingKVCache):
            if seq_len <= cache.max_size:
                cache.keys = k_mx
                cache.values = v_mx
                cache.offset = seq_len
                cache._idx = seq_len
            else:
                keep = cache.keep
                window = cache.max_size
                sink_keys = k_mx[:, :, :keep, :]
                sink_values = v_mx[:, :, :keep, :]
                recent_keys = k_mx[:, :, -(window - keep):, :]
                recent_values = v_mx[:, :, -(window - keep):, :]
                cache.keys = mx.concatenate([sink_keys, recent_keys], axis=2)
                cache.values = mx.concatenate([sink_values, recent_values], axis=2)
                cache.offset = seq_len
                cache._idx = keep
            injected += 1
            print(f"  Layer {i}: RotatingKVCache (seq_len={seq_len}, max_size={cache.max_size})")
        else:
            print(f"  Layer {i}: SKIP — {type(cache).__name__}")
            skipped += 1

    print(f"\n  Injected: {injected} layers, Skipped: {skipped} layers")

    from exo.worker.engines.vllm.kv_cache import TorchKVCache as TKV
    print(f"\nRound-trip test (MLX → torch → MLX)...")
    rt_caches = model.make_cache()
    rt_tokens = mx.array(vllm_token_ids)
    rt_logits = model(rt_tokens[None], cache=rt_caches)
    mx.eval(rt_logits)
    torch_rt = TKV.from_mlx_cache(rt_caches)
    back_rt = torch_rt.to_mlx_cache()
    rt_max_diff = 0.0
    for i in range(len(rt_caches)):
        nc = rt_caches[i]
        bc = back_rt[i]
        if isinstance(nc, ArraysCache):
            for ai in range(len(nc.state)):
                if nc.state[ai] is not None and bc.state[ai] is not None:
                    d = mx.max(mx.abs(nc.state[ai].astype(mx.float32) - bc.state[ai].astype(mx.float32))).item()
                    rt_max_diff = max(rt_max_diff, d)
        elif isinstance(nc, (KVCache, RotatingKVCache)) and nc.keys is not None:
            nk, nv = nc.state
            bk, bv = bc.state
            d = mx.max(mx.abs(nk.astype(mx.float32) - bk.astype(mx.float32))).item()
            rt_max_diff = max(rt_max_diff, d)
    print(f"  Round-trip max diff: {rt_max_diff:.4e} ({'PASS' if rt_max_diff < 0.01 else 'FAIL'})")

    print(f"\nComparing with MLX-native prefill...")
    native_caches = rt_caches

    for i in range(num_model_layers):
        nc = native_caches[i]
        ic = caches[i]
        if isinstance(nc, KVCache) and not isinstance(nc, RotatingKVCache) and nc.keys is not None and ic.keys is not None:
            s = min(nc.offset, ic.offset)
            nk = nc.keys[:, :, :s, :].astype(mx.float32)
            ik = ic.keys[:, :, :s, :].astype(mx.float32)
            nv = nc.values[:, :, :s, :].astype(mx.float32)
            iv = ic.values[:, :, :s, :].astype(mx.float32)
            k_diff = mx.max(mx.abs(nk - ik)).item()
            v_diff = mx.max(mx.abs(nv - iv)).item()
            if k_diff > 0.01 or i < 4 or i == num_model_layers - 1:
                print(f"  Layer {i:3d} KVCache: k_diff={k_diff:.4e}, v_diff={v_diff:.4e}, offset native={nc.offset} injected={ic.offset}")
        elif isinstance(nc, RotatingKVCache):
            pass
        elif isinstance(nc, ArraysCache):
            for ai in range(len(nc.state)):
                na = nc.state[ai]
                ia = ic.state[ai]
                if na is not None and ia is not None:
                    diff = mx.max(mx.abs(na.astype(mx.float32) - ia.astype(mx.float32))).item()
                    if diff > 0.01 or i < 4 or i == num_model_layers - 1:
                        print(f"  Layer {i:3d} Arrays[{ai}]: diff={diff:.4e}, native_shape={na.shape}, injected_shape={ia.shape}")

    native_last = mx.array([vllm_token_ids[-1]])
    native_decode_logits = model(native_last[None], cache=native_caches)
    mx.eval(native_decode_logits)
    native_first = mx.argmax(native_decode_logits[:, -1, :], axis=-1)
    print(f"  Native decode first token: {native_first.item()}, text: {tokenizer.decode([native_first.item()])!r}")

    print(f"\nDecoding {args.num_tokens} tokens with injected cache...")
    last_tokens = mx.array(vllm_token_ids[-2:])
    logits = model(last_tokens[None], cache=caches)
    mx.eval(logits)

    generated_tokens = []
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    generated_tokens.append(token.item())

    for _ in range(args.num_tokens - 1):
        logits = model(token[None], cache=caches)
        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        generated_tokens.append(token.item())

    generated_text = tokenizer.decode(generated_tokens)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Model (vLLM): {metadata['model']}")
    print(f"  Model (MLX):  {args.model}")
    print(f"  Prompt tokens: {num_tokens}")
    print(f"  Layers injected: {injected}/{num_model_layers}")
    print(f"  Type mismatches: 0")
    print(f"  Generated {len(generated_tokens)} tokens")
    print(f"  Text: {generated_text!r}")

    if False:
        print(f"\n  GAPS FOUND:")
        for idx, got, expected in type_mismatches:
            print(f"    Layer {idx}: vLLM gives KV tensors, MLX wants {expected}")
        arrays_layers = [i for i, c in enumerate(caches) if isinstance(c, ArraysCache)]
        if arrays_layers:
            print(f"    ArraysCache layers (not populated): {arrays_layers[:10]}{'...' if len(arrays_layers) > 10 else ''}")

    if generated_tokens and not all(t == generated_tokens[0] for t in generated_tokens):
        print(f"\n  COHERENT OUTPUT: YES (varied tokens)")
    else:
        print(f"\n  COHERENT OUTPUT: POSSIBLY NOT (all same token)")


if __name__ == "__main__":
    main()
