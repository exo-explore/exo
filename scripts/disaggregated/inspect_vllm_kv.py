"""Inspect vLLM KV cache structure per-layer after prefill.

Runs on DGX Spark. Prints per-layer shapes, dtypes, kv_cache_config,
and layer_to_group mapping to understand what vLLM stores for each
model architecture (standard attention, sliding window, GatedDeltaNet).

Usage:
  uv run python scripts/disaggregated/inspect_vllm_kv.py --model ~/.local/share/exo/models/openai--gpt-oss-20b
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from exo.worker.runner.bootstrap import _ensure_cuda_libs
_ensure_cuda_libs()

import torch


def _build_layer_groups(kv_cache_config):
    group_lookup = {}
    for group_idx, group_spec in enumerate(kv_cache_config.kv_cache_groups):
        for layer_name in group_spec.layer_names:
            group_lookup[layer_name] = group_idx

    layer_to_group = []
    for tensor_spec in kv_cache_config.kv_cache_tensors:
        for name in tensor_spec.shared_by:
            layer_to_group.append(group_lookup[name])
    return layer_to_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--prompt", default="Hello, world! How are you today?", help="Prompt to prefill")
    args = parser.parse_args()

    from exo.worker.engines.vllm.vllm_generator import load_vllm_engine
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    print(f"Loading vLLM engine from {args.model}...")
    engine, _, prefix_cache = load_vllm_engine(
        model_path=args.model,
        model_id=args.model,
        trust_remote_code=True,
    )
    print("Engine loaded.\n")

    from vllm import SamplingParams

    tokenizer = engine.get_tokenizer()
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"Prompt: {args.prompt!r}")
    print(f"Token IDs: {len(token_ids)} tokens\n")

    request_id = "inspect-test"
    params = SamplingParams(max_tokens=1, detokenize=False)
    engine.add_request(request_id, {"prompt_token_ids": token_ids}, params)

    while engine.has_unfinished_requests():
        engine.step()

    model_runner = get_model_runner()
    if model_runner is None:
        print("ERROR: model_runner is None")
        return

    print("=" * 70)
    print("PER-LAYER KV CACHE TENSORS (model_runner.kv_caches)")
    print("=" * 70)
    kv_caches = model_runner.kv_caches
    for i, kv in enumerate(kv_caches):
        if isinstance(kv, list):
            shapes = [t.shape for t in kv]
            dtypes = [t.dtype for t in kv]
            print(f"  Layer {i:3d}: list of {len(kv)} tensors — shapes={shapes}, dtypes={dtypes}")
        elif isinstance(kv, torch.Tensor):
            print(f"  Layer {i:3d}: shape={tuple(kv.shape)}, dtype={kv.dtype}, device={kv.device}")
        else:
            print(f"  Layer {i:3d}: type={type(kv).__name__}")
    print(f"\n  Total layers with KV: {len(kv_caches)}\n")

    engine_core = engine.engine_core.engine_core
    kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config

    print("=" * 70)
    print("KV CACHE CONFIG")
    print("=" * 70)

    print(f"\n  Number of KV cache groups: {len(kv_cache_config.kv_cache_groups)}")
    for gi, group in enumerate(kv_cache_config.kv_cache_groups):
        print(f"\n  Group {gi}:")
        print(f"    Layer names ({len(group.layer_names)}):")
        for name in group.layer_names[:5]:
            print(f"      {name}")
        if len(group.layer_names) > 5:
            print(f"      ... and {len(group.layer_names) - 5} more")

    print(f"\n  Number of KV cache tensors: {len(kv_cache_config.kv_cache_tensors)}")
    for ti, tensor_spec in enumerate(kv_cache_config.kv_cache_tensors):
        shared = tensor_spec.shared_by[:3]
        extra = f" ... +{len(tensor_spec.shared_by)-3}" if len(tensor_spec.shared_by) > 3 else ""
        print(f"  Tensor {ti}: shared_by={shared}{extra}")

    layer_to_group = _build_layer_groups(kv_cache_config)
    print(f"\n  layer_to_group ({len(layer_to_group)} entries): {layer_to_group[:10]}{'...' if len(layer_to_group) > 10 else ''}")

    coordinator = engine_core.scheduler.kv_cache_manager.coordinator
    null_block = coordinator.block_pool.null_block

    internal_id = None
    for mgr in coordinator.single_type_managers:
        for key in mgr.req_to_blocks:
            if str(key).startswith(request_id):
                internal_id = str(key)
                break
        if internal_id:
            break

    if internal_id:
        print(f"\n  Request internal_id: {internal_id}")
        for gi, mgr in enumerate(coordinator.single_type_managers):
            blocks = mgr.req_to_blocks.get(internal_id)
            if blocks:
                real_blocks = [b for b in blocks if b is not null_block and not b.is_null]
                null_count = len(blocks) - len(real_blocks)
                print(f"  Group {gi}: {len(real_blocks)} real blocks, {null_count} null blocks, block_size={mgr.block_size}")
            else:
                print(f"  Group {gi}: no blocks")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  KV cache layers: {len(kv_caches)}")
    print(f"  KV cache groups: {len(kv_cache_config.kv_cache_groups)}")
    print(f"  Layer-to-group mapping entries: {len(layer_to_group)}")
    unique_shapes = set()
    for kv in kv_caches:
        if isinstance(kv, torch.Tensor):
            unique_shapes.add(tuple(kv.shape))
    print(f"  Unique tensor shapes: {unique_shapes}")


if __name__ == "__main__":
    main()
