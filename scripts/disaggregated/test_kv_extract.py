"""Extract KV cache per-layer from vLLM using a real KVConnector.

Patches vLLM to allow KVConnector on hybrid models (attention + GDN).

Usage:
  uv run python scripts/disaggregated/test_kv_extract.py --model ~/.local/share/exo/models/Qwen--Qwen3.5-2B --output /tmp/kv_cache_qwen35/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from exo.worker.runner.bootstrap import _ensure_cuda_libs
_ensure_cuda_libs()

import torch


def _patch_vllm_for_connector():
    """Patch vLLM to allow KVConnector on hybrid models."""
    from vllm.v1.core import kv_cache_utils

    original_unify = kv_cache_utils.unify_hybrid_kv_cache_specs

    def patched_unify(kv_cache_spec):
        try:
            original_unify(kv_cache_spec)
        except ValueError:
            pass

    kv_cache_utils.unify_hybrid_kv_cache_specs = patched_unify

    from vllm.v1.core.sched import scheduler as sched_mod
    original_connector_finished = sched_mod.Scheduler._connector_finished

    def patched_connector_finished(self, request):
        return False, None

    sched_mod.Scheduler._connector_finished = patched_connector_finished

    from capture_connector import CaptureConnector
    from vllm.distributed.kv_transfer.kv_connector import factory

    original_get = factory.KVConnectorFactory._get_connector_class_with_compat

    @classmethod
    def patched_get(cls, kv_transfer_config):
        if "capture_connector" in (kv_transfer_config.kv_connector or ""):
            return CaptureConnector, None
        return original_get.__func__(cls, kv_transfer_config)

    factory.KVConnectorFactory._get_connector_class_with_compat = patched_get


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    _lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor. Ut ullamcorper, ligula ut dictum pharetra, nisi nunc fringilla magna, in commodo elit erat nec turpis. Ut pharetra augue nec augue. Nam elit agna, endrerit sit amet, tincidunt ac, viverra sed, nulla. Donec porta diam eu massa. Quisque diam lorem, interdum vitae, dapibus ac, scelerisque vitae, pede. Donec eget tellus non erat lacinia fermentum. Donec in velit vel ipsum auctor pulvinar. Vestibulum iaculis lacinia est. Proin dictum elementum velit. Fusce euismod consequat ante. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Pellentesque sed dolor. Aliquam congue fermentum nisl. Mauris accumsan nulla vel diam. Sed in lacus ut enim adipiscing aliquet. Nulla venenatis. In pede mi, aliquet sit amet, euismod in, auctor ut, ligula. Aliquam dapibus tincidunt metus. Praesent justo dolor, lobortis quis, lobortis dignissim, pulvinar ac, lorem. "
    parser.add_argument("--prompt", default=_lorem * 21 + "Now answer this question: What is the capital of France and why is it historically significant? Give a detailed answer.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_vllm_for_connector()

    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from exo.worker.engines.vllm.growable_cache import patch_vllm, set_prefix_cache
    from exo.worker.engines.mlx.cache import KVPrefixCache

    patch_vllm()

    prefix_cache = KVPrefixCache(group=None)
    set_prefix_cache(prefix_cache)

    engine_args = EngineArgs(
        model=args.model,
        served_model_name=args.model,
        gpu_memory_utilization=0.05,
        trust_remote_code=True,
        load_format="fastsafetensors",
        enable_prefix_caching=False,
        attention_backend="TRITON_ATTN",
        enforce_eager=True,
        disable_log_stats=True,
        kv_transfer_config={
            "kv_connector": "capture_connector:CaptureConnector",
            "kv_role": "kv_both",
        },
    )

    print(f"Loading engine with KVConnector...")
    engine = LLMEngine.from_engine_args(engine_args)
    print("Engine loaded.")

    from exo.worker.engines.vllm.growable_cache import get_model_runner
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from capture_connector import captured_layers as gdn_captured

    model_runner = get_model_runner()

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn as orig_causal_conv1d_fn
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc_mod

    gdn_states: dict[int, dict[str, torch.Tensor]] = {}
    gdn_call_idx = [0]
    gdn_layer_order = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]

    def patched_causal_conv1d_fn(*args, conv_states=None, cache_indices=None, **kwargs):
        result = orig_causal_conv1d_fn(*args, conv_states=conv_states, cache_indices=cache_indices, **kwargs)
        if conv_states is not None and cache_indices is not None:
            x = args[0] if args else None
            if x is not None and x.shape[0] <= 100:
                return result
            import time as _time
            t0 = _time.perf_counter()
            torch.cuda.synchronize()
            t_sync = _time.perf_counter() - t0
            ci = cache_indices[0].item() if cache_indices.numel() > 0 else 0
            idx = gdn_call_idx[0]
            layer_idx = gdn_layer_order[idx % len(gdn_layer_order)]
            t1 = _time.perf_counter()
            conv_at_ci = conv_states[ci:ci+1].transpose(-1, -2).contiguous().cpu()
            t_copy = _time.perf_counter() - t1
            gdn_states.setdefault(layer_idx, {})["conv"] = conv_at_ci
            gdn_states[layer_idx]["ci"] = ci
            if gdn_call_idx[0] < 3:
                print(f"    [gdn save] sync={t_sync*1000:.1f}ms copy={t_copy*1000:.1f}ms layer={layer_idx}")
            gdn_call_idx[0] += 1
        return result

    cc_mod.causal_conv1d_fn = patched_causal_conv1d_fn
    for mod in list(sys.modules.values()):
        if mod is None or mod is cc_mod:
            continue
        if hasattr(mod, 'causal_conv1d_fn') and mod.causal_conv1d_fn is orig_causal_conv1d_fn:
            mod.causal_conv1d_fn = patched_causal_conv1d_fn
    print(f"  Patched causal_conv1d_fn")

    from exo.worker.engines.vllm.vllm_generator import VllmBatchEngine
    from exo.shared.types.text_generation import TextGenerationTaskParams, InputMessage
    from exo.shared.types.tasks import TaskId

    batch_engine = VllmBatchEngine(engine=engine, model_id=args.model, prefix_cache=prefix_cache)

    task = TextGenerationTaskParams(
        model=args.model,
        input=[InputMessage(role="user", content=args.prompt)],
        max_completion_tokens=1,
    )

    task_id = batch_engine.submit(task_id=TaskId("extract"), task_params=task, prompt=args.prompt)

    print(f"Running prefill via VllmBatchEngine...")
    t0 = time.perf_counter()
    while batch_engine.has_work:
        results = batch_engine.step()
        for tid, resp in results:
            print(f"  Prefill done in {(time.perf_counter()-t0)*1000:.0f}ms")
            batch_engine.cancel([tid])
            break
        if results:
            break
    t1 = time.perf_counter()
    print(f"Total: {(t1-t0)*1000:.0f}ms")

    prompt_mx = prefix_cache.prompts[0] if prefix_cache.prompts else None
    token_ids = [int(x) for x in prompt_mx.tolist()] if prompt_mx is not None else []

    from capture_connector import captured_layers

    print(f"\nCaptured {len(captured_layers)} layers via save_kv_layer:")
    for name in sorted(captured_layers.keys()):
        v = captured_layers[name]
        if isinstance(v, list):
            print(f"  {name}: {[tuple(t.shape) for t in v]}")
        elif isinstance(v, torch.Tensor):
            print(f"  {name}: {tuple(v.shape)}")
        else:
            print(f"  {name}: {type(v).__name__}")

    num_tokens = len(token_ids)
    print(f"  Chat-templated prompt: {num_tokens} tokens")
    total_layers = 24

    for f_old in out_dir.glob("layer_*"):
        f_old.unlink()

    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "num_tokens": num_tokens,
        "token_ids": token_ids,
        "num_layers": total_layers,
        "layers": [],
    }

    print(f"\nSaving {total_layers} layers...")

    torch.cuda.synchronize()
    for layer_idx in sorted(gdn_states.keys()):
        ci = gdn_states[layer_idx]["ci"]
        kv = model_runner.kv_caches[layer_idx]
        if isinstance(kv, (list, tuple)) and len(kv) > 1:
            rec_pool = kv[1]
            rec = rec_pool[ci:ci+1].cpu().clone()
            gdn_states[layer_idx]["rec"] = rec

    for li in range(total_layers):
        if li in gdn_states:
            s = gdn_states[li]
            conv = s.get("conv")
            rec = s.get("rec")
            torch.save(conv, out_dir / f"layer_{li:03d}_conv.pt")
            if rec is not None:
                torch.save(rec, out_dir / f"layer_{li:03d}_rec.pt")
            metadata["layers"].append({"type": "gdn", "conv": list(conv.shape), "rec": list(rec.shape) if rec is not None else None})
            print(f"  Layer {li}: GDN conv={tuple(conv.shape)}, rec={tuple(rec.shape) if rec is not None else 'None'}")
        else:
            attn_name = None
            for n in captured_layers:
                parts = n.split(".")
                for pi, p in enumerate(parts):
                    if p == "layers" and pi + 1 < len(parts) and parts[pi + 1] == str(li):
                        attn_name = n
                        break
            if attn_name and isinstance(captured_layers[attn_name], dict):
                kv = captured_layers[attn_name]
                torch.save(kv["keys"], out_dir / f"layer_{li:03d}_keys.pt")
                torch.save(kv["values"], out_dir / f"layer_{li:03d}_values.pt")
                if "last_chunk_keys" in kv:
                    torch.save(kv["last_chunk_keys"], out_dir / f"layer_{li:03d}_keys_last.pt")
                    torch.save(kv["last_chunk_values"], out_dir / f"layer_{li:03d}_values_last.pt")
                metadata["layers"].append({"type": "kv", "keys_shape": list(kv["keys"].shape), "values_shape": list(kv["values"].shape)})
                print(f"  Layer {li}: KV keys={tuple(kv['keys'].shape)}, values={tuple(kv['values'].shape)}")
            else:
                metadata["layers"].append({"type": "missing"})
                print(f"  Layer {li}: MISSING")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {out_dir}/metadata.json")


if __name__ == "__main__":
    main()
