"""Test hybrid prefix cache: _extract_vllm_cache for attn + captured SSM for mamba."""
import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
from exo.worker.engines.vllm.growable_cache import patch_vllm, set_prefix_cache

patch_vllm()
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

KVConnectorFactory.register_connector("StreamingConnector", "exo.disaggregated.streaming_connector", "StreamingConnector")
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

MODEL = os.path.expanduser("~/.local/share/exo/models/Sehyo--Qwen3.5-35B-A3B-NVFP4")
GEN = 600
ea = EngineArgs(model=MODEL, served_model_name="test", gpu_memory_utilization=0.05, trust_remote_code=False,
    load_format="fastsafetensors", enable_prefix_caching=True, attention_backend="FLASH_ATTN",
    compilation_config={"cudagraph_mode": "none"}, disable_log_stats=True, max_num_batched_tokens=4096,
    kv_transfer_config={"kv_connector": "StreamingConnector", "kv_role": "kv_both"},
    disable_hybrid_kv_cache_manager=False)
engine = LLMEngine.from_engine_args(ea)
tok = engine.get_tokenizer()
from exo.worker.engines.mlx.cache import KVPrefixCache

pc = KVPrefixCache(group=None)
set_prefix_cache(pc)

from exo.disaggregated.prefill_server import (
    _extract_vllm_cache,
    _gdn_call_idx,
    _gdn_states,
    _init_gdn_layer_order,
    _patch_gdn_capture,
    _ssm_call_idx,
)
from exo.disaggregated.streaming_connector import reset_shared_queue

_patch_gdn_capture()
_init_gdn_layer_order()
print("Engine loaded")

article = ("The European Union announced sweeping new regulations on artificial intelligence. " * 500)
tids = tok.encode(article)[:22000]
msgs = [{"role": "user", "content": tok.decode(tids) + "\nSummarize the key points of this article."}]
tids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
ptids = tids[:-2]
print(f"Prompt: {len(ptids)} tokens")

reset_shared_queue()
_gdn_states.clear()
_gdn_call_idx[0] = 0
_ssm_call_idx[0] = 0

engine.add_request("r1", {"prompt_token_ids": ptids}, SamplingParams(max_tokens=2, temperature=0.7))
done = False
tc = None
while engine.has_unfinished_requests() and not done:
    for out in engine.step():
        if out.outputs and out.outputs[0].token_ids:
            tc = _extract_vllm_cache(engine, "r1", len(ptids))
            engine.abort_request(["r1"])
            done = True; break
print(f"Extracted: {tc.num_layers if tc else 'NONE'} layers")

if tc and _gdn_states:
    from exo.worker.engines.vllm.kv_cache import ArraysLayerState, KVLayerState
    replaced = 0
    for layer_idx in sorted(_gdn_states.keys()):
        state = _gdn_states[layer_idx]
        arrays = []
        if "conv" in state: arrays.append(state["conv"])
        if "ssm" in state: arrays.append(state["ssm"])
        if arrays and layer_idx < len(tc.layers):
            tc.layers[layer_idx] = ArraysLayerState(arrays=arrays)
            replaced += 1
    print(f"Replaced {replaced} GDN layers with clean prefill state")
    kv_c = sum(1 for l in tc.layers if isinstance(l, KVLayerState) and l.keys.numel() > 0)
    arr_c = sum(1 for l in tc.layers if isinstance(l, ArraysLayerState))
    print(f"Final cache: {kv_c} KV layers, {arr_c} Arrays layers")

import mlx.core as mx

pc.add_kv_cache(mx.array(ptids), tc, None)
print("Stored hybrid cache")

engine.add_request("r2", {"prompt_token_ids": ptids}, SamplingParams(max_tokens=GEN, temperature=0.7))
t2 = time.perf_counter()
prev = 0; text2 = ""; done2 = False
while engine.has_unfinished_requests() and not done2:
    for out in engine.step():
        if out.outputs:
            prev = len(out.outputs[0].token_ids)
            if out.outputs[0].text: text2 = out.outputs[0].text
        if out.finished: done2 = True; break
e2 = time.perf_counter() - t2
print(f"\nRequest 2: {prev} tokens in {e2:.1f}s ({prev/max(e2,0.01):.1f} tok/s)")
print(f"Output: {text2[:500]}")

keywords = ["regulation", "AI", "high-risk", "compliance", "transparency", "ban", "EU", "framework"]
hits = sum(1 for kw in keywords if kw.lower() in text2.lower())
print(f"\nKeyword hits: {hits}/{len(keywords)}")
if hits >= 2:
    print("PASS")
else:
    print(f"FAIL ({hits} hits)")
    exit(1)
