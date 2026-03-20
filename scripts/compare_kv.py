import sys
sys.path.insert(0, "src")
from exo.worker.engines.mlx.gdn_softplus_patch import patch_gdn_softplus
from exo.worker.engines.mlx.yarn_rope_patch import patch_yarn_rope
patch_gdn_softplus()
patch_yarn_rope()
import mlx.core as mx
import torch
import socket
from pathlib import Path
import json
from collections import defaultdict
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, RotatingKVCache, KVCache
from exo.disaggregated.protocol import read_header, read_message, ArraysState, KVChunk, Done
from exo.disaggregated.prefill_client import _nhd_to_bhsd, _torch_to_mx

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "10.43.0.1:62988"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/Llama-3.2-1B-Instruct-bf16"
MODEL_PATH = sys.argv[3] if len(sys.argv) > 3 else None

model, tok = load(MODEL_PATH or str(Path.home() / ".exo/models" / MODEL.replace("/", "--")))
prompt = "The quick brown fox jumps over the lazy dog. " * 3000
tokens = tok.encode(prompt)
print(f"Tokens: {len(tokens)}")

host, port = ENDPOINT.rsplit(":", 1)
sock = socket.create_connection((host, int(port)), timeout=60)
request = json.dumps({"model": MODEL, "token_ids": tokens, "start_pos": 0}).encode() + b"\n"
sock.sendall(request)
stream = sock.makefile("rb", buffering=65536)
header = read_header(stream)

vllm_kv = defaultdict(list)
vllm_arrays: dict[int, list[torch.Tensor]] = {}
while True:
    msg = read_message(stream, header)
    if msg is None or isinstance(msg, Done):
        break
    if isinstance(msg, KVChunk):
        vllm_kv[msg.layer_idx].append((msg.keys, msg.values))
    elif isinstance(msg, ArraysState):
        vllm_arrays[msg.layer_idx] = msg.arrays
sock.close()

print(f"Received {len(vllm_kv)} KV layers, {len(vllm_arrays)} arrays layers from vLLM")

if hasattr(model, "make_cache"):
    mlx_cache = model.make_cache()
else:
    from mlx_lm.models.cache import make_prompt_cache
    mlx_cache = make_prompt_cache(model)
token_arr = mx.array([tokens[:-2]])
mlx_logits = model(token_arr, cache=mlx_cache)
mx.eval(mlx_logits)

for i in range(min(6, len(mlx_cache))):
    c = mlx_cache[i]
    if isinstance(c, ArraysCache):
        if i in vllm_arrays:
            vllm_arrs = vllm_arrays[i]
            mlx_state = c.state
            print(f"Layer {i} (Arrays): mlx_state={len(mlx_state)} arrays, vllm={len(vllm_arrs)} arrays")
            for ai, (m_arr, v_arr) in enumerate(zip(mlx_state, vllm_arrs)):
                if m_arr is None:
                    continue
                v_mx = _torch_to_mx(v_arr).astype(mx.float32)
                m_f = m_arr.astype(mx.float32)
                if m_f.shape != v_mx.shape:
                    print(f"  [{ai}] SHAPE MISMATCH mlx={m_f.shape} vllm={v_mx.shape}")
                else:
                    d = mx.abs(m_f - v_mx)
                    a = m_f.reshape(-1)
                    b = v_mx.reshape(-1)
                    cos = float(mx.sum(a * b).item()) / (float(mx.sqrt(mx.sum(a * a)).item()) * float(mx.sqrt(mx.sum(b * b)).item()) + 1e-8)
                    print(f"  [{ai}] cosine_sim={cos:.6f} max_diff={mx.max(d).item():.6f} mean_diff={mx.mean(d).item():.6f} shape={m_f.shape}")
        else:
            print(f"Layer {i} (Arrays): no vLLM data")
        continue
    if c.keys is None:
        continue
    mlx_k = c.keys.astype(mx.float32)

    if i not in vllm_kv:
        print(f"Layer {i}: no vLLM data")
        continue
    chunks = vllm_kv[i]
    vk = torch.cat([k for k, v in chunks], dim=0) if len(chunks) > 1 else chunks[0][0]
    vk_mx = _torch_to_mx(vk.permute(1, 0, 2).unsqueeze(0)).astype(mx.float32)

    n = min(mlx_k.shape[2], vk_mx.shape[2])
    diff = mx.abs(mlx_k[:, :, :n, :] - vk_mx[:, :, :n, :])
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    cache_type = "RotatingKV" if isinstance(c, RotatingKVCache) else "KV"
    print(f"Layer {i} ({cache_type}): mlx={mlx_k.shape} vllm={vk_mx.shape} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

    a = mlx_k[:, :, :n, :].reshape(-1)
    b_vec = vk_mx[:, :, :n, :].reshape(-1)
    cos_sim = float(mx.sum(a * b_vec).item()) / (float(mx.sqrt(mx.sum(a * a)).item()) * float(mx.sqrt(mx.sum(b_vec * b_vec)).item()) + 1e-8)
    diff_tensor = mx.abs(mlx_k[:, :, :n, :] - vk_mx[:, :, :n, :])
    max_idx = mx.argmax(diff_tensor.reshape(-1)).item()
    total_elems = diff_tensor.shape[1] * n * diff_tensor.shape[3]
    h_idx = (max_idx // (n * diff_tensor.shape[3])) % diff_tensor.shape[1]
    s_idx = (max_idx // diff_tensor.shape[3]) % n
    d_idx = max_idx % diff_tensor.shape[3]
    print(f"  cosine_sim={cos_sim:.6f} max_diff={max_diff:.4f} at h={h_idx} pos={s_idx} dim={d_idx}: mlx={float(mlx_k[0,h_idx,s_idx,d_idx].item()):.6f} vllm={float(vk_mx[0,h_idx,s_idx,d_idx].item()):.6f}")

    D = mlx_k.shape[3]
    for pos in [0, 100, n-1]:
        mlx_row = [float(mlx_k[0, 0, pos, d].item()) for d in range(D)]
        vllm_row = [float(vk_mx[0, 0, pos, d].item()) for d in range(D)]
        diffs = [abs(mlx_row[d] - vllm_row[d]) for d in range(D)]
        top5 = sorted(range(D), key=lambda d: -diffs[d])[:5]
        print(f"  pos={pos} top5 diff dims: {[(d, f'{diffs[d]:.3f}', f'mlx={mlx_row[d]:.3f}', f'vllm={vllm_row[d]:.3f}') for d in top5]}")

print("\n--- Run 2: cached request ---")
sock2 = socket.create_connection((host, int(port)), timeout=60)
request2 = json.dumps({"model": MODEL, "token_ids": tokens, "start_pos": 0}).encode() + b"\n"
sock2.sendall(request2)
stream2 = sock2.makefile("rb", buffering=65536)

first_byte = stream2.peek(1)[:1]
if first_byte == b"{":
    line2 = stream2.readline()
    print(f"Server error: {json.loads(line2.decode())}")
    sys.exit(1)

header2 = read_header(stream2)
vllm_kv2 = defaultdict(list)
vllm_arrays2: dict[int, list[torch.Tensor]] = {}
total_tokens2 = 0
while True:
    msg = read_message(stream2, header2)
    if msg is None:
        break
    if isinstance(msg, KVChunk):
        vllm_kv2[msg.layer_idx].append((msg.keys, msg.values))
    elif isinstance(msg, ArraysState):
        vllm_arrays2[msg.layer_idx] = msg.arrays
    elif isinstance(msg, Done):
        total_tokens2 = msg.total_tokens
        break
sock2.close()

kv_tokens2 = 0
if vllm_kv2:
    first_layer = next(iter(vllm_kv2.values()))
    kv_tokens2 = sum(k.shape[0] for k, v in first_layer)
print(f"Received {len(vllm_kv2)} KV layers ({kv_tokens2} tokens), {len(vllm_arrays2)} arrays layers, total_tokens={total_tokens2}")

for i in range(min(6, len(mlx_cache))):
    c = mlx_cache[i]
    if isinstance(c, ArraysCache):
        if i in vllm_arrays2:
            vllm_arrs = vllm_arrays2[i]
            mlx_state = c.state
            for ai, (m_arr, v_arr) in enumerate(zip(mlx_state, vllm_arrs)):
                if m_arr is None:
                    continue
                v_mx = _torch_to_mx(v_arr).astype(mx.float32)
                m_f = m_arr.astype(mx.float32)
                if m_f.shape != v_mx.shape:
                    print(f"Layer {i} [{ai}] SHAPE MISMATCH mlx={m_f.shape} vllm={v_mx.shape}")
                else:
                    a2 = m_f.reshape(-1)
                    b2 = v_mx.reshape(-1)
                    cos2 = float(mx.sum(a2 * b2).item()) / (float(mx.sqrt(mx.sum(a2 * a2)).item()) * float(mx.sqrt(mx.sum(b2 * b2)).item()) + 1e-8)
                    print(f"Layer {i} (Arrays) [{ai}] cosine_sim={cos2:.6f} shape={m_f.shape}")
        continue
    if c.keys is None or i not in vllm_kv2:
        continue
    mlx_k = c.keys.astype(mx.float32)
    chunks = vllm_kv2[i]
    vk = torch.cat([k for k, v in chunks], dim=0) if len(chunks) > 1 else chunks[0][0]
    vk_mx = _torch_to_mx(vk.permute(1, 0, 2).unsqueeze(0)).astype(mx.float32)
    n = min(mlx_k.shape[2], vk_mx.shape[2])
    a2 = mlx_k[:, :, :n, :].reshape(-1)
    b2 = vk_mx[:, :, :n, :].reshape(-1)
    cos2 = float(mx.sum(a2 * b2).item()) / (float(mx.sqrt(mx.sum(a2 * a2)).item()) * float(mx.sqrt(mx.sum(b2 * b2)).item()) + 1e-8)
    print(f"Layer {i} (KV) cosine_sim={cos2:.6f} mlx={mlx_k.shape} vllm={vk_mx.shape}")

if len(vllm_kv2) > 0:
    print("PASS")
else:
    print("FAIL")
