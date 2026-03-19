import sys
sys.path.insert(0, "src")
import mlx.core as mx
import torch
import socket
from pathlib import Path
import json
from collections import defaultdict
from mlx_lm import load
from mlx_lm.models.cache import RotatingKVCache, KVCache
from exo.disaggregated.protocol import read_header, read_message, KVChunk, Done
from exo.disaggregated.prefill_client import _nhd_to_bhsd, _torch_to_mx

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "10.43.0.1:62988"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/Llama-3.2-1B-Instruct-bf16"
MODEL_PATH = sys.argv[3] if len(sys.argv) > 3 else None

model, tok = load(MODEL_PATH or str(Path.home() / ".exo/models" / MODEL.replace("/", "--")))
prompt = "The quick brown fox jumps over the lazy dog. " * 300
tokens = tok.encode(prompt)
print(f"Tokens: {len(tokens)}")

host, port = ENDPOINT.rsplit(":", 1)
sock = socket.create_connection((host, int(port)), timeout=60)
request = json.dumps({"model": MODEL, "token_ids": tokens, "start_pos": 0}).encode() + b"\n"
sock.sendall(request)
stream = sock.makefile("rb", buffering=65536)
header = read_header(stream)

vllm_kv = defaultdict(list)
while True:
    msg = read_message(stream, header)
    if msg is None or isinstance(msg, Done):
        break
    if isinstance(msg, KVChunk):
        vllm_kv[msg.layer_idx].append((msg.keys, msg.values))
sock.close()

print(f"Received {len(vllm_kv)} layers from vLLM")

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
    h = (max_idx // (n * diff_tensor.shape[3])) % diff_tensor.shape[1]
    s = (max_idx // diff_tensor.shape[3]) % n
    d = max_idx % diff_tensor.shape[3]
    print(f"  cosine_sim={cos_sim:.6f} max_diff={max_diff:.4f} at h={h} pos={s} dim={d}: mlx={float(mlx_k[0,h,s,d].item()):.6f} vllm={float(vk_mx[0,h,s,d].item()):.6f}")
    print(f"  mean_diff={mean_diff:.6f} p50={float(mx.sort(diff_tensor.reshape(-1))[diff_tensor.size // 2].item()):.6f} p99={float(mx.sort(diff_tensor.reshape(-1))[int(diff_tensor.size * 0.99)].item()):.6f}")
