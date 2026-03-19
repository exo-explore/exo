import sys
sys.path.insert(0, "src")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import RotatingKVCache, KVCache

model, tok = load("mlx-community/gpt-oss-20b-MXFP4-Q8")

prompt = "Hello " * 2000
tokens = tok.encode(prompt)
print(f"Tokens: {len(tokens)}")

cache = model.make_cache()
token_arr = mx.array([tokens])
logits = model(token_arr, cache=cache)
mx.eval(logits)

for i, c in enumerate(cache[:6]):
    if isinstance(c, KVCache) and not isinstance(c, RotatingKVCache) and c.keys is not None:
        k = c.keys.astype(mx.float32)
        print(f"Layer {i} KVCache: shape={c.keys.shape} offset={c.offset} first=[{float(k[0,0,0,0]):.6f}, {float(k[0,0,0,1]):.6f}] last=[{float(k[0,0,-1,-2]):.6f}, {float(k[0,0,-1,-1]):.6f}]")
    elif isinstance(c, RotatingKVCache) and c.keys is not None:
        k = c.keys.astype(mx.float32)
        print(f"Layer {i} RotatingKV: shape={c.keys.shape} _idx={c._idx} offset={c.offset} first=[{float(k[0,0,0,0]):.6f}, {float(k[0,0,0,1]):.6f}]")
