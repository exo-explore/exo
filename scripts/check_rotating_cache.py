import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import RotatingKVCache

model, tok = load("mlx-community/gpt-oss-20b-MXFP4-Q8")
cache = model.make_cache()
tokens = mx.ones((1, 5000), dtype=mx.int32)
model(tokens, cache=cache)
mx.eval([c.keys for c in cache if c.keys is not None])
for i, c in enumerate(cache[:4]):
    if isinstance(c, RotatingKVCache):
        print(f"Layer {i}: _idx={c._idx} offset={c.offset} keep={c.keep} max_size={c.max_size} keys={c.keys.shape}")
