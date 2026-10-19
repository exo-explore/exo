import os

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"

# Number of bits to quantize the KV cache to (mlx_lm's QuantizedKVCache, e.g.
# 4 or 8). None (the default) keeps the cache in full precision. Opt-in via
# env var: this path is wired into every generation code path here (both
# make_kv_cache's direct QuantizedKVCache construction and mlx_lm's
# maybe_quantize_kv_cache during stream_generate/pipeline prefill), and
# exo's prefix-cache trim/snapshot logic (cache.py) already handles
# QuantizedKVCache correctly via the shared _BaseCache.trim()/.offset
# interface. What's unverified is real-hardware behavior -- turn on only
# after testing on an actual cluster.
KV_CACHE_BITS: int | None = (
    int(os.environ["EXO_KV_CACHE_BITS"]) if "EXO_KV_CACHE_BITS" in os.environ else None
)
KV_CACHE_GROUP_SIZE: int = int(os.environ.get("EXO_KV_CACHE_GROUP_SIZE", "64"))

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
