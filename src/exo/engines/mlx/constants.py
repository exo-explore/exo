# TODO: Do we want so many constants?

KV_GROUP_SIZE = 32
KV_BITS = None
ATTENTION_KV_BITS = 4
MAX_TOKENS = 8192
MAX_KV_SIZE = 3200
KEEP_KV_SIZE = 1600
QUANTIZE_MODEL_MODE = "affine"
CACHE_GROUP_SIZE = 64
KV_CACHE_BITS = 8
TEMPERATURE = 1.0

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE = True
# TODO: Do we really want this?
HIDE_THINKING = False
