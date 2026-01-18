# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 8192
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True

# Multi-Token Prediction (MTP) configuration for DeepSeek V3
# MTP enables speculative decoding using the model's built-in draft layer
MTP_ENABLED: bool = True  # Feature flag to enable/disable MTP
MTP_NUM_DRAFT_TOKENS: int = 1  # Number of tokens to draft (vLLM reports k=1 is optimal)
