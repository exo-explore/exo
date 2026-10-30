"""Type stubs for dsa_attention.py (GLM-5.x attention; basedpyright skips the .py)."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.glm_moe_dsa import ModelArgs

SPARSE_PREFILL_MIN_CONTEXT: int
SPARSE_QUERY_BLOCK: int

def indexer_topk(
    indexer: nn.Module,
    x: mx.array,
    qr: mx.array,
    mask: mx.array | None,
    cache: object | None,
) -> mx.array | None: ...
def blocked_indexer_topk(
    indexer: nn.Module,
    x: mx.array,
    qr: mx.array,
    mask: mx.array | None,
    cache: object | None,
    block: int = 4096,
) -> mx.array | None: ...

class GlmAttention(nn.Module):
    sparse_prefill: bool
    sparse_prefill_min_context: int
    indexer: nn.Module | None
    def __init__(self, config: ModelArgs, layer_idx: int) -> None: ...
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
        prev_topk_indices: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]: ...
