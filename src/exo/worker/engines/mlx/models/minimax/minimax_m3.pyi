"""Type stubs for exo's MiniMax M3 / M3.1 model (minimax_m3.py next to this file)."""

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx_lm.models import minimax_m3_vl as upstream

# The Metal kernels (None without Metal), which the tests compare to the ops they replace.
_MetalKernel = Callable[..., list[mx.array]]
_route_kernel: _MetalKernel | None
_block_scores_kernel: _MetalKernel | None
_attend_to_blocks_kernel: _MetalKernel | None
_merge_blocks_kernel: _MetalKernel | None
_MAX_BLOCKS: int

class GemmaRMSNorm(upstream.GemmaRMSNorm):
    def __call__(self, x: mx.array) -> mx.array: ...

def _route_ops(
    gates: mx.array, bias: mx.array, top_k: int, scaling: float, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]: ...
def _route(
    gates: mx.array, bias: mx.array, top_k: int, scaling: float, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]: ...
def _combine(shared: mx.array, *parts: mx.array) -> mx.array: ...
def _sparse_decode_kernels_apply(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    index_keys: mx.array,
    topk: int,
    block_size: int,
    n_keys: int,
) -> bool: ...
def _sparse_decode_kernels(
    q: mx.array,
    index_q: mx.array,
    keys: mx.array,
    values: mx.array,
    index_keys: mx.array,
    mask: mx.array | None,
    n_keys: int,
    block_size: int,
    topk: int,
    local_blocks: int,
    scale: float,
) -> mx.array: ...

class SparseMoeBlock(upstream.SparseMoeBlock):
    top_k: int
    def __init__(self, args: upstream.TextArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Attention(upstream.Attention):
    def __init__(self, args: upstream.TextArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array: ...

class DecoderLayer(upstream.DecoderLayer):
    def __init__(self, args: upstream.TextArgs, layer_idx: int) -> None: ...

class TextModel(upstream.TextModel):
    def __init__(self, args: upstream.TextArgs) -> None: ...

class LanguageModel(upstream.LanguageModel):
    def __init__(self, config: upstream.TextArgs) -> None: ...

class Model(upstream.Model):
    def __init__(self, config: upstream.ModelArgs) -> None: ...
