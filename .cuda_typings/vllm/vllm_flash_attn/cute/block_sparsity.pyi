import cutlass.cute as cute
import torch
from typing import Callable, NamedTuple
from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
    get_broadcast_dims as get_broadcast_dims,
    to_cute_tensor as to_cute_tensor,
)

def ceildiv(a: int, b: int) -> int: ...

class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None
    full_block_idx: cute.Tensor | None
    def __new_from_mlir_values__(self, values): ...

class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = ...
    full_block_idx: torch.Tensor | None = ...
    block_size: tuple[int, int] | None = ...

def get_block_sparse_expected_shapes(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]: ...
def infer_block_sparse_expected_shapes(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    context: str,
    sparse_block_size_q: int | None = None,
    sparse_block_size_kv: int | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], int]: ...
def get_block_sparse_expected_shapes_bwd(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]: ...
def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch,
    *,
    expected_count_shape: tuple[int, int, int],
    expected_index_shape: tuple[int, int, int, int],
    context: str | None = None,
    hint: str | Callable[[], str] | None = None,
) -> BlockSparseTensorsTorch: ...
def is_block_sparsity_enabled(tensors: BlockSparseTensorsTorch) -> bool: ...
def get_block_sparse_broadcast_pattern(
    tensors: BlockSparseTensorsTorch,
) -> tuple[tuple[bool, ...], ...] | None: ...
def normalize_block_sparse_config(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    q_stage: int,
) -> tuple[BlockSparseTensorsTorch, tuple[tuple[bool, ...], ...] | None, int]: ...
def normalize_block_sparse_config_bwd(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    subtile_factor: int,
) -> tuple[BlockSparseTensorsTorch, tuple[tuple[bool, ...], ...] | None]: ...
def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch, enable_tvm_ffi: bool = True
) -> BlockSparseTensors | None: ...
def fast_sampling(mask_mod): ...
