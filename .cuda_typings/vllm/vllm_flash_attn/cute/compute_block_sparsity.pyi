import cutlass.cute as cute
from _typeshed import Incomplete
from cutlass import Int32
from typing import Callable
from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensors as BlockSparseTensors,
    BlockSparseTensorsTorch as BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors as to_cute_block_sparse_tensors,
)
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK
from vllm.vllm_flash_attn.cute.utils import (
    hash_callable as hash_callable,
    scalar_to_ssa as scalar_to_ssa,
    ssa_to_scalar as ssa_to_scalar,
)

class BlockSparsityKernel:
    mask_mod: Incomplete
    tile_mn: Incomplete
    compute_full_blocks: Incomplete
    use_aux_tensors: Incomplete
    use_fast_sampling: Incomplete
    def __init__(
        self,
        mask_mod: Callable,
        tile_mn: tuple[int, int],
        compute_full_blocks: bool = True,
        use_aux_tensors: bool = False,
        use_fast_sampling: bool = False,
    ) -> None: ...
    num_warps: int
    @cute.jit
    def __call__(
        self,
        blocksparse_tensors: BlockSparseTensors,
        seqlen_q: Int32,
        seqlen_k: Int32,
        aux_tensors: list | None = None,
    ): ...
    @cute.kernel
    def kernel(
        self,
        mask_cnt: cute.Tensor,
        mask_idx: cute.Tensor,
        full_cnt: cute.Tensor,
        full_idx: cute.Tensor,
        num_n_blocks: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        aux_tensors: list | None = None,
    ): ...

def compute_block_sparsity(
    tile_m,
    tile_n,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod: Callable,
    aux_tensors: list | None,
    device,
    compute_full_blocks: bool = True,
    use_fast_sampling: bool = False,
) -> tuple[BlockSparseTensors, BlockSparseTensorsTorch]: ...
