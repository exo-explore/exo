import cutlass
import cutlass.cute as cute
from cutlass import Int32 as Int32
from dataclasses import dataclass
from typing import Callable
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK

@cute.jit
def mask_r2p(
    X: cute.Tensor, col_limit: Int32, arch: int = 90, rank1: bool = False
) -> None: ...
@cute.jit
def mask_r2p_transposed(X: cute.Tensor, row_limit_top: Int32, num_rep: int) -> None: ...
@cute.jit
def mask_r2p_dual_bound(
    X: cute.Tensor, col_limit_left: Int32, col_limit_right: Int32
) -> None: ...
@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Int32 | None = ...
    window_size_right: Int32 | None = ...
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = ...
    swap_AB: cutlass.Constexpr[bool] = ...
    @property
    def seqlen_q(self) -> Int32: ...
    @property
    def seqlen_k(self) -> Int32: ...
    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
    ) -> None: ...
    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None: ...
    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        is_full_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None: ...
