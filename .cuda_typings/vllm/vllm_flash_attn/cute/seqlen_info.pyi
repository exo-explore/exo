import cutlass
import cutlass.cute as cute
from cutlass import Int32 as Int32
from dataclasses import dataclass

@dataclass(frozen=True)
class SeqlenInfo:
    offset: cutlass.Int32
    seqlen: cutlass.Int32
    @staticmethod
    def create(
        batch_idx: cutlass.Int32,
        seqlen_static: cutlass.Int32,
        cu_seqlens: cute.Tensor | None = None,
        seqused: cute.Tensor | None = None,
    ): ...

@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: cutlass.Int32
    offset_k: cutlass.Int32
    padded_offset_q: cutlass.Int32
    padded_offset_k: cutlass.Int32
    seqlen_q: cutlass.Int32
    seqlen_k: cutlass.Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]
    has_seqused_q: cutlass.Constexpr[bool]
    has_seqused_k: cutlass.Constexpr[bool]
    @staticmethod
    def create(
        batch_idx: cutlass.Int32,
        seqlen_q_static: cutlass.Int32,
        seqlen_k_static: cutlass.Int32,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        tile_m: cutlass.Constexpr[cutlass.Int32] = 128,
        tile_n: cutlass.Constexpr[cutlass.Int32] = 128,
    ): ...
    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor: ...
    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor: ...
