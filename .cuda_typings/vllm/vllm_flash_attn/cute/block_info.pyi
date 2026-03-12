import cutlass
import cutlass.cute as cute
from cutlass import Int32 as Int32
from dataclasses import dataclass
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK

@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = ...
    is_split_kv: cutlass.Constexpr[bool] = ...
    window_size_left: Int32 | None = ...
    window_size_right: Int32 | None = ...
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = ...
    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: cutlass.Int32 = 0,
        num_splits: cutlass.Int32 = 1,
    ) -> tuple[Int32, Int32]: ...
    @cute.jit
    def get_m_block_min_max(
        self, seqlen_info: SeqlenInfoQK, n_block: Int32
    ) -> tuple[Int32, Int32]: ...
    @cute.jit
    def get_n_block_min_causal_local_mask(
        self, seqlen_info: SeqlenInfoQK, m_block: Int32, n_block_min: Int32
    ) -> Int32: ...
    @cute.jit
    def get_n_block_min_before_local_mask(
        self, seqlen_info: SeqlenInfoQK, m_block: Int32, n_block_min: Int32
    ) -> Int32: ...
