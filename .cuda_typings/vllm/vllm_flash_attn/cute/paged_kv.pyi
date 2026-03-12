import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute import FastDivmodDivisor as FastDivmodDivisor
from dataclasses import dataclass
from vllm.vllm_flash_attn.cute import utils as utils
from vllm.vllm_flash_attn.cute.cute_dsl_utils import ParamsBase as ParamsBase

@dataclass
class PagedKVManager(ParamsBase):
    mPageTable: cute.Tensor
    mK_paged: cute.Tensor
    mV_paged: cute.Tensor
    thread_idx: Int32
    page_size_divmod: FastDivmodDivisor
    seqlen_k: Int32
    leftpad_k: Int32
    n_block_size: Int32
    num_threads: cutlass.Constexpr[Int32]
    head_dim_padded: cutlass.Constexpr[Int32]
    head_dim_v_padded: cutlass.Constexpr[Int32]
    gmem_threads_per_row: cutlass.Constexpr[Int32]
    page_entry_per_thread: Int32
    async_copy_elems: Int32
    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy
    tPrPage: cute.Tensor
    tPrPageOffset: cute.Tensor
    tKpK: cute.Tensor
    tVpV: cute.Tensor
    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        mK_paged: cute.Tensor,
        mV_paged: cute.Tensor,
        page_size_divmod: FastDivmodDivisor,
        bidb: Int32,
        bidh: Int32,
        thread_idx: Int32,
        seqlen_k: Int32,
        leftpad_k: Int32,
        n_block_size: cutlass.Constexpr[Int32],
        head_dim_padded: cutlass.Constexpr[Int32],
        head_dim_v_padded: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: type[cutlass.Numeric],
    ): ...
    @cute.jit
    def load_page_table(self, n_block: Int32): ...
    @cute.jit
    def compute_X_ptr(self, K_or_V: str): ...
    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, K_or_V: str): ...
