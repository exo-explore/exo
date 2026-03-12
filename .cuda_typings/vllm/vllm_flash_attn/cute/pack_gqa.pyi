import cutlass
import cutlass.cute as cute
from _typeshed import Incomplete

class PackGQA:
    m_block_size: Incomplete
    head_dim_padded: Incomplete
    check_hdim_oob: Incomplete
    qhead_per_kvhead: Incomplete
    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        head_dim_padded: cutlass.Constexpr[int],
        check_hdim_oob: cutlass.Constexpr[bool],
        qhead_per_kvhead: cutlass.Constexpr[bool],
    ) -> None: ...
    @cute.jit
    def compute_ptr(
        self,
        tensor: cute.Tensor,
        cRows: cute.Tensor,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        threads_per_row: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
    ): ...
    @cute.jit
    def load_Q(
        self,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ): ...
    @cute.jit
    def store_LSE(
        self,
        mLSE: cute.Tensor,
        tLSErLSE: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ): ...
    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,
        tOrO: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ): ...
