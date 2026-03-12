import cutlass
import cutlass.cute as cute
import cutlass.pipeline
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import dsl_user_op
from typing import Callable

@dsl_user_op
def cvt_copy(
    atom: cute.CopyAtom,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    loc=None,
    ip=None,
    **kwargs,
) -> None: ...
@dsl_user_op
def load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor: ...
@dsl_user_op
def get_copy_atom(
    dtype: type[cutlass.Numeric],
    num_copy_elems: int,
    is_async: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom: ...
@dsl_user_op
def make_tmem_copy(
    tmem_copy_atom: cute.CopyAtom, num_wg: int = 1, *, loc=None, ip=None
) -> cute.CopyAtom: ...
@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None: ...
def tiled_copy_1d(
    dtype: type[cutlass.Numeric],
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy: ...
def tiled_copy_2d(
    dtype: type[cutlass.Numeric],
    major_mode_size: int,
    num_threads: int,
    is_async: bool = False,
) -> cute.TiledCopy: ...
@dsl_user_op
def atomic_add_fp32x4(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    gmem_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None: ...
@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32: ...
@dsl_user_op
def store_shared_remote_fp32x4(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None: ...
@dsl_user_op
def cpasync_bulk_g2s(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    tma_bar_ptr: cute.Pointer,
    size: int | Int32,
    *,
    loc=None,
    ip=None,
): ...
@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
): ...
def cpasync_bulk_get_copy_fn(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool = False,
    **kwargs,
) -> Callable: ...
def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    **kwargs,
) -> Callable: ...
def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync): ...
