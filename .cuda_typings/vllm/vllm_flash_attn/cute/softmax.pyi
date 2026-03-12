import cutlass
import cutlass.cute as cute
from cutlass import Float32
from dataclasses import dataclass
from vllm.vllm_flash_attn.cute.cute_dsl_utils import ParamsBase as ParamsBase
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK as SeqlenInfoQK

@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = ...
    softmax_scale: Float32 | None = ...
    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ): ...
    def reset(self) -> None: ...
    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor: ...
    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor: ...
    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None: ...

@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = ...
    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
    ): ...
    @cute.jit
    def update_row_max(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> tuple[Float32, Float32]: ...
    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None: ...
    @cute.jit
    def scale_subtract_rowmax(self, acc_S_row: cute.Tensor, row_max: Float32): ...
    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        e2e: cutlass.Constexpr[bool] = False,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ): ...
    @cute.jit
    def scale_apply_exp2_convert(
        self, acc_S_row: cute.Tensor, row_max: Float32, acc_S_row_converted: cute.Tensor
    ): ...

@cute.jit
def floor_if_packed(q_idx, qhead_per_kvhead: cutlass.Constexpr[int]) -> cute.Tensor: ...
@cute.jit
def apply_score_mod_inner(
    score_tensor,
    index_tensor,
    score_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info: SeqlenInfoQK,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
): ...
@cute.jit
def apply_score_mod_bwd_inner(
    grad_tensor,
    score_tensor,
    index_tensor,
    score_mod_bwd: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
): ...
