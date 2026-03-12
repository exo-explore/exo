import torch
from _typeshed import Incomplete
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    get_fp8_min_max as get_fp8_min_max,
    group_broadcast as group_broadcast,
    prep_scale_for_group_broadcast as prep_scale_for_group_broadcast,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT as DeepGemmQuantScaleFMT,
    is_deep_gemm_e8m0_used as is_deep_gemm_e8m0_used,
    is_deep_gemm_supported as is_deep_gemm_supported,
)

class QuantFP8(CustomOp):
    static: Incomplete
    group_shape: Incomplete
    use_per_token_if_dynamic: Incomplete
    num_token_padding: Incomplete
    column_major_scales: Incomplete
    tma_aligned_scales: Incomplete
    use_ue8m0: Incomplete
    use_deep_gemm_supported: Incomplete
    use_aiter: Incomplete
    is_group_quant: Incomplete
    group_size: Incomplete
    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
        column_major_scales: bool = False,
        tma_aligned_scales: bool = False,
        use_ue8m0: bool | None = None,
        compile_native: bool = True,
    ) -> None: ...
    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ): ...
