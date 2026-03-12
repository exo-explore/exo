import functools
import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8 as per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8 as per_token_group_quant_int8,
    per_token_quant_int8 as per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4 as quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    quant_dequant_mxfp6 as quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize as mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    per_tensor_dequantize as per_tensor_dequantize,
)
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.torch_utils import is_torch_equal_or_newer as is_torch_equal_or_newer

def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor: ...
def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
    is_fp4_scale_swizzled: bool = True,
    ocp_mx_scheme: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...
def normalize_scales_shape(scales: torch.Tensor | None) -> torch.Tensor | None: ...
def normalize_batched_scales_shape(
    scales: torch.Tensor | None, num_experts: int
) -> torch.Tensor | None: ...
@functools.cache
def disable_inplace() -> bool: ...
