import torch
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE as OCP_MX_BLOCK_SIZE,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

def quant_dequant_mxfp6(
    x: torch.Tensor, quant_dtype: str, scale_calculation_mode: str = "even"
) -> torch.Tensor: ...
def dequant_mxfp6(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor: ...
