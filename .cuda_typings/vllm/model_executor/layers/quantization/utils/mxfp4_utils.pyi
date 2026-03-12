from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import triton as triton
from vllm.utils.import_utils import has_triton_kernels as has_triton_kernels
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
    is_torch_equal_or_newer as is_torch_equal_or_newer,
)

logger: Incomplete
CK_MXFP4_MOE_DIM_ALIGNMENT: int

def get_padding_alignment(): ...

dequant_mxfp4: Incomplete
quant_dequant_mxfp4: Incomplete
