from .utils import supports_pdl as supports_pdl, supports_tma as supports_tma
from _typeshed import Incomplete
from vllm.distributed import (
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl as tl, triton as triton
from vllm.triton_utils.allocation import set_triton_allocator as set_triton_allocator
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

fused_moe_lora: Incomplete
fused_moe_lora_shrink: Incomplete
fused_moe_lora_expand: Incomplete
