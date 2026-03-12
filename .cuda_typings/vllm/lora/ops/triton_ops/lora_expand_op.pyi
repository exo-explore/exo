from _typeshed import Incomplete
from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel as do_expand_kernel
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs as get_lora_op_configs
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

lora_expand: Incomplete
