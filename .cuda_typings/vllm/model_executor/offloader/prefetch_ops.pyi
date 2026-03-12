from vllm.model_executor.offloader.base import get_offloader as get_offloader
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

def register_prefetch_offloader_ops() -> None: ...
