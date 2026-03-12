from collections.abc import Callable as Callable
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group as get_kv_transfer_group,
    has_kv_transfer_group as has_kv_transfer_group,
    is_v1_kv_transfer_group as is_v1_kv_transfer_group,
)

def maybe_transfer_kv_layer(func: Callable) -> Callable: ...
