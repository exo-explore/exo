import functools
import torch
from _typeshed import Incomplete
from functools import lru_cache
from vllm import envs as envs
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import next_power_of_2 as next_power_of_2

logger: Incomplete
is_batch_invariant: Incomplete

@functools.lru_cache
def load_lora_op_config(op_type: str, add_inputs: bool | None) -> dict | None: ...
@functools.lru_cache
def get_lora_op_configs(
    op_type: str,
    max_loras: int,
    batch: int,
    hidden_size: int,
    rank: int,
    num_slices: int,
    add_inputs: bool | None = None,
    moe_intermediate_size: int | None = None,
) -> dict[str, int | None]: ...
@lru_cache
def supports_pdl(device: torch.device | None = None) -> bool: ...
@lru_cache
def supports_tma(device: torch.device | None = None) -> bool: ...
