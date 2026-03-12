from dataclasses import dataclass
from enum import Enum
from vllm.model_executor.layers.fused_moe.fused_moe import (
    try_get_optimal_moe_config as try_get_optimal_moe_config,
)
from vllm.utils.math_utils import next_power_of_2 as next_power_of_2

class LoRAMappingType(Enum):
    LANGUAGE = 1
    TOWER = 2
    CONNECTOR = 3

@dataclass
class LoRAMapping:
    index_mapping: tuple[int, ...]
    prompt_mapping: tuple[int, ...]
    is_prefill: bool = ...
    type: LoRAMappingType = ...
    def __post_init__(self) -> None: ...

def try_get_optimal_moe_lora_config(
    op_type: str,
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    rank: int,
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, int | None]: ...
