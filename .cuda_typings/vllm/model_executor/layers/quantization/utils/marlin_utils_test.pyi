import torch
from .marlin_utils import (
    GPTQ_MARLIN_TILE as GPTQ_MARLIN_TILE,
    marlin_permute_scales as marlin_permute_scales,
    marlin_zero_points as marlin_zero_points,
)
from .quant_utils import (
    get_pack_factor as get_pack_factor,
    gptq_quantize_weights as gptq_quantize_weights,
    quantize_weights as quantize_weights,
    sort_weights as sort_weights,
)
from _typeshed import Incomplete
from vllm.scalar_type import ScalarType as ScalarType, scalar_types as scalar_types

class MarlinWorkspace:
    scratch: Incomplete
    def __init__(self, out_features, min_thread_n, max_parallel) -> None: ...

def marlin_permute_weights(
    q_w, size_k, size_n, perm, tile=..., is_a_8bit: bool = False
): ...
def marlin_weights(q_w, size_k, size_n, num_bits, perm, is_a_8bit: bool = False): ...
def get_weight_perm(num_bits: int, is_a_8bit: bool = False): ...
def marlin_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
): ...
def awq_marlin_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    input_dtype: torch.dtype | None = None,
): ...
