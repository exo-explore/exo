import torch
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import next_power_of_2 as next_power_of_2
from vllm.utils.platform_utils import num_compute_units as num_compute_units

def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = ...,
) -> torch.Tensor: ...
def reset_buffer_cache() -> None: ...
