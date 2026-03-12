import torch
from vllm.model_executor.layers.utils import apply_penalties as apply_penalties
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad as make_tensor_with_pad

def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor: ...
