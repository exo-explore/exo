import torch
from vllm.lora.layers import LoRAMapping as LoRAMapping

def compute_meta(
    token_lora_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, bool]: ...
def convert_mapping(
    mapping: LoRAMapping,
    lora_index_to_id: list[int | None],
    max_loras: int,
    vocab_size: int,
    extra_vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]: ...
