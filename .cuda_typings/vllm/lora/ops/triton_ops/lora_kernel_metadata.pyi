import torch
from dataclasses import dataclass, field

@dataclass
class LoRAKernelMeta:
    token_lora_mapping: torch.Tensor
    token_indices_sorted_by_lora_ids: torch.Tensor
    active_lora_ids: torch.Tensor
    num_tokens_per_lora: torch.Tensor
    lora_token_start_loc: torch.Tensor
    no_lora_flag_cpu: torch.Tensor
    num_active_loras_cpu: torch.Tensor
    default_num_active_loras_cpu: torch.Tensor
    captured_lora_counts: list[int] = field(default_factory=list)
    @staticmethod
    def make(
        max_loras: int,
        max_num_tokens: int,
        device: torch.device | str,
        captured_lora_counts: list[int] | None = None,
    ) -> LoRAKernelMeta: ...
    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None: ...
    def meta_args(
        self, token_nums: int, specialize_active_lora: bool
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: ...
