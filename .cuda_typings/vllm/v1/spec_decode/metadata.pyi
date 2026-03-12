import torch
from dataclasses import dataclass

@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor
    num_draft_tokens: list[int]
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor
    max_spec_len = ...
    def __post_init__(self) -> None: ...
    @classmethod
    def make_dummy(
        cls, draft_token_ids: list[list[int]], device: torch.device
    ) -> SpecDecodeMetadata: ...
