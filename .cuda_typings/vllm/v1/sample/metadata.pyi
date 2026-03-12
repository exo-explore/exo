import torch
from dataclasses import dataclass
from vllm.v1.sample.logits_processor import LogitsProcessors as LogitsProcessors

@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    generators: dict[int, torch.Generator]
    max_num_logprobs: int | None
    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    output_token_ids: list[list[int]]
    allowed_token_ids_mask: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    logitsprocs: LogitsProcessors
    spec_token_ids: list[list[int]] | None = ...
