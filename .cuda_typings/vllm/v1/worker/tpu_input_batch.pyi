import numpy as np
import torch
from _typeshed import Incomplete
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.sampling_params import SamplingType as SamplingType
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
)
from vllm.utils.collection_utils import swap_dict_values as swap_dict_values
from vllm.v1.outputs import LogprobsTensors as LogprobsTensors
from vllm.v1.worker.block_table import MultiGroupBlockTable as MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState as CachedRequestState

class InputBatch:
    max_num_reqs: Incomplete
    max_model_len: Incomplete
    max_num_batched_tokens: Incomplete
    device: Incomplete
    pin_memory: Incomplete
    vocab_size: Incomplete
    req_id_to_index: dict[str, int]
    token_ids_cpu_tensor: Incomplete
    token_ids_cpu: Incomplete
    num_tokens_no_spec: Incomplete
    num_prompt_tokens: Incomplete
    num_computed_tokens_cpu_tensor: Incomplete
    num_computed_tokens_cpu: Incomplete
    block_table: Incomplete
    temperature: Incomplete
    temperature_cpu_tensor: Incomplete
    temperature_cpu: Incomplete
    greedy_reqs: set[str]
    random_reqs: set[str]
    top_p: Incomplete
    top_p_cpu_tensor: Incomplete
    top_p_cpu: Incomplete
    top_p_reqs: set[str]
    top_k: Incomplete
    top_k_cpu_tensor: Incomplete
    top_k_cpu: Incomplete
    top_k_reqs: set[str]
    min_p: Incomplete
    min_p_cpu_tensor: Incomplete
    min_p_cpu: Incomplete
    min_p_reqs: set[str]
    frequency_penalties: Incomplete
    frequency_penalties_cpu_tensor: Incomplete
    frequency_penalties_cpu: Incomplete
    frequency_penalties_reqs: set[str]
    presence_penalties: Incomplete
    presence_penalties_cpu_tensor: Incomplete
    presence_penalties_cpu: Incomplete
    presence_penalties_reqs: set[str]
    repetition_penalties: Incomplete
    repetition_penalties_cpu_tensor: Incomplete
    repetition_penalties_cpu: Incomplete
    repetition_penalties_reqs: set[str]
    min_tokens: dict[int, tuple[int, set[int]]]
    request_lora_mapping: Incomplete
    lora_id_to_request_ids: dict[int, set[str]]
    lora_id_to_lora_request: dict[int, LoRARequest]
    generators: dict[int, torch.Generator]
    num_logprobs: dict[str, int]
    in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors]
    logit_bias: list[dict[int, float] | None]
    has_allowed_token_ids: set[str]
    allowed_token_ids_mask: torch.Tensor | None
    allowed_token_ids_mask_cpu_tensor: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    req_output_token_ids: list[list[int] | None]
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
    ) -> None: ...
    @property
    def req_ids(self) -> list[str]: ...
    def add_request(
        self, request: CachedRequestState, req_index: int | None = None
    ) -> None: ...
    def remove_request(self, req_id: str) -> int | None: ...
    def swap_states(self, i1: int, i2: int) -> None: ...
    def condense(self, empty_req_indices: list[int]) -> None: ...
    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]: ...
    @property
    def num_reqs(self) -> int: ...
    @property
    def all_greedy(self) -> bool: ...
    @property
    def all_random(self) -> bool: ...
    @property
    def no_top_p(self) -> bool: ...
    @property
    def no_top_k(self) -> bool: ...
    @property
    def no_min_p(self) -> bool: ...
    @property
    def no_penalties(self) -> bool: ...
    @property
    def max_num_logprobs(self) -> int | None: ...
    @property
    def no_allowed_token_ids(self) -> bool: ...
