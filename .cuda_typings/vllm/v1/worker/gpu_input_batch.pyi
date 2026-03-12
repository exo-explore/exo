import numpy as np
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec as MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.sampling_params import (
    SamplingParams as SamplingParams,
    SamplingType as SamplingType,
)
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
)
from vllm.utils.collection_utils import swap_dict_values as swap_dict_values
from vllm.v1.outputs import LogprobsTensors as LogprobsTensors
from vllm.v1.pool.metadata import (
    PoolingMetadata as PoolingMetadata,
    PoolingStates as PoolingStates,
)
from vllm.v1.sample.logits_processor import (
    BatchUpdateBuilder as BatchUpdateBuilder,
    LogitsProcessors as LogitsProcessors,
    MoveDirectionality as MoveDirectionality,
)
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadata
from vllm.v1.utils import copy_slice as copy_slice
from vllm.v1.worker.block_table import MultiGroupBlockTable as MultiGroupBlockTable

@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
    mrope_positions: torch.Tensor | None = ...
    mrope_position_delta: int | None = ...
    xdrope_positions: torch.Tensor | None = ...
    lora_request: LoRARequest | None = ...
    prompt_embeds: torch.Tensor | None = ...
    prev_num_draft_len: int = ...
    pooling_params: PoolingParams | None = ...
    pooling_states: PoolingStates | None = ...
    num_prompt_tokens = ...
    def __post_init__(self) -> None: ...
    @property
    def num_tokens(self) -> int: ...
    def get_token_id(self, idx: int) -> int: ...

class InputBatch:
    is_pooling_model: Incomplete
    is_spec_decode: Incomplete
    max_num_reqs: Incomplete
    max_model_len: Incomplete
    max_num_batched_tokens: Incomplete
    device: Incomplete
    pin_memory: Incomplete
    vocab_size: Incomplete
    req_id_to_index: dict[str, int]
    token_ids_cpu_tensor: Incomplete
    token_ids_cpu: Incomplete
    is_token_ids_tensor: Incomplete
    is_token_ids: Incomplete
    req_prompt_embeds: dict[int, torch.Tensor]
    num_tokens_no_spec_cpu_tensor: Incomplete
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
    num_accepted_tokens_cpu_tensor: Incomplete
    num_accepted_tokens_cpu: Incomplete
    request_lora_mapping: Incomplete
    lora_id_to_request_ids: dict[int, set[str]]
    lora_id_to_lora_request: dict[int, LoRARequest]
    generators: dict[int, torch.Generator]
    num_logprobs: dict[str, int]
    in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors]
    batch_update_builder: Incomplete
    has_allowed_token_ids: set[str]
    allowed_token_ids_mask: torch.Tensor | None
    allowed_token_ids_mask_cpu_tensor: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    logits_processing_needs_token_ids: Incomplete
    req_output_token_ids: list[list[int] | None]
    logitsprocs: Incomplete
    logitsprocs_need_output_token_ids: Incomplete
    spec_token_ids: list[list[int]]
    sampling_metadata: Incomplete
    pooling_params: dict[str, PoolingParams]
    pooling_states: dict[str, PoolingStates]
    prev_sampled_token_ids: torch.Tensor | None
    prev_req_id_to_index: dict[str, int] | None
    sampled_token_ids_cpu: torch.Tensor | None
    async_copy_ready_event: torch.Event | None
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
        max_num_blocks_per_req: list[int] | None = None,
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None: ...
    @property
    def req_ids(self) -> list[str]: ...
    def add_request(self, request: CachedRequestState) -> int: ...
    def update_req_spec_token_ids(
        self, request: CachedRequestState, scheduled_spec_tokens: dict[str, list[int]]
    ) -> None: ...
    def remove_request(self, req_id: str) -> int | None: ...
    def swap_states(self, i1: int, i2: int) -> None: ...
    def condense(self) -> None: ...
    def refresh_metadata(self) -> None: ...
    def get_pooling_params(self) -> list[PoolingParams]: ...
    def get_pooling_states(self) -> list[PoolingStates]: ...
    def get_pooling_metadata(self) -> PoolingMetadata: ...
    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]: ...
    def set_async_sampled_token_ids(
        self, sampled_token_ids_cpu: torch.Tensor, async_copy_ready_event: torch.Event
    ) -> None: ...
    def update_async_output_token_ids(self) -> None: ...
    def update_async_spec_token_ids(self, draft_token_ids: list[list[int]]) -> None: ...
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
    def no_penalties(self) -> bool: ...
    @property
    def max_num_logprobs(self) -> int | None: ...
    @property
    def no_allowed_token_ids(self) -> bool: ...
