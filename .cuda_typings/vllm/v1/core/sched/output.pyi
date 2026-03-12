import numpy as np
import numpy.typing as npt
import torch
from dataclasses import dataclass
from functools import cached_property as cached_property
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata as ECConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata as KVConnectorMetadata,
)
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec as MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.v1.request import Request as Request

@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: torch.Tensor | None = ...
    prefill_token_ids: list[int] | None = ...
    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> NewRequestData: ...
    def anon_repr(self) -> str: ...

@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    new_token_ids: list[list[int]]
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
    def anon_repr(self) -> str: ...
    @property
    def num_reqs(self) -> int: ...
    def is_context_phase(self, req_id: str) -> bool: ...
    @classmethod
    def make_empty(cls) -> CachedRequestData: ...

@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]
    preempted_req_ids: set[str] | None = ...
    has_structured_output_requests: bool = ...
    pending_structured_output_tokens: bool = ...
    num_invalid_spec_tokens: dict[str, int] | None = ...
    kv_connector_metadata: KVConnectorMetadata | None = ...
    ec_connector_metadata: ECConnectorMetadata | None = ...
    new_block_ids_to_zero: list[int] | None = ...
    @classmethod
    def make_empty(cls) -> SchedulerOutput: ...

@dataclass
class GrammarOutput:
    structured_output_request_ids: list[str]
    grammar_bitmask: npt.NDArray[np.int32]
