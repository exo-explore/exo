import abc
import numpy as np
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable
from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias, TypeVar
from vllm.compilation.cuda_graph import CUDAGraphStat as CUDAGraphStat
from vllm.distributed.kv_events import KVConnectorKVEvents as KVConnectorKVEvents
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats as KVConnectorStats,
)
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput

class LogprobsLists(NamedTuple):
    logprob_token_ids: np.ndarray
    logprobs: np.ndarray
    sampled_token_ranks: np.ndarray
    cu_num_generated_tokens: list[int] | None = ...
    def slice_request(self, req_idx: int, num_positions: int): ...

class LogprobsTensors(NamedTuple):
    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor
    cu_num_generated_tokens: list[int] | None = ...
    def tolists(self, cu_num_generated_tokens: list[int] | None = None): ...
    def to_cpu_nonblocking(self) -> LogprobsTensors: ...
    def filter(self, mask: torch.Tensor) -> LogprobsTensors: ...
    @staticmethod
    def empty_cpu(
        num_positions: int, num_tokens_per_position: int
    ) -> LogprobsTensors: ...

PoolerOutput: TypeAlias

@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None

T = TypeVar("T")

@dataclass
class KVConnectorOutput:
    finished_sending: set[str] | None = ...
    finished_recving: set[str] | None = ...
    kv_connector_stats: KVConnectorStats | None = ...
    kv_cache_events: KVConnectorKVEvents | None = ...
    invalid_block_ids: set[int] = field(default_factory=set)
    expected_finished_count: int = ...
    def is_empty(self): ...
    @classmethod
    def merge(cls, *outputs: KVConnectorOutput): ...

@dataclass
class ECConnectorOutput:
    finished_sending: set[str] | None = ...
    finished_recving: set[str] | None = ...

@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    logprobs: LogprobsLists | None = ...
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )
    pooler_output: list[torch.Tensor | None] | None = ...
    kv_connector_output: KVConnectorOutput | None = ...
    ec_connector_output: ECConnectorOutput | None = ...
    num_nans_in_logits: dict[str, int] | None = ...
    cudagraph_stats: CUDAGraphStat | None = ...

class AsyncModelRunnerOutput(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput: ...

@dataclass
class DraftTokenIds:
    req_ids: list[str]
    draft_token_ids: list[list[int]]

def make_empty_encoder_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput: ...

EMPTY_MODEL_RUNNER_OUTPUT: Incomplete
