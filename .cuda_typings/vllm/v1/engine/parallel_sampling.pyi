from vllm.outputs import CompletionOutput as CompletionOutput
from vllm.sampling_params import (
    RequestOutputKind as RequestOutputKind,
    SamplingParams as SamplingParams,
)
from vllm.v1.engine import EngineCoreRequest as EngineCoreRequest
from vllm.v1.metrics.stats import IterationStats as IterationStats

class ParentRequest:
    request_id: str
    external_req_id: str
    sampling_params: SamplingParams
    child_requests: set[str]
    output_aggregator: list[CompletionOutput]
    max_num_generation_tokens: int
    cached_child_sampling_params: SamplingParams | None
    def __init__(self, request: EngineCoreRequest) -> None: ...
    def get_child_info(self, index: int) -> tuple[str, SamplingParams]: ...
    @property
    def n(self) -> int: ...
    def get_outputs(
        self, child_request_id: str, completion_output: CompletionOutput
    ) -> tuple[list[CompletionOutput], bool]: ...
    def observe_num_generation_tokens(self, num_generation_tokens: int): ...
    @staticmethod
    def observe_finished_request(
        parent_req: ParentRequest | None,
        iteration_stats: IterationStats,
        num_generation_tokens: int,
    ): ...
