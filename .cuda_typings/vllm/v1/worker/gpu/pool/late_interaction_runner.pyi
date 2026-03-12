from collections.abc import Iterable
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.v1.outputs import PoolerOutput as PoolerOutput
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY as LATE_INTERACTION_MODE_CACHE_QUERY,
    LATE_INTERACTION_MODE_SCORE_DOC as LATE_INTERACTION_MODE_SCORE_DOC,
    compute_maxsim_score as compute_maxsim_score,
)

class LateInteractionRunner:
    def __init__(self) -> None: ...
    def clear(self) -> None: ...
    def register_request(
        self, req_id: str, pooling_params: PoolingParams | None
    ) -> None: ...
    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None: ...
    def postprocess_pooler_output(
        self,
        raw_pooler_output: PoolerOutput,
        pooling_params: list[PoolingParams],
        req_ids: list[str],
        finished_mask: list[bool],
    ) -> PoolerOutput: ...
