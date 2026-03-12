import torch
from vllm.pooling_params import (
    LateInteractionParams as LateInteractionParams,
    PoolingParams as PoolingParams,
)

LATE_INTERACTION_MODE_CACHE_QUERY: str
LATE_INTERACTION_MODE_SCORE_DOC: str

def get_late_interaction_engine_index(
    pooling_params: PoolingParams | None, num_engines: int
) -> int | None: ...
def build_late_interaction_query_params(
    query_key: str, query_uses: int
) -> LateInteractionParams: ...
def build_late_interaction_doc_params(query_key: str) -> LateInteractionParams: ...
def compute_maxsim_score(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor: ...
