from _typeshed import Incomplete
from transformers import PretrainedConfig

class MLPSpeculatorConfig(PretrainedConfig):
    model_type: str
    attribute_map: Incomplete
    vocab_size: Incomplete
    emb_dim: Incomplete
    inner_dim: Incomplete
    n_predict: Incomplete
    top_k_tokens_per_head: Incomplete
    n_candidates: Incomplete
    num_lookahead_tokens: Incomplete
    tie_weights: Incomplete
    scale_input: Incomplete
    def __init__(
        self,
        vocab_size: int = 32000,
        emb_dim: int = 4096,
        inner_dim: int = 0,
        n_predict: int = 3,
        top_k_tokens_per_head: list[int] | None = None,
        n_candidates: int = 5,
        tie_weights: bool = False,
        scale_input: bool = False,
        **kwargs,
    ) -> None: ...
