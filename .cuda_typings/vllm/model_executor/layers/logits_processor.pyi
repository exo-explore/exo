import torch
from _typeshed import Incomplete
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
    tensor_model_parallel_gather as tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.platforms import current_platform as current_platform

class LogitsProcessor(CustomOp):
    scale: Incomplete
    vocab_size: Incomplete
    logits_as_input: Incomplete
    org_vocab_size: Incomplete
    soft_cap: Incomplete
    use_all_gather: Incomplete
    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None: ...
    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None: ...
    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...
