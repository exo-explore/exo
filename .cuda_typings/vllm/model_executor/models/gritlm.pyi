import abc
import torch
from .interfaces_base import default_pooling_type as default_pooling_type
from _typeshed import Incomplete
from collections.abc import Set as Set
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.pooler import (
    DispatchPooler as DispatchPooler,
    PoolingParamsUpdate as PoolingParamsUpdate,
)
from vllm.model_executor.layers.pooler.activations import (
    PoolerNormalize as PoolerNormalize,
)
from vllm.model_executor.layers.pooler.seqwise import (
    EmbeddingPoolerHead as EmbeddingPoolerHead,
    SequencePooler as SequencePooler,
    SequencePoolingMethod as SequencePoolingMethod,
    SequencePoolingMethodOutput as SequencePoolingMethodOutput,
    get_seq_pooling_method as get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.tokwise import (
    pooler_for_token_embed as pooler_for_token_embed,
)
from vllm.model_executor.models.llama import LlamaForCausalLM as LlamaForCausalLM
from vllm.tasks import PoolingTask as PoolingTask
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

logger: Incomplete

class GritLMMeanPool(SequencePoolingMethod):
    model_config: Incomplete
    token_ids: Incomplete
    user_pattern_ids: Incomplete
    embed_newline_pattern_ids: Incomplete
    embed_pattern_ids: Incomplete
    def __init__(self, model_config: ModelConfig) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> SequencePoolingMethodOutput: ...

class GritLMPooler(SequencePooler):
    def __init__(self, model_config: ModelConfig) -> None: ...

class GritLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    is_pooling_model: bool
    pooler: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None: ...
