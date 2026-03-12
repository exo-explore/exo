import torch
from .bert import BertEmbeddingModel as BertEmbeddingModel, BertModel as BertModel
from .bert_with_rope import JinaRobertaModel as JinaRobertaModel
from .interfaces import SupportsLateInteraction as SupportsLateInteraction
from .interfaces_base import default_pooling_type as default_pooling_type
from .modernbert import ModernBertModel as ModernBertModel
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.config import PoolerConfig as PoolerConfig, VllmConfig as VllmConfig
from vllm.model_executor.layers.pooler import Pooler as Pooler
from vllm.model_executor.layers.pooler.tokwise import (
    pooler_for_token_embed as pooler_for_token_embed,
)

class ColBERTMixin(nn.Module, SupportsLateInteraction):
    colbert_dim: int | None
    colbert_linear: nn.Linear | None
    hidden_size: int
    head_dtype: torch.dtype
    @classmethod
    def get_colbert_dim_from_config(cls, hf_config) -> int | None: ...

class ColBERTModel(ColBERTMixin, BertEmbeddingModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

class ColBERTModernBertModel(ColBERTMixin, nn.Module):
    is_pooling_model: bool
    model: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

class ColBERTJinaRobertaModel(ColBERTMixin, nn.Module):
    is_pooling_model: bool
    model: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
