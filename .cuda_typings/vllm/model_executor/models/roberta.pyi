import torch
from .bert_with_rope import (
    BertWithRope as BertWithRope,
    JinaRobertaModel as JinaRobertaModel,
)
from .interfaces import SupportsCrossEncoding as SupportsCrossEncoding
from .interfaces_base import default_pooling_type as default_pooling_type
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import RobertaConfig as RobertaConfig
from vllm.config import (
    ModelConfig as ModelConfig,
    PoolerConfig as PoolerConfig,
    VllmConfig as VllmConfig,
)
from vllm.model_executor.layers.pooler import (
    BOSEOSFilter as BOSEOSFilter,
    DispatchPooler as DispatchPooler,
    Pooler as Pooler,
)
from vllm.model_executor.layers.pooler.seqwise import (
    pooler_for_embed as pooler_for_embed,
)
from vllm.model_executor.layers.pooler.tokwise import (
    AllPool as AllPool,
    pooler_for_token_classify as pooler_for_token_classify,
    pooler_for_token_embed as pooler_for_token_embed,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.default_loader import (
    DefaultModelLoader as DefaultModelLoader,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.bert import (
    BertEmbeddingModel as BertEmbeddingModel,
    BertModel as BertModel,
    TOKEN_TYPE_SHIFT as TOKEN_TYPE_SHIFT,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class RobertaEmbedding(nn.Module):
    size: Incomplete
    word_embeddings: Incomplete
    padding_idx: Incomplete
    position_embeddings: Incomplete
    token_type_embeddings: Incomplete
    LayerNorm: Incomplete
    def __init__(self, config: RobertaConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class RobertaClassificationHead(nn.Module):
    dense: Incomplete
    out_proj: Incomplete
    def __init__(self, model_config: ModelConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class RobertaEmbeddingModel(BertEmbeddingModel):
    padding_idx: int
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

def filter_secondary_weights(
    all_weights: Iterable[tuple[str, torch.Tensor]], secondary_weights: list[str]
) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str, torch.Tensor]]]: ...

class BgeM3EmbeddingModel(RobertaEmbeddingModel):
    hidden_size: Incomplete
    head_dtype: Incomplete
    bos_token_id: Incomplete
    eos_token_id: Incomplete
    secondary_weight_prefixes: Incomplete
    secondary_weight_files: Incomplete
    secondary_weights: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, all_weights: Iterable[tuple[str, torch.Tensor]]): ...

class RobertaForSequenceClassification(nn.Module, SupportsCrossEncoding):
    is_pooling_model: bool
    jina_to_vllm_mapper: Incomplete
    padding_idx: int
    num_labels: Incomplete
    roberta: Incomplete
    classifier: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

def replace_roberta_positions(
    input_ids: torch.Tensor, position_ids: torch.Tensor, padding_idx: int
) -> None: ...
