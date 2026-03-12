import torch
from .interfaces import (
    SupportsCrossEncoding as SupportsCrossEncoding,
    SupportsQuant as SupportsQuant,
)
from .interfaces_base import (
    attn_type as attn_type,
    default_pooling_type as default_pooling_type,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Set as Set
from torch import nn
from transformers import BertConfig as BertConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    PoolerConfig as PoolerConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    EncoderOnlyAttention as EncoderOnlyAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.pooler import (
    DispatchPooler as DispatchPooler,
    Pooler as Pooler,
    PoolingParamsUpdate as PoolingParamsUpdate,
)
from vllm.model_executor.layers.pooler.activations import (
    LambdaPoolerActivation as LambdaPoolerActivation,
)
from vllm.model_executor.layers.pooler.seqwise import (
    EmbeddingPoolerHead as EmbeddingPoolerHead,
    SequencePooler as SequencePooler,
    SequencePoolerOutput as SequencePoolerOutput,
    get_seq_pooling_method as get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.tokwise import (
    pooler_for_token_classify as pooler_for_token_classify,
    pooler_for_token_embed as pooler_for_token_embed,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tasks import PoolingTask as PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

class BertEmbedding(nn.Module):
    size: Incomplete
    word_embeddings: Incomplete
    position_embeddings: Incomplete
    token_type_embeddings: Incomplete
    LayerNorm: Incomplete
    position_embedding_type: Incomplete
    def __init__(self, config: BertConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class BertPooler(SequencePooler):
    dense: Incomplete
    act_fn: Incomplete
    head: Incomplete
    def __init__(self, model_config: ModelConfig) -> None: ...

class BertEncoder(nn.Module):
    layer: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BertLayer(nn.Module):
    attention: Incomplete
    intermediate: Incomplete
    output: Incomplete
    def __init__(
        self,
        config: BertConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class BertAttention(nn.Module):
    self: Incomplete
    output: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BertSelfAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    head_dim: Incomplete
    num_kv_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    attn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BertSelfOutput(nn.Module):
    dense: Incomplete
    LayerNorm: Incomplete
    def __init__(
        self,
        hidden_size: int,
        layer_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class BertIntermediate(nn.Module):
    dense: Incomplete
    intermediate_act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BertOutput(nn.Module):
    dense: Incomplete
    LayerNorm: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class BertModel(nn.Module, SupportsQuant):
    is_pooling_model: bool
    packed_modules_mapping: Incomplete
    config: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        embedding_class: type[nn.Module] = ...,
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class BertPoolingModel(BertModel):
    is_pooling_model: bool
    pooler: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        embedding_class: type[nn.Module] = ...,
    ) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class BertEmbeddingModel(nn.Module, SupportsQuant):
    is_pooling_model: bool
    model: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

TOKEN_TYPE_SHIFT: int

class BertMLMHead(nn.Module):
    dense: Incomplete
    activation: Incomplete
    layer_norm: Incomplete
    decoder: Incomplete
    def __init__(
        self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12
    ) -> None: ...
    def tie_weights_with_embeddings(self, embeddings_weight: torch.Tensor): ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class SPLADESparsePooler(Pooler):
    mlm_head: Incomplete
    cls_token_id: Incomplete
    sep_token_id: Incomplete
    pooling: Incomplete
    remove_cls_sep: Incomplete
    def __init__(
        self,
        mlm_head: nn.Module,
        cls_token_id: int | None = 101,
        sep_token_id: int | None = 102,
        pooling: str = "max",
        remove_cls_sep: bool = True,
    ) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> SequencePoolerOutput: ...

class BertSpladeSparseEmbeddingModel(BertEmbeddingModel):
    mlm_head: Incomplete
    pooler: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", splade_pooling: str = "max"
    ) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

class BertForSequenceClassification(nn.Module, SupportsCrossEncoding, SupportsQuant):
    is_pooling_model: bool
    num_labels: Incomplete
    bert: Incomplete
    classifier: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class BertForTokenClassification(nn.Module):
    is_pooling_model: bool
    head_dtype: Incomplete
    num_labels: Incomplete
    bert: Incomplete
    classifier: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
