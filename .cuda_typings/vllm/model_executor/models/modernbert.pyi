import torch
from .interfaces import SupportsCrossEncoding as SupportsCrossEncoding
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
from collections.abc import Iterable
from torch import nn
from transformers import ModernBertConfig as ModernBertConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import (
    EncoderOnlyAttention as EncoderOnlyAttention,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler as DispatchPooler
from vllm.model_executor.layers.pooler.activations import (
    LambdaPoolerActivation as LambdaPoolerActivation,
)
from vllm.model_executor.layers.pooler.seqwise import (
    EmbeddingPoolerHead as EmbeddingPoolerHead,
    SequencePooler as SequencePooler,
    get_seq_pooling_method as get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.tokwise import (
    pooler_for_token_classify as pooler_for_token_classify,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class ModernBertEmbeddings(nn.Module):
    config: Incomplete
    tok_embeddings: Incomplete
    norm: Incomplete
    def __init__(self, config: ModernBertConfig) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class ModernBertAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    layer_id: Incomplete
    deterministic_flash_attn: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    all_head_size: Incomplete
    scaling: Incomplete
    Wqkv: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    Wo: Incomplete
    def __init__(
        self, config: ModernBertConfig, layer_id: int | None = None, prefix: str = ""
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor: ...

class ModernBertMLP(nn.Module):
    config: Incomplete
    Wi: Incomplete
    act: Incomplete
    Wo: Incomplete
    def __init__(self, config: ModernBertConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class ModernBertLayer(nn.Module):
    config: Incomplete
    attn_norm: Incomplete
    attn: Incomplete
    mlp_norm: Incomplete
    mlp: Incomplete
    def __init__(
        self, config: ModernBertConfig, prefix: str = "", layer_id: int | None = None
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor: ...

class ModernBertEncoderLayer(nn.Module):
    layers: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor: ...

class ModernBertModel(nn.Module):
    hf_to_vllm_mapper: Incomplete
    config: Incomplete
    embeddings: Incomplete
    encoder_layer: Incomplete
    final_norm: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class ModernBertPooler(SequencePooler):
    dense: Incomplete
    act: Incomplete
    norm: Incomplete
    head: Incomplete
    def __init__(self, model_config: ModelConfig) -> None: ...

class ModernBertForSequenceClassification(nn.Module, SupportsCrossEncoding):
    is_pooling_model: bool
    config: Incomplete
    model: Incomplete
    classifier: Incomplete
    pooling: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
    def forward(
        self,
        input_ids: torch.LongTensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class ModernBertPredictionHead(nn.Module):
    config: Incomplete
    dense: Incomplete
    act: Incomplete
    norm: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class ModernBertForTokenClassification(nn.Module):
    is_pooling_model: bool
    head_dtype: Incomplete
    num_labels: Incomplete
    model: Incomplete
    head: Incomplete
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
    ) -> torch.Tensor: ...
