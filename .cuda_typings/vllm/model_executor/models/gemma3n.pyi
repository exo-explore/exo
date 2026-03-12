import torch
from .interfaces import SupportsQuant as SupportsQuant
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context as get_forward_context
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import (
    GeluAndMul as GeluAndMul,
    GeluAndMulSparse as GeluAndMulSparse,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backends.utils import (
    KVSharingFastPrefillMetadata as KVSharingFastPrefillMetadata,
)

logger: Incomplete
EPS: Incomplete

class Gemma3nAltUp(nn.Module):
    altup_num_inputs: Incomplete
    altup_active_idx: Incomplete
    altup_coef_clip: Incomplete
    correction_coefs: Incomplete
    prediction_coefs: Incomplete
    modality_router: Incomplete
    router_norm: Incomplete
    router_input_scale: Incomplete
    correct_output_scale: Incomplete
    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float,
        altup_num_inputs: int,
        altup_coef_clip: float,
        altup_active_idx: int,
        quant_config: QuantizationConfig,
        prefix: str,
    ) -> None: ...
    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor: ...
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def correct(
        self, predictions: torch.Tensor, activated: torch.Tensor
    ) -> torch.Tensor: ...

class Gemma3nLaurelBlock(nn.Module):
    linear_left: Incomplete
    linear_right: Incomplete
    post_laurel_norm: Incomplete
    def __init__(
        self,
        hidden_size: int,
        laurel_rank: int,
        rms_norm_eps: float,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Gemma3nMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        activation_sparsity: float = 0.0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Gemma3nAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    v_norm: Incomplete
    sliding_window: Incomplete
    is_kv_shared: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: Gemma3nTextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...

class Gemma3nDecoderLayer(nn.Module):
    altup_active_idx: Incomplete
    altup: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    laurel: Incomplete
    per_layer_input_gate: Incomplete
    per_layer_projection: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    pre_feedforward_layernorm: Incomplete
    post_feedforward_layernorm: Incomplete
    post_per_layer_input_norm: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        config: Gemma3nTextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Gemma3nSelfDecoder(nn.Module):
    decoder_layers: Incomplete
    layer_idx_start: Incomplete
    config: Incomplete
    embed_tokens: Incomplete
    embed_scale: Incomplete
    embed_tokens_per_layer: Incomplete
    embed_scale_per_layer: Incomplete
    per_layer_model_projection: Incomplete
    per_layer_projection_norm: Incomplete
    per_layer_input_scale: Incomplete
    per_layer_projection_scale: Incomplete
    altup_projections: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layers: list[Gemma3nDecoderLayer],
        layer_idx_start: int,
    ) -> None: ...
    def get_per_layer_input_embeddings(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor: ...
    def get_per_layer_inputs(
        self, hidden_states_0: torch.Tensor, per_layer_inputs: torch.Tensor | None
    ) -> torch.Tensor: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def altup_embed(self, hidden_states_0: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Gemma3nCrossDecoder(nn.Module):
    decoder_layers: Incomplete
    layer_idx_start: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layers: list[Gemma3nDecoderLayer],
        layer_idx_start: int,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor: ...

class Gemma3nTextModel(nn.Module, SupportsQuant):
    config: Incomplete
    quant_config: Incomplete
    altup_unembed_projections: Incomplete
    self_decoder: Incomplete
    cross_decoder: Incomplete
    norm: Incomplete
    fast_prefill_enabled: Incomplete
    positions: Incomplete
    hidden_states: Incomplete
    per_layer_inputs: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @property
    def embed_tokens(self): ...
    def get_per_layer_input_embeddings(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def fast_prefill_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
    def normal_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
    def altup_unembed(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Gemma3nForCausalLM(nn.Module):
    packed_modules_mapping: Incomplete
    config: Incomplete
    cache_config: Incomplete
    model: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        per_layer_inputs: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
