import torch
import torch.nn as nn
from .interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers.configuration_utils import PretrainedConfig as PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context as get_forward_context
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fla.ops.layernorm_guard import (
    RMSNormGated as RMSNormGated,
    layernorm_fn as layernorm_fn,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE as FusedMoE,
    SharedFusedMoE as SharedFusedMoE,
)
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
from vllm.model_executor.layers.mamba.abstract import MambaBase as MambaBase
from vllm.model_executor.layers.mamba.linear_attn import (
    MiniMaxText01LinearAttention as MiniMaxText01LinearAttention,
    MiniMaxText01LinearKernel as MiniMaxText01LinearKernel,
    MiniMaxText01RMSNormTP as MiniMaxText01RMSNormTP,
    clear_linear_attention_cache_for_new_sequences as clear_linear_attention_cache_for_new_sequences,
    linear_attention_decode as linear_attention_decode,
    linear_attention_prefill_and_mix as linear_attention_prefill_and_mix,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator as MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mla import (
    MLAModules as MLAModules,
    MultiHeadLatentAttentionWrapper as MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.bailing_moe import BailingMLP as BailingMLP
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.linear_attn import (
    LinearAttentionMetadata as LinearAttentionMetadata,
)

logger: Incomplete

def is_linear_layer(layer_idx, layer_group_size): ...

class BailingMoeV25MLAAttention(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    layer_id: Incomplete
    prefix: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    o_proj: Incomplete
    fused_qkv_a_proj: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    q_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    rotary_emb: Incomplete
    mla_attn: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "attention",
        cache_config: CacheConfig | None = None,
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor: ...

class BailingMoEGate(nn.Module):
    params_dtype: Incomplete
    weight: Incomplete
    expert_bias: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states): ...

class BailingMoeV25(nn.Module):
    layer_id: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    num_experts: Incomplete
    top_k: Incomplete
    norm_expert_prob: Incomplete
    hidden_size: Incomplete
    quant_config: Incomplete
    num_shared_experts: Incomplete
    score_function: Incomplete
    n_group: Incomplete
    topk_group: Incomplete
    use_grouped_topk: Incomplete
    routed_scaling_factor: Incomplete
    router_dtype: Incomplete
    gate: Incomplete
    shared_experts: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

BailingRMSNormTP = MiniMaxText01RMSNormTP

class BailingGroupRMSNormGate(RMSNormGated):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-05,
        group_size=None,
        norm_before_gate: bool = True,
        device=None,
        dtype=None,
    ) -> None: ...

class BailingMoELinearAttention(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str: ...
    def get_state_shape(self) -> tuple[tuple[int, ...], ...]: ...
    def get_state_dtype(self) -> tuple[torch.dtype, ...]: ...
    layer_id: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    total_kv_heads: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    prefix: Incomplete
    head_dim: Incomplete
    hidden_inner_size: Incomplete
    scaling: Incomplete
    tp_heads: Incomplete
    max_position_embeddings: Incomplete
    rope_theta: Incomplete
    tp_kv_heads: Incomplete
    q_size_per_rank: Incomplete
    kv_size_per_rank: Incomplete
    use_qk_norm: Incomplete
    linear_backend: str
    linear_scale: Incomplete
    linear_rope: Incomplete
    linear_silu: Incomplete
    BLOCK: Incomplete
    query_key_value: Incomplete
    query_layernorm: Incomplete
    key_layernorm: Incomplete
    g_proj: Incomplete
    dense: Incomplete
    group_norm_size: Incomplete
    rms_norm_eps: Incomplete
    g_norm: Incomplete
    rotary_emb: Incomplete
    slope_rate: Incomplete
    tp_slope: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "linear_attn",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
    ) -> None: ...
    @staticmethod
    def weight_direct_load(
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor, positions: torch.Tensor
    ) -> None: ...

class BailingMoeV25DecoderLayer(nn.Module):
    layer_id: Incomplete
    hidden_size: Incomplete
    attention_type: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "layer",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class BailingMoeV25Model(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_dim: Incomplete
    layer_group_size: Incomplete
    num_layers: Incomplete
    decoder_attention_types: Incomplete
    word_embeddings: Incomplete
    norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class BailingMoeV25ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]: ...
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]: ...
    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
