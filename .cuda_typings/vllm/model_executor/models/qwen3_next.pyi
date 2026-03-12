import abc
import torch
from .interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    MixtureOfExperts as MixtureOfExperts,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    SpeculativeConfig as SpeculativeConfig,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    divide as divide,
    get_ep_group as get_ep_group,
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating_delta_rule_update as fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd as l2norm_fwd
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNormGated as RMSNormGated
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
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    mamba_v2_sharded_weight_loader as mamba_v2_sharded_weight_loader,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
    MambaStateCopyFuncCalculator as MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as causal_conv1d_fn,
    causal_conv1d_update as causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import (
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
    sharded_weight_loader as sharded_weight_loader,
)
from vllm.model_executor.models.utils import (
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig as Qwen3NextConfig
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata as GDNAttentionMetadata,
)

logger: Incomplete
KVCache: Incomplete

def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
): ...

class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None: ...
    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ): ...
    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ): ...

class Qwen3NextSparseMoeBlock(nn.Module):
    tp_size: Incomplete
    ep_group: Incomplete
    ep_rank: Incomplete
    ep_size: Incomplete
    n_routed_experts: Incomplete
    is_sequence_parallel: Incomplete
    enable_eplb: Incomplete
    n_logical_experts: Incomplete
    n_redundant_experts: Incomplete
    n_physical_experts: Incomplete
    n_local_physical_experts: Incomplete
    physical_expert_start: Incomplete
    physical_expert_end: Incomplete
    gate: Incomplete
    shared_expert_gate: Incomplete
    shared_expert: Incomplete
    experts: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str: ...
    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]: ...
    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size: Incomplete
    num_v_heads: Incomplete
    num_k_heads: Incomplete
    head_k_dim: Incomplete
    head_v_dim: Incomplete
    key_dim: Incomplete
    value_dim: Incomplete
    conv_kernel_size: Incomplete
    layer_idx: Incomplete
    activation: Incomplete
    act: Incomplete
    layer_norm_epsilon: Incomplete
    prefix: Incomplete
    config: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    speculative_config: Incomplete
    num_spec: Incomplete
    conv_dim: Incomplete
    conv1d: Incomplete
    in_proj_qkvz: Incomplete
    in_proj_ba: Incomplete
    dt_bias: Incomplete
    A_log: Incomplete
    norm: Incomplete
    out_proj: Incomplete
    chunk_gated_delta_rule: Incomplete
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear: ...
    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear: ...
    def fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ): ...
    def rearrange_mixed_qkv(self, mixed_qkv): ...
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor): ...

class Qwen3NextAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    dual_chunk_attention_config: Incomplete
    attn_output_gate: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, output: torch.Tensor, hidden_states: torch.Tensor
    ): ...

class Qwen3NextDecoderLayer(nn.Module):
    layer_type: Incomplete
    layer_idx: Incomplete
    linear_attn: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    layer_scale: Incomplete
    attn_layer_scale: Incomplete
    ffn_layer_scale: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, layer_type: str, prefix: str = ""
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ): ...

class Qwen3NextModel(nn.Module):
    num_redundant_experts: Incomplete
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    make_empty_intermediate_tensors: Incomplete
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

class QwenNextMixtureOfExperts(MixtureOfExperts):
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_redundant_experts: Incomplete
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    expert_weights: Incomplete
    moe_layers: Incomplete
    num_moe_layers: Incomplete
    num_expert_groups: int
    num_shared_experts: int
    num_logical_experts: Incomplete
    num_routed_experts: Incomplete
    def set_moe_parameters(self) -> None: ...

class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    vllm_config: Incomplete
    model_config: Incomplete
    quant_config: Incomplete
    config: Incomplete
    scheduler_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ): ...
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]: ...
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]: ...
    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...

def gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None: ...
def gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None: ...
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
): ...
def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
