import abc
import torch
from .interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
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
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.distributed.utils import (
    split_tensor_along_last_dim as split_tensor_along_last_dim,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.layernorm import (
    RMSNorm as RMSNorm,
    RMSNormGated as RMSNormGated,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase as MambaBase
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
    sharded_weight_loader as sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.triton_utils import tl as tl, triton as triton
from vllm.triton_utils.allocation import set_triton_allocator as set_triton_allocator
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata as GDNAttentionMetadata,
)

logger: Incomplete

class OlmoHybridGatedDeltaNet(nn.Module, MambaBase):
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
    allow_neg_eigval: Incomplete
    prefix: Incomplete
    config: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    speculative_config: Incomplete
    num_spec: Incomplete
    in_proj_qkvg: Incomplete
    b_proj: Incomplete
    a_proj: Incomplete
    conv_dim: Incomplete
    conv1d: Incomplete
    dt_bias: Incomplete
    A_log: Incomplete
    o_norm: Incomplete
    o_proj: Incomplete
    def __init__(
        self,
        config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def rearrange_mixed_qkv(self, mixed_qkv): ...
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor): ...

class OlmoHybridAttention(nn.Module):
    config: Incomplete
    tp_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    max_position_embeddings: Incomplete
    qkv_proj: Incomplete
    tp_rank: Incomplete
    k_norm: Incomplete
    q_norm: Incomplete
    scaling: Incomplete
    attn: Incomplete
    rotary_emb: Incomplete
    o_proj: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class OlmoHybridMLP(nn.Module):
    gate_up_proj: Incomplete
    act_fn: Incomplete
    down_proj: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class OlmoHybridDecoderLayer(nn.Module):
    layer_type: Incomplete
    layer_idx: Incomplete
    linear_attn: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    self_attn: Incomplete
    post_feedforward_layernorm: Incomplete
    mlp: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class OlmoHybridModel(nn.Module):
    config: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class OlmoHybridForCausalLM(
    nn.Module, HasInnerState, SupportsPP, SupportsLoRA, IsHybrid, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    config: Incomplete
    vllm_config: Incomplete
    model_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
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
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

def olmo_hybrid_gdn_full_forward(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
def olmo_hybrid_gdn_full_forward_fake(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
@triton.jit
def fused_olmo_hybrid_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    allow_neg_eigval: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
): ...
def fused_olmo_hybrid_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    allow_neg_eigval: bool = False,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
