import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_pp_group as get_pp_group
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.model_executor.custom_op import PluggableLayer as PluggableLayer
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
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
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update as selective_state_update,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen as mamba_chunk_scan_combined_varlen,
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
    composed_weight_loader as composed_weight_loader,
    default_weight_loader as default_weight_loader,
    sharded_weight_loader as sharded_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadata as Mamba2AttentionMetadata,
)

class Plamo2Config(PretrainedConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    num_attention_heads: int
    hidden_size_per_head: int
    num_key_value_heads: int
    mamba_d_state: int
    mamba_d_conv: int
    mamba_num_heads: int
    mamba_step: int
    intermediate_size: int
    vocab_size: int

def is_mamba(config: Plamo2Config, i: int) -> bool: ...

class Plamo2MambaMixer(MambaBase, PluggableLayer):
    config: Incomplete
    cache_config: Incomplete
    model_config: Incomplete
    quant_config: Incomplete
    is_lora_enabled: Incomplete
    hidden_size: Incomplete
    ssm_state_size: Incomplete
    conv_kernel_size: Incomplete
    intermediate_size: Incomplete
    tp_size: Incomplete
    head_dim: Incomplete
    num_heads: Incomplete
    time_step_rank: Incomplete
    conv1d: Incomplete
    in_proj: Incomplete
    bcdt_proj: Incomplete
    dt_proj: Incomplete
    A: Incomplete
    D: Incomplete
    dt_bias: Incomplete
    out_proj: Incomplete
    activation: str
    dt_norm: Incomplete
    B_norm: Incomplete
    C_norm: Incomplete
    chunk_size: Incomplete
    kv_cache: Incomplete
    prefix: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, *, prefix: str = "", **kwargs
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor, **kwargs): ...
    def forward_impl(
        self, hidden_states: torch.Tensor, output: torch.Tensor, **kwargs
    ): ...
    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]: ...
    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    @property
    def mamba_type(self) -> str: ...

def plamo2_mamba_mixer(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
def plamo2_mamba_mixer_fake(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...

class DenseMLP(nn.Module):
    hidden_size: Incomplete
    intermediate_size: Incomplete
    gate_up_proj: Incomplete
    act: Incomplete
    down_proj: Incomplete
    def __init__(
        self,
        config: Plamo2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Plamo2AttentionMixer(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    attn: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...

class Plamo2DecoderLayer(nn.Module):
    is_mamba: Incomplete
    mixer: Incomplete
    mlp: Incomplete
    pre_mixer_norm: Incomplete
    post_mixer_norm: Incomplete
    pre_mlp_norm: Incomplete
    post_mlp_norm: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, layer_idx: int, prefix: str = "", **kwargs
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ): ...

class Plamo2Decoder(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor: ...

class Plamo2Model(torch.nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    make_empty_intermediate_tensors: Incomplete
    layers: Incomplete
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

class Plamo2ForCausalLM(
    torch.nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    config: Incomplete
    vllm_config: Incomplete
    model_config: Incomplete
    scheduler_config: Incomplete
    model: Incomplete
    vocab_size: Incomplete
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
        **kwargs,
    ): ...
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]: ...
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]: ...
    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
