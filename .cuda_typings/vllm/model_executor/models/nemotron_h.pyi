import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.parallel import ParallelConfig as ParallelConfig
from vllm.distributed import (
    get_ep_group as get_ep_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import get_pp_group as get_pp_group
from vllm.model_executor.layers.activation import (
    ReLUSquaredActivation as ReLUSquaredActivation,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import (
    GateLinear as GateLinear,
    SharedFusedMoE as SharedFusedMoE,
    activation_without_mul as activation_without_mul,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2 as MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
    MambaStateCopyFuncCalculator as MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    MixtureOfExperts as MixtureOfExperts,
    SupportsLoRA as SupportsLoRA,
    SupportsMambaPrefixCaching as SupportsMambaPrefixCaching,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import NemotronHConfig as NemotronHConfig

class NemotronHMLP(nn.Module):
    up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor): ...

class NemotronHMoE(nn.Module):
    tp_size: Incomplete
    routed_scaling_factor: Incomplete
    ep_group: Incomplete
    ep_rank: Incomplete
    ep_size: Incomplete
    n_routed_experts: int
    n_shared_experts: int
    use_latent_moe: bool
    moe_hidden_size: int
    is_sequence_parallel: Incomplete
    gate: Incomplete
    enable_eplb: Incomplete
    n_redundant_experts: Incomplete
    n_logical_experts: Incomplete
    n_physical_experts: Incomplete
    n_local_physical_experts: Incomplete
    physical_expert_start: Incomplete
    physical_expert_end: Incomplete
    shared_experts: Incomplete
    fc1_latent_proj: Incomplete
    fc2_latent_proj: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class NemotronHMLPDecoderLayer(nn.Module):
    config: Incomplete
    mixer: Incomplete
    norm: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None, **kwargs
    ): ...

class NemotronHMoEDecoderLayer(nn.Module):
    config: Incomplete
    mixer: Incomplete
    norm: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None, **kwargs
    ): ...

class NemotronHMambaDecoderLayer(nn.Module):
    config: Incomplete
    mixer: Incomplete
    norm: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None, **kwargs
    ): ...

class NemotronHAttention(nn.Module):
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
    attn: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor: ...

class NemotronHAttentionDecoderLayer(nn.Module):
    mixer: Incomplete
    norm: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ): ...

ALL_DECODER_LAYER_TYPES: Incomplete

class NemotronHModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    has_moe: Incomplete
    make_empty_intermediate_tensors: Incomplete
    norm_f: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def is_spec_layer(self, config: NemotronHConfig, weight_name: str) -> bool: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class NemotronHForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsQuant,
    MixtureOfExperts,
    SupportsMambaPrefixCaching,
    metaclass=abc.ABCMeta,
):
    is_non_gated_moe: bool
    hf_to_vllm_mapper: Incomplete
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
    lora_skip_prefixes: Incomplete
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
    vllm_config: Incomplete
    model_config: Incomplete
    quant_config: Incomplete
    config: Incomplete
    scheduler_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    expert_weights: Incomplete
    num_expert_groups: Incomplete
    moe_layers: Incomplete
    num_moe_layers: Incomplete
    num_logical_experts: Incomplete
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_routed_experts: Incomplete
    num_shared_experts: Incomplete
    num_redundant_experts: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ): ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
