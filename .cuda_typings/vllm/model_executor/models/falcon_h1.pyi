import abc
import torch
from .interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    SupportsLoRA as SupportsLoRA,
    SupportsMambaPrefixCaching as SupportsMambaPrefixCaching,
    SupportsPP as SupportsPP,
)
from .utils import (
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import FalconH1Config as FalconH1Config
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_pp_group as get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
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
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.config import (
    set_default_rope_theta as set_default_rope_theta,
)

class FalconH1MLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    tp_size: Incomplete
    intermediate_size: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        config: FalconH1Config,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class FalconH1SSMDecoderLayer(nn.Module):
    config: Incomplete
    tp_size: Incomplete
    d_ssm: Incomplete
    mamba: Incomplete
    groups_time_state_size: Incomplete
    zxbcdt_multipliers: Incomplete
    def __init__(
        self,
        config: FalconH1Config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None, **kwargs
    ): ...

class FalconH1AttentionDecoderLayer(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    rotary_emb: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    attn: Incomplete
    key_multiplier: Incomplete
    def __init__(
        self,
        config: FalconH1Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def self_attention(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ): ...

class FalconH1ParallelHybrid(nn.Module):
    self_attn: Incomplete
    mamba: Incomplete
    ssm_out_multiplier: Incomplete
    ssm_in_multiplier: Incomplete
    attention_in_multiplier: Incomplete
    attn_out_multiplier: Incomplete
    feed_forward: Incomplete
    input_layernorm: Incomplete
    pre_ff_layernorm: Incomplete
    def __init__(
        self,
        config: FalconH1Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs
    ): ...

class FalconH1Model(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    embedding_multiplier: Incomplete
    make_empty_intermediate_tensors: Incomplete
    final_layernorm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class FalconH1ForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsMambaPrefixCaching,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
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
    tie_word_embeddings: Incomplete
    lm_head: Incomplete
    lm_head_multiplier: Incomplete
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
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
