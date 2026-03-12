import torch
from .interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    SupportsMambaPrefixCaching as SupportsMambaPrefixCaching,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import Zamba2Config as Zamba2Config
from typing import Any
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import GeluAndMul as GeluAndMul
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
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class Zamba2LoRA(nn.Module):
    A: Incomplete
    B: Incomplete
    def __init__(
        self,
        input_dim: int,
        rank: int,
        output_dim: int | list[int],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class Zamba2Attention(nn.Module):
    config: Incomplete
    num_hybrid_layers: Incomplete
    attention_hidden_size: Incomplete
    total_num_attention_heads: Incomplete
    num_attention_heads: Incomplete
    attention_head_dim: Incomplete
    qkv_size: Incomplete
    scale: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    dpa_list: Incomplete
    linear_q_adapter_list: Incomplete
    linear_k_adapter_list: Incomplete
    linear_v_adapter_list: Incomplete
    rotary_emb: Incomplete
    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        num_hybrid_layers: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, block_idx: int, position_ids: torch.Tensor
    ) -> torch.Tensor: ...

class Zamba2MLP(nn.Module):
    config: Incomplete
    tp_size: Incomplete
    num_hybrid_layers: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    gate_up_proj_adapter_list: Incomplete
    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        num_hybrid_layers: dict[int, int],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, block_idx: int) -> torch.Tensor: ...

class Zamba2AttentionDecoderLayer(nn.Module):
    self_attn: Incomplete
    feed_forward: Incomplete
    input_layernorm: Incomplete
    pre_ff_layernorm: Incomplete
    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        num_hybrid_layers: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        block_idx: int,
        positions: torch.Tensor,
    ) -> torch.Tensor: ...

class Zamba2MambaDecoderLayer(nn.Module):
    mamba: Incomplete
    input_layernorm: Incomplete
    def __init__(
        self,
        config: Zamba2Config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        transformer_hidden_states: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        original_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Zamba2HybridLayer(nn.Module):
    block_idx: Incomplete
    shared_transformer: Incomplete
    linear: Incomplete
    mamba_decoder: Incomplete
    def __init__(
        self,
        shared_transformer: Zamba2AttentionDecoderLayer,
        config: Zamba2Config,
        block_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor: ...

class Zamba2Model(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    final_layernorm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Zamba2ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMambaPrefixCaching):
    hf_to_vllm_mapper: Incomplete
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
    config: Incomplete
    vllm_config: Incomplete
    scheduler_config: Incomplete
    model_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
