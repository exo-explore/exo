import abc
import torch
from .interfaces import SupportsPP as SupportsPP
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
from typing import Any
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
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
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.step3_vl import Step3TextConfig as Step3TextConfig

logger: Incomplete

class FusedMoEBlock(nn.Module):
    tp_size: Incomplete
    experts: Incomplete
    gate: Incomplete
    def __init__(
        self,
        config: ModelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Step3TextMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    hidden_size: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Step3TextAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    kv_size: Incomplete
    q_size: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    inter_norm: Incomplete
    wq: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        rope_parameters: dict[str, Any],
        share_q_dim: int | None = None,
        max_position_embedding: int = 8192,
        head_dim: int = 256,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Step3TextDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    moe: Incomplete
    share_expert: Incomplete
    use_moe: bool
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: Step3TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Step3TextModel(nn.Module):
    vocab_size: Incomplete
    config: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Step3TextForCausalLM(nn.Module, SupportsPP, metaclass=abc.ABCMeta):
    config: Incomplete
    vllm_config: Incomplete
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
    ): ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
