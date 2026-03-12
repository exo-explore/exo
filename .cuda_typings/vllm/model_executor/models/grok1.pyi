import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
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
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import GeluAndMul as GeluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
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
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

DEFAULT_ATTN_OUTPUT_MULTIPLIER: float
DEFAULT_OUTPUT_MULTIPLIER_SCALE: float
DEFAULT_EMBEDDING_MULTIPLIER_SCALE: float
DEFAULT_ROUTER_LOGIT_SOFTCAP: float
logger: Incomplete

class Grok1MLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Grok1MoE(nn.Module):
    hidden_size: Incomplete
    gate: Incomplete
    experts: Incomplete
    router_logit_soft_cap: Incomplete
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        router_logit_soft_cap: float = 0.0,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        renormalize: bool = False,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Grok1Attention(nn.Module):
    hidden_size: Incomplete
    config: Incomplete
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
    attn: Incomplete
    attn_multiplier: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = ...,
        rope_parameters: dict[str, Any] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        config=None,
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Grok1DecoderLayer(nn.Module):
    hidden_size: Incomplete
    use_fp8: bool
    attn: Incomplete
    moe_block: Incomplete
    residual_moe: Incomplete
    residual_moe_scale: Incomplete
    pre_attn_norm: Incomplete
    post_attn_norm: Incomplete
    pre_moe_norm: Incomplete
    post_moe_norm: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        config,
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

class Grok1Model(nn.Module):
    config: Incomplete
    quant_config: Incomplete
    ckpt_gate_proj_name: Incomplete
    ckpt_down_proj_name: Incomplete
    ckpt_up_proj_name: Incomplete
    weight_name_remapping: Incomplete
    vocab_size: Incomplete
    embedding_multiplier_scale: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        ckpt_gate_proj_name: str = "linear",
        ckpt_down_proj_name: str = "linear_1",
        ckpt_up_proj_name: str = "linear_v",
        weight_name_remapping: dict[str, str] | None = None,
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class GrokBaseForCausalLM(nn.Module, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta):
    fall_back_to_pt_during_load: bool
    packed_modules_mapping: Incomplete
    ckpt_gate_proj_name: str
    ckpt_down_proj_name: str
    ckpt_up_proj_name: str
    def get_weight_name_remapping(self) -> dict[str, str]: ...
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    output_multiplier_scale: Incomplete
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
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...

class Grok1ForCausalLM(GrokBaseForCausalLM, metaclass=abc.ABCMeta):
    ckpt_gate_proj_name: str
    ckpt_down_proj_name: str
    ckpt_up_proj_name: str
    def get_weight_name_remapping(self) -> dict[str, str]: ...

class Grok2ForCausalLM(GrokBaseForCausalLM, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    ckpt_gate_proj_name: str
    ckpt_down_proj_name: str
    ckpt_up_proj_name: str
    def get_weight_name_remapping(self) -> dict[str, str]: ...

class GrokForCausalLM(GrokBaseForCausalLM, metaclass=abc.ABCMeta):
    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""): ...
