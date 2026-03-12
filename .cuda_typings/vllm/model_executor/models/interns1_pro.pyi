import abc
import functools
import torch
from .interfaces import MixtureOfExperts as MixtureOfExperts
from .qwen3_moe import Qwen3MoeForCausalLM as Qwen3MoeForCausalLM
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder as Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration as Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor as Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo as Qwen3VLProcessingInfo,
    Qwen3_VisionTransformer as Qwen3_VisionTransformer,
)
from .qwen3_vl_moe import Qwen3MoeLLMModel as Qwen3MoeLLMModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    extract_layer_index as extract_layer_index,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import (
    AutoProcessor as AutoProcessor,
    PretrainedConfig as PretrainedConfig,
)
from typing import Any
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_ep_group as get_ep_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
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
)
from vllm.model_executor.models.utils import (
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY

logger: Incomplete

class InternS1ProProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> AutoProcessor: ...

class InternS1ProMoeMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class InternS1ProMoeSparseMoeBlock(nn.Module):
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
    n_groups: Incomplete
    experts: Incomplete
    gate: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @staticmethod
    @functools.lru_cache
    def get_group_offsets(n_groups: int, group_size: int, device: str): ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class InternS1ProMoeAttention(nn.Module):
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
    dual_chunk_attention_config: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class InternS1ProMoeDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class InternS1ProMoeLLMModel(Qwen3MoeLLMModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[torch.nn.Module] = ...,
    ) -> None: ...

class InternS1ProMoeLLMForCausalLM(Qwen3MoeForCausalLM, metaclass=abc.ABCMeta):
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class InternS1ProMoeMixtureOfExperts(MixtureOfExperts):
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

class InternS1ProForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    InternS1ProMoeMixtureOfExperts,
    metaclass=abc.ABCMeta,
):
    is_3d_moe_weight: bool
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    video_pruning_rate: Incomplete
    is_multimodal_pruning_enabled: Incomplete
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    use_deepstack: Incomplete
    deepstack_num_level: Incomplete
    visual_dim: Incomplete
    multiscale_dim: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_frope_params_map(self) -> str: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
