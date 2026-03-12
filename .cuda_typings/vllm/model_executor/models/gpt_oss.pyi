import abc
import torch
from .interfaces import (
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import GptOssConfig as GptOssConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_dp_group as get_dp_group,
    get_ep_group as get_ep_group,
    get_pcp_group as get_pcp_group,
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig as FusedMoEParallelConfig,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
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
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE as OCP_MX_BLOCK_SIZE,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.utils import (
    rocm_unquantized_gemm as rocm_unquantized_gemm,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.attention.backend import AttentionType as AttentionType

class OAIAttention(nn.Module):
    layer_idx: Incomplete
    head_dim: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    hidden_size: Incomplete
    rotary_emb: Incomplete
    sinks: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    num_local_attention_heads: Incomplete
    num_local_key_value_heads: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: GptOssConfig,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor: ...

class MLPBlock(torch.nn.Module):
    is_sequence_parallel: Incomplete
    layer_idx: Incomplete
    num_experts: Incomplete
    hidden_size: Incomplete
    experts_per_token: Incomplete
    world_size: Incomplete
    router: Incomplete
    experts: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, layer_idx: int, prefix: str = ""
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class TransformerBlock(torch.nn.Module):
    layer_idx: Incomplete
    attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor: ...

class GptOssModel(nn.Module):
    config: Incomplete
    quant_config: Incomplete
    parallel_config: Incomplete
    embedding: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    aux_hidden_state_layers: Incomplete
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

class GptOssForCausalLM(
    nn.Module, SupportsPP, SupportsEagle3, SupportsLoRA, metaclass=abc.ABCMeta
):
    is_3d_moe_weight: bool
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    vllm_config: Incomplete
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
