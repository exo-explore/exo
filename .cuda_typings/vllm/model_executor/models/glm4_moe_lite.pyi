import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig as Glm4MoeLiteConfig
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention as DeepseekV2Attention,
    DeepseekV2MLAAttention as DeepseekV2MLAAttention,
)
from vllm.model_executor.models.glm4_moe import (
    Glm4MixtureOfExperts as Glm4MixtureOfExperts,
    Glm4MoE as Glm4MoE,
    Glm4MoeMLP as Glm4MoeMLP,
)
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors

logger: Incomplete

class Glm4MoeLiteMLP(Glm4MoeMLP): ...
class Glm4MoeLite(Glm4MoE): ...
class Glm4LiteMixtureOfExperts(Glm4MixtureOfExperts, metaclass=abc.ABCMeta): ...
class Glm4MoeLiteAttention(DeepseekV2Attention): ...
class Glm4MoeLiteMLAAttention(DeepseekV2MLAAttention): ...

class Glm4MoeLiteDecoderLayer(nn.Module):
    hidden_size: Incomplete
    layer_idx: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    routed_scaling_factor: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: Glm4MoeLiteConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class Glm4MoeLiteModel(nn.Module):
    config: Incomplete
    device: Incomplete
    vocab_size: Incomplete
    is_v32: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Glm4MoeLiteForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, Glm4LiteMixtureOfExperts, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    use_mha: Incomplete
    fuse_qkv_a_proj: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    num_moe_layers: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    expert_weights: Incomplete
    num_expert_groups: Incomplete
    moe_layers: Incomplete
    moe_mlp_layers: Incomplete
    def set_moe_parameters(self) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def get_spec_layer_idx_from_weight_name(
    config: Glm4MoeLiteConfig, weight_name: str
) -> int | None: ...
