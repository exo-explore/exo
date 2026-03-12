import abc
import torch
import torch.nn as nn
from .glm4_moe_lite import (
    Glm4MixtureOfExperts as Glm4MixtureOfExperts,
    Glm4MoeLite as Glm4MoeLite,
    Glm4MoeLiteDecoderLayer as Glm4MoeLiteDecoderLayer,
    get_spec_layer_idx_from_weight_name as get_spec_layer_idx_from_weight_name,
)
from .interfaces import SupportsPP as SupportsPP
from .utils import maybe_prefix as maybe_prefix
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.fused_moe import (
    FusedMoE as FusedMoE,
    SharedFusedMoE as SharedFusedMoE,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
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
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors

class SharedHead(nn.Module):
    norm: Incomplete
    head: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Glm4MoeLiteMultiTokenPredictorLayer(nn.Module):
    config: Incomplete
    enorm: Incomplete
    hnorm: Incomplete
    eh_proj: Incomplete
    device: Incomplete
    is_v32: Incomplete
    shared_head: Incomplete
    mtp_block: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor: ...

class Glm4MoeLiteMultiTokenPredictor(nn.Module):
    mtp_start_layer_idx: Incomplete
    num_mtp_layers: Incomplete
    layers: Incomplete
    embed_tokens: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor: ...
    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor: ...

class Glm4MoeLiteMTP(
    nn.Module, SupportsPP, Glm4MixtureOfExperts, metaclass=abc.ABCMeta
):
    config: Incomplete
    model: Incomplete
    expert_weights: Incomplete
    num_moe_layers: Incomplete
    num_expert_groups: Incomplete
    moe_layers: list[FusedMoE]
    moe_mlp_layers: list[Glm4MoeLite]
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor: ...
    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
