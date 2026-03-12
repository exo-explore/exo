import torch
import torch.nn as nn
from .glm4_moe import (
    Glm4MixtureOfExperts as Glm4MixtureOfExperts,
    Glm4MoE as Glm4MoE,
    Glm4MoeDecoderLayer as Glm4MoeDecoderLayer,
    get_spec_layer_idx_from_weight_name as get_spec_layer_idx_from_weight_name,
)
from .utils import maybe_prefix as maybe_prefix
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import (
    CacheConfig as CacheConfig,
    ParallelConfig as ParallelConfig,
    VllmConfig as VllmConfig,
)
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
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
)
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

class Glm4MoeMultiTokenPredictorLayer(nn.Module):
    enorm: Incomplete
    hnorm: Incomplete
    eh_proj: Incomplete
    shared_head: Incomplete
    enable_eplb: Incomplete
    mtp_block: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor: ...

class Glm4MoeMultiTokenPredictor(nn.Module):
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

class Glm4MoeMTP(nn.Module, Glm4MixtureOfExperts):
    config: Incomplete
    model: Incomplete
    expert_weights: Incomplete
    num_moe_layers: Incomplete
    num_expert_groups: Incomplete
    moe_layers: list[FusedMoE]
    moe_mlp_layers: list[Glm4MoE]
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
