import torch
import torch.nn as nn
from .llama import LlamaDecoderLayer as LlamaDecoderLayer
from .utils import (
    is_pp_missing_parameter as is_pp_missing_parameter,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import VllmConfig as VllmConfig
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
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class ErnieMultiTokenPredictorLayer(nn.Module):
    mtp_emb_norm: Incomplete
    mtp_hidden_norm: Incomplete
    mtp_linear_proj: Incomplete
    mtp_block: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor: ...

class ErnieMultiTokenPredictor(nn.Module):
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
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor: ...

class ErnieMTP(nn.Module):
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
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
