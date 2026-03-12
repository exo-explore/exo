import torch
import torch.nn as nn
from .step3p5 import (
    Step3p5DecoderLayer as Step3p5DecoderLayer,
    get_spec_layer_idx_from_weight_name as get_spec_layer_idx_from_weight_name,
)
from .utils import maybe_prefix as maybe_prefix
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as GemmaRMSNorm
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

logger: Incomplete

class SharedHead(nn.Module):
    norm: Incomplete
    head: Incomplete
    def __init__(
        self, config: PretrainedConfig, quant_config: QuantizationConfig | None = None
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Step3p5AMultiTokenPredictorLayer(nn.Module):
    enorm: Incomplete
    hnorm: Incomplete
    eh_proj: Incomplete
    shared_head: Incomplete
    mtp_block: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor: ...

class Step3p5AMultiTokenPredictor(nn.Module):
    embed_tokens: Incomplete
    mtp_start_layer_idx: Incomplete
    num_mtp_layers: Incomplete
    layers: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
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
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...

class Step3p5MTP(nn.Module):
    config: Incomplete
    vllm_config: Incomplete
    model: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
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
