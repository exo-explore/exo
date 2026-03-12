import torch
import torch.nn as nn
from .deepseek_v2 import DeepseekV2DecoderLayer as DeepseekV2DecoderLayer
from .utils import maybe_prefix as maybe_prefix
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    block_dequant as block_dequant,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.longcat_flash import FlashConfig as FlashConfig
from vllm.sequence import IntermediateTensors as IntermediateTensors

class LongCatMultiTokenPredictorLayer(nn.Module):
    enorm: Incomplete
    hnorm: Incomplete
    eh_proj: Incomplete
    mtp_block: Incomplete
    final_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor: ...

class LongCatMultiTokenPredictor(nn.Module):
    mtp_start_layer_idx: Incomplete
    num_mtp_layers: int
    layers: Incomplete
    embed_tokens: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor: ...

class LongCatFlashMTP(nn.Module):
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
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
    def get_spec_layer_idx_from_weight_name(
        self, config: PretrainedConfig, weight_name: str
    ) -> int | None: ...
