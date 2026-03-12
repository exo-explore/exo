import torch
import torch.nn as nn
from .openpangu import OpenPanguDecoderLayer as OpenPanguDecoderLayer
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMultiTokenPredictor as DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer as DeepSeekMultiTokenPredictorLayer,
    SharedHead as SharedHead,
)
from vllm.model_executor.models.utils import maybe_prefix as maybe_prefix
from vllm.sequence import IntermediateTensors as IntermediateTensors

class OpenPanguMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    config: Incomplete
    enorm: Incomplete
    hnorm: Incomplete
    eh_proj: Incomplete
    shared_head: Incomplete
    mtp_block: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None: ...

class OpenPanguMultiTokenPredictor(DeepSeekMultiTokenPredictor):
    mtp_start_layer_idx: Incomplete
    num_mtp_layers: Incomplete
    layers: Incomplete
    embed_tokens: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class OpenPanguMTP(nn.Module):
    config: Incomplete
    model: Incomplete
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
    def get_spec_layer(self, name): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
