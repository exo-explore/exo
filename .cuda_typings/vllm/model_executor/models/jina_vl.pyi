import abc
import torch
import torch.nn as nn
from .interfaces import (
    SupportsCrossEncoding as SupportsCrossEncoding,
    SupportsMultiModal as SupportsMultiModal,
    SupportsScoreTemplate as SupportsScoreTemplate,
)
from .qwen2_vl import (
    Qwen2VLDummyInputsBuilder as Qwen2VLDummyInputsBuilder,
    Qwen2VLForConditionalGeneration as Qwen2VLForConditionalGeneration,
    Qwen2VLMultiModalProcessor as Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo as Qwen2VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import BatchFeature as BatchFeature
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.inputs import TokensPrompt as TokensPrompt
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler as DispatchPooler
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors as IntermediateTensors

logger: Incomplete

class JinaVLScorer(nn.Module):
    dense: Incomplete
    out_proj: Incomplete
    def __init__(self, model_config: ModelConfig, prefix: str = "") -> None: ...
    def forward(self, x, **kwargs): ...

class JinaVLMultiModalProcessor(Qwen2VLMultiModalProcessor): ...

class JinaVLForSequenceClassification(
    Qwen2VLForConditionalGeneration,
    SupportsCrossEncoding,
    SupportsMultiModal,
    SupportsScoreTemplate,
    metaclass=abc.ABCMeta,
):
    is_pooling_model: bool
    weight_mapper: Incomplete
    score: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    @classmethod
    def get_score_template(cls, query: str, document: str) -> str | None: ...
    @classmethod
    def post_process_tokens(cls, prompt: TokensPrompt) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
