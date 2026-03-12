import abc
import torch
import torch.nn as nn
from .interfaces import (
    IsAttentionFree as IsAttentionFree,
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
)
from .interfaces_base import attn_type as attn_type
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from functools import cached_property as cached_property
from terratorch.vllm import InputDefinition
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.pooler import IdentityPooler as IdentityPooler
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.utils import AutoWeightsLoader as AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem as ImageItem,
    ModalityData as ModalityData,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalInputs as MultiModalInputs,
    MultiModalKwargsItems as MultiModalKwargsItems,
    PlaceholderRange as PlaceholderRange,
    mm_inputs as mm_inputs,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    ProcessorInputs as ProcessorInputs,
    PromptUpdate as PromptUpdate,
    TimingContext as TimingContext,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

logger: Incomplete

class TerratorchMultiModalDataParser(MultiModalDataParser):
    input_definition: Incomplete
    def __init__(self, input_definition: InputDefinition, *args, **kwargs) -> None: ...
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems: ...

class TerratorchProcessingInfo(BaseProcessingInfo):
    @cached_property
    def input_definition(self) -> InputDefinition: ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class TerratorchInputBuilder(BaseDummyInputsBuilder[TerratorchProcessingInfo]):
    dummy_data_generator: Incomplete
    def __init__(self, info: TerratorchProcessingInfo) -> None: ...
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class TerratorchMultiModalProcessor(BaseMultiModalProcessor[TerratorchProcessingInfo]):
    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInputs: ...

class Terratorch(nn.Module, IsAttentionFree, SupportsMultiModal, metaclass=abc.ABCMeta):
    supports_multimodal_raw_input_only: bool
    is_pooling_model: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    inference_runner: Incomplete
    model: Incomplete
    pooler: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
