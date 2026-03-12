import abc
import torch
from _typeshed import Incomplete
from collections.abc import Mapping
from transformers import BatchFeature as BatchFeature
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.config.utils import getattr_iter as getattr_iter
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.multimodal import MultiModalKwargsItems as MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalInputs as MultiModalInputs,
    PlaceholderRange as PlaceholderRange,
    mm_inputs as mm_inputs,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    ProcessorInputs as ProcessorInputs,
    TimingContext as TimingContext,
)
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors

DYNAMIC_ARG_DIMS: Incomplete
logger: Incomplete

class MultiModalProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self): ...
    def get_mm_max_tokens_per_item(self, seq_len, mm_counts): ...
    def get_max_image_tokens(self) -> int: ...
    def get_max_image_size(self): ...

class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder[MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"],
    ) -> MultiModalDataDict: ...

class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):
    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInputs: ...

class MultiModalMixin(SupportsMultiModal, SupportsMRoPE, metaclass=abc.ABCMeta):
    supports_multimodal_raw_input_only: bool
    hf_to_vllm_mapper: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def get_language_model(self) -> torch.nn.Module: ...
    def embed_multimodal(self, **kwargs): ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
