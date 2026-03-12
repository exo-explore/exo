import abc
from .intern_vit import InternVisionModel as InternVisionModel
from .internvl import (
    BaseInternVLDummyInputsBuilder as BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor as BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo as BaseInternVLProcessingInfo,
    BaseInternVLProcessor as BaseInternVLProcessor,
    InternVLChatModel as InternVLChatModel,
)
from collections.abc import Mapping
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems as ImageEmbeddingItems,
    ImageProcessorItems as ImageProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)

IMG_PAD: str

class NVLMProcessor(BaseInternVLProcessor):
    @property
    def image_token_id(self) -> int: ...
    def get_image_repl(
        self, feature_size: int, num_patches: int | None
    ) -> PromptUpdateDetails[str]: ...

class NVLMProcessingInfo(BaseInternVLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> NVLMProcessor: ...

class NVLMDummyInputsBuilder(BaseInternVLDummyInputsBuilder[NVLMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class NVLMMultiModalProcessor(BaseInternVLMultiModalProcessor[NVLMProcessingInfo]): ...
class NVLM_D_Model(InternVLChatModel, metaclass=abc.ABCMeta): ...
