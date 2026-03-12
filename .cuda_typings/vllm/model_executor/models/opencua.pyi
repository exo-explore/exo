import abc
from .qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration,
)
from .qwen2_vl import (
    Qwen2VLDummyInputsBuilder as Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalDataParser as Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo as Qwen2VLProcessingInfo,
)
from .utils import (
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Mapping
from transformers.models.qwen2_vl import Qwen2VLProcessor
from vllm.config import VllmConfig as VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems as MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike

class OpenCUAProcessingInfo(Qwen2VLProcessingInfo):
    def get_data_parser(self): ...
    def get_hf_config(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_hf_processor(self, **kwargs: object): ...

class OpenCUAProcessor(Qwen2VLProcessor):
    def check_argument_for_proper_class(
        self, attribute_name: str, arg: object
    ) -> None: ...
    image_token: str
    def __init__(
        self, vision_config: dict, tokenizer: TokenizerLike, **kwargs
    ) -> None: ...
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs): ...

class OpenCUAMultiModalProcessor(BaseMultiModalProcessor[OpenCUAProcessingInfo]): ...

class OpenCUADummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...

class OpenCUAForConditionalGeneration(
    Qwen2_5_VLForConditionalGeneration, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    use_data_parallel: Incomplete
    config: Incomplete
    vllm_config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    is_multimodal_pruning_enabled: Incomplete
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
