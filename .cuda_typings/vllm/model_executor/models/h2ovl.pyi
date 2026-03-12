import abc
import torch
from .intern_vit import InternVisionModel as InternVisionModel
from .internvl import (
    BaseInternVLDummyInputsBuilder as BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor as BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo as BaseInternVLProcessingInfo,
    BaseInternVLProcessor as BaseInternVLProcessor,
    IMG_CONTEXT as IMG_CONTEXT,
    IMG_END as IMG_END,
    IMG_START as IMG_START,
    InternVLChatModel as InternVLChatModel,
    build_transform as build_transform,
    find_closest_aspect_ratio as find_closest_aspect_ratio,
    get_internvl_target_ratios as get_internvl_target_ratios,
)
from PIL import Image as Image
from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems as MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems as ImageEmbeddingItems,
    ImageProcessorItems as ImageProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing.processor import (
    MultiModalProcessingInfo as MultiModalProcessingInfo,
    ProcessorInputs as ProcessorInputs,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
    TimingContext as TimingContext,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike

def resolve_h2ovl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]: ...
def get_h2ovl_target_ratios(
    min_num: int, max_num: int, *, prior_aspect_ratio: tuple[int, int] | None
) -> list[tuple[int, int]]: ...
def calculate_h2ovl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int, tuple[int, int]]: ...
def dynamic_preprocess_h2ovl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[list[Image.Image], tuple[int, int]]: ...
def image_to_pixel_values_h2ovl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
    use_msac: bool,
) -> torch.Tensor: ...

class H2OVLProcessor(BaseInternVLProcessor):
    use_msac: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_msac: bool | None = None,
    ) -> None: ...
    @property
    def image_token_id(self) -> int: ...
    def get_image_repl(
        self, feature_size: int, num_patches: int | None
    ) -> PromptUpdateDetails[str]: ...
    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
    ) -> tuple[int, int]: ...
    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
        prior_aspect_ratio: tuple[int, int] | None = None,
        override_min_num: int | None = None,
    ) -> list[tuple[int, int]]: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, use_msac: bool | None = None
    ) -> int: ...

class H2OVLProcessingInfo(BaseInternVLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> H2OVLProcessor: ...
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: H2OVLProcessor,
        use_msac: bool | None = None,
    ) -> int: ...

class H2OVLMultiModalProcessor(
    BaseInternVLMultiModalProcessor[H2OVLProcessingInfo]
): ...
class H2OVLChatModel(InternVLChatModel, metaclass=abc.ABCMeta): ...
