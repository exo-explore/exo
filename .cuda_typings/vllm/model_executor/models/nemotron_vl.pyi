import abc
import torch
import torch.nn as nn
import torchvision.transforms as T
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsCrossEncoding as SupportsCrossEncoding,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .interfaces_base import VllmModelForPooling as VllmModelForPooling
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from PIL import Image as Image
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import PretrainedConfig as PretrainedConfig
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast as BaseImageProcessorFast,
)
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.pooler import DispatchPooler as DispatchPooler
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig as AWQConfig
from vllm.model_executor.models.internvl import (
    BaseInternVLDummyInputsBuilder as BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor as BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo as BaseInternVLProcessingInfo,
    InternVLImageEmbeddingInputs as InternVLImageEmbeddingInputs,
    InternVLImageInputs as InternVLImageInputs,
    InternVLImagePixelInputs as InternVLImagePixelInputs,
    InternVLProcessor as InternVLProcessor,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.siglip import SiglipVisionModel as SiglipVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.image import convert_image_mode as convert_image_mode
from vllm.multimodal.processing import PromptUpdateDetails as PromptUpdateDetails
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.transformers_utils.processor import (
    cached_image_processor_from_config as cached_image_processor_from_config,
)
from vllm.transformers_utils.repo_utils import (
    get_hf_file_to_dict as get_hf_file_to_dict,
)

def build_transform(input_size: int): ...
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]: ...
def calculate_nemotron_vl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]: ...
def dynamic_preprocess_nemotron_vl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]: ...
def get_nemotron_vl_target_ratios(
    min_num: int, max_num: int
) -> list[tuple[int, int]]: ...
def image_to_pixel_values_nemotron_vl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
    transform: T.Compose | None = None,
) -> torch.Tensor: ...

class NemotronVLProcessor(InternVLProcessor):
    IMG_START: str
    IMG_END: str
    IMG_CONTEXT: str
    config: Incomplete
    tokenizer: Incomplete
    image_processor: Incomplete
    num_image_token: Incomplete
    image_size: Incomplete
    min_dynamic_patch: Incomplete
    max_dynamic_patch: Incomplete
    dynamic_image_size: Incomplete
    use_thumbnail: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        image_processor: BaseImageProcessorFast | None = None,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None: ...
    @property
    def image_token_id(self) -> int: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    def get_image_repl(
        self, feature_size: int, num_patches: int | None
    ) -> PromptUpdateDetails[str]: ...

class NemotronVLProcessingInfo(BaseInternVLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> NemotronVLProcessor: ...
    def get_image_processor(self, **kwargs: object): ...

class LlamaNemotronVLChatModel(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    model_config: Incomplete
    multimodal_config: Incomplete
    patch_size: Incomplete
    num_image_token: Incomplete
    downsample_ratio: Incomplete
    ps_version: Incomplete
    vision_model: Incomplete
    mlp1: Incomplete
    language_model: Incomplete
    img_context_token_id: Incomplete
    visual_token_mask: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def pixel_shuffle(self, x, scale_factor: float = 0.5): ...
    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
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
    ) -> IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...

SIGLIP_MEAN: Incomplete
SIGLIP_STD: Incomplete

def build_siglip_transform(input_size: int): ...

class LlamaNemotronVLEmbedProcessor(NemotronVLProcessor):
    IMG_CONTEXT: str
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        processor_config: dict,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None: ...

class LlamaNemotronVLEmbedProcessingInfo(NemotronVLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> LlamaNemotronVLEmbedProcessor: ...

class LlamaNemotronVLForEmbedding(
    LlamaNemotronVLChatModel, VllmModelForPooling, metaclass=abc.ABCMeta
):
    is_pooling_model: bool
    weight_mapper: Incomplete
    img_context_token_id: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class LlamaNemotronVLForSequenceClassification(
    LlamaNemotronVLForEmbedding, SupportsCrossEncoding, metaclass=abc.ABCMeta
):
    weight_mapper: Incomplete
    score: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
