import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    BaseImageProcessor as BaseImageProcessor,
    PretrainedConfig as PretrainedConfig,
)
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.ovis import VisualEmbedding as VisualEmbedding
from vllm.model_executor.models.siglip2navit import (
    Siglip2NavitModel as Siglip2NavitModel,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    flatten_bn as flatten_bn,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.processors.ovis2_5 import (
    Ovis2_5Processor as Ovis2_5Processor,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

IMAGE_TOKEN: str
VIDEO_TOKEN: str
INDICATOR_IDS: Incomplete
IMAGE_PAD_TOKEN_ID: int

class Ovis2_5ImagePatchInputs(TensorSchema):
    type: Literal["image_patches"]
    flat_data: Annotated[torch.Tensor, None]
    indicator_tokens: Annotated[torch.Tensor, None]
    patches_per_item: Annotated[list[int], None]
    grids: Annotated[torch.Tensor, None]

class Ovis2_5VideoPatchInputs(TensorSchema):
    type: Literal["video_patches"]
    flat_data: Annotated[torch.Tensor, None]
    indicator_tokens: Annotated[torch.Tensor, None]
    patches_per_item: Annotated[list[int], None]
    grids: Annotated[torch.Tensor, None]

class VisualTokenizer(torch.nn.Module):
    config: Incomplete
    vit: Incomplete
    head: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        visual_vocab_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def tokenize(self, logits: torch.Tensor) -> torch.Tensor: ...
    def encode(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor: ...

class Ovis2_5ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs): ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    def get_image_processor(self) -> BaseImageProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, num_frames: int = 1
    ) -> int: ...
    def get_max_image_tokens(self) -> int: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...
    def get_num_video_tokens(
        self, *, image_width: int, image_height: int, num_frames: int
    ) -> int: ...
    def get_max_video_tokens(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class Ovis2_5DummyInputsBuilder(BaseDummyInputsBuilder[Ovis2_5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Ovis2_5MultiModalProcessor(BaseMultiModalProcessor[Ovis2_5ProcessingInfo]):
    def visual_indicators_to_visual_tokens(
        self, visual_indicators: list[int]
    ) -> list[int]: ...

class Ovis2_5(nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: PretrainedConfig
    llm: Incomplete
    visual_tokenizer: Incomplete
    vte: Incomplete
    image_pad_token_id: int
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
