import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_vision_model as run_dp_sharded_vision_model,
)
from PIL import Image
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    BatchFeature,
    PretrainedConfig as PretrainedConfig,
    TensorType as TensorType,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
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
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.transformers_utils.configs import (
    Step3VisionEncoderConfig as Step3VisionEncoderConfig,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Step3VLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    patch_pixel_values: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class Step3VLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

Step3VLImageInputs: TypeAlias = Step3VLImagePixelInputs | Step3VLImageEmbeddingInputs
ImageWithPatches: Incomplete
MAX_IMAGE_SIZE: int

class Step3VisionProcessor:
    transform: Incomplete
    patch_transform: Incomplete
    def __init__(
        self, size, interpolation_mode: str = "bicubic", patch_size=None
    ) -> None: ...
    def __call__(self, image, is_patch: bool = False): ...

class ImagePatcher:
    enable_patch: Incomplete
    def __init__(self, enable_patch: bool = True) -> None: ...
    def determine_window_size(self, long: int, short: int) -> int: ...
    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]: ...
    def square_pad(self, img: Image.Image) -> Image.Image: ...
    def get_image_size_for_padding(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]: ...
    def get_image_size_for_preprocess(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]: ...
    def get_image_size_for_crop(
        self, img_width: int, img_height: int, window_size: int
    ): ...
    def patch_crop(self, img: Image.Image, i: int, j: int, th: int, tw: int): ...
    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]: ...
    def __call__(
        self, img: Image.Image
    ) -> tuple[Image.Image, list[Image.Image], list[bool] | None]: ...

class Step3VLProcessor:
    config: Incomplete
    tokenizer: Incomplete
    image_size: int
    patch_size: int
    image_preprocessor: Incomplete
    num_image_feature_size: int
    num_patch_feature_size: int
    image_token: str
    image_feature_placeholder: Incomplete
    patch_feature_placeholder: Incomplete
    patcher: Incomplete
    def __init__(self, config: PretrainedConfig, tokenizer: TokenizerLike) -> None: ...
    @property
    def image_token_id(self) -> int: ...
    def get_num_image_tokens(self, img_width: int, img_height: int) -> int: ...
    def replace_placeholder(
        self, text: str, placeholder: str, repls: list[str]
    ) -> str: ...
    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature: ...

class Step3VLProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self) -> Step3VLProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_max_image_tokens(self) -> int: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_num_mm_tokens(self, mm_data: MultiModalDataDict) -> int: ...

class Step3VLDummyInputsBuilder(BaseDummyInputsBuilder[Step3VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Step3VLMultiModalProcessor(BaseMultiModalProcessor[Step3VLProcessingInfo]): ...

def get_abs_pos(abs_pos, tgt_size): ...

class Step3VisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    class_embedding: Incomplete
    patch_embedding: Incomplete
    num_patches: Incomplete
    pad_tp_size: int
    position_embedding: Incomplete
    def __init__(self, config: Step3VisionEncoderConfig) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class Step3VisionAttention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    total_num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    num_heads: Incomplete
    q_size: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor): ...

class Step3VisionMLP(nn.Module):
    config: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Step3VisionEncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    layer_norm1: Incomplete
    mlp: Incomplete
    layer_norm2: Incomplete
    def __init__(
        self,
        config: Step3VisionEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.FloatTensor: ...

class Step3VisionEncoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    def __init__(
        self,
        config: Step3VisionEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, inputs_embeds): ...

class Step3VisionTransformer(nn.Module):
    config: Incomplete
    use_data_parallel: Incomplete
    image_size: Incomplete
    embeddings: Incomplete
    transformer: Incomplete
    def __init__(
        self,
        config: Step3VisionEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, pixel_values: torch.Tensor): ...

class Step3VLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    vision_model: Incomplete
    vit_downsampler: Incomplete
    vit_downsampler2: Incomplete
    vit_large_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    @property
    def device(self): ...
    @property
    def dtype(self): ...
    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings: ...
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
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
