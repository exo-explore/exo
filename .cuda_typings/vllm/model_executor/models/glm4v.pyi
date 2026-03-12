import abc
import torch
from .chatglm import (
    ChatGLMBaseModel as ChatGLMBaseModel,
    ChatGLMModel as ChatGLMModel,
    GLMTransformer as GLMTransformer,
)
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from _typeshed import Incomplete
from collections.abc import Iterator, Mapping
from torch import nn
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import (
    SiluAndMul as SiluAndMul,
    get_act_fn as get_act_fn,
)
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems as MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.chatglm import ChatGLMConfig as ChatGLMConfig
from vllm.transformers_utils.processors.glm4v import GLM4VProcessor as GLM4VProcessor
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class GLMVImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]

class EVA2CLIPPatchEmbedding(nn.Module):
    proj: Incomplete
    cls_embedding: Incomplete
    position_embedding: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, images: torch.Tensor) -> torch.Tensor: ...

class EVA2CLIPAttention(nn.Module):
    hidden_size: Incomplete
    tp_size: Incomplete
    num_heads_per_rank: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    query_key_value: Incomplete
    dense: Incomplete
    attn: Incomplete
    output_dropout: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class EVA2CLIPMLP(nn.Module):
    config: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class EVA2CLIPTransformerLayer(nn.Module):
    input_layernorm: Incomplete
    attention: Incomplete
    mlp: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, hidden_states): ...

class EVA2CLIPTransformer(nn.Module):
    layers: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, hidden_states): ...

class EVA2CLIPGLU(nn.Module):
    linear_proj: Incomplete
    norm1: Incomplete
    act1: Incomplete
    act2: Incomplete
    merged_proj: Incomplete
    dense_4h_to_h: Incomplete
    def __init__(
        self,
        config,
        in_features,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class EVA2CLIPModel(nn.Module):
    patch_embedding: Incomplete
    transformer: Incomplete
    linear_proj: Incomplete
    conv: Incomplete
    boi: Incomplete
    eoi: Incomplete
    scaling_factor: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, images: torch.Tensor) -> torch.Tensor: ...

class GLM4VModel(ChatGLMModel):
    vision: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class GLM4VProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> GLM4VProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self) -> int: ...
    def get_num_image_feature_tokens(self) -> int: ...

class GLM4VDummyInputsBuilder(BaseDummyInputsBuilder[GLM4VProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class GLM4VMultiModalProcessor(BaseMultiModalProcessor[GLM4VProcessingInfo]): ...

class GLM4VForCausalLM(
    ChatGLMBaseModel,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    def get_mm_mapping(self) -> MultiModelKeys: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    transformer: GLM4VModel
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        transformer_type: type[GLM4VModel] = ...,
    ) -> None: ...
    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int, int]]: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    embed_input_ids: Incomplete
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
