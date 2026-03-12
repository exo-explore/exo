import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .qwen import (
    QWenBaseModel as QWenBaseModel,
    QWenBlock as QWenBlock,
    QWenModel as QWenModel,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Mapping
from torch import nn
from transformers import BatchFeature as BatchFeature
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.resampler import (
    Resampler2 as Resampler2,
    get_abs_pos as get_abs_pos,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
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
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.processors.qwen_vl import (
    QwenVLProcessor as QwenVLProcessor,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class QwenImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, None]

class QwenImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

QwenImageInputs: TypeAlias = QwenImagePixelInputs | QwenImageEmbeddingInputs

class VisualAttention(nn.Module):
    embed_dim: Incomplete
    kdim: Incomplete
    vdim: Incomplete
    num_heads: Incomplete
    hidden_size_per_attention_head: Incomplete
    num_attention_heads_per_partition: Incomplete
    hidden_size_per_partition: Incomplete
    in_proj: Incomplete
    out_proj: Incomplete
    norm_factor: Incomplete
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class QwenVLMLP(nn.Module):
    c_fc: Incomplete
    act_fn: Incomplete
    c_proj: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class VisualAttentionBlock(nn.Module):
    ln_1: Incomplete
    ln_2: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[[int], nn.Module] = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def attention(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class TransformerBlock(nn.Module):
    width: Incomplete
    layers: Incomplete
    resblocks: Incomplete
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[[int], nn.Module] = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def get_cast_dtype(self) -> torch.dtype: ...
    def get_cast_device(self) -> torch.device: ...
    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class VisionTransformer(nn.Module):
    grid_size: Incomplete
    output_dim: Incomplete
    conv1: Incomplete
    positional_embedding: Incomplete
    ln_pre: Incomplete
    transformer: Incomplete
    attn_pool: Incomplete
    ln_post: Incomplete
    proj: Incomplete
    image_start_id: Incomplete
    image_end_id: Incomplete
    image_pad_id: Incomplete
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        n_queries: int = 256,
        output_dim: int = 512,
        image_start_id: int = 151857,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class QwenVLModel(QWenModel):
    visual: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class QwenVLProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> QwenVLProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self) -> int: ...

class QwenVLDummyInputsBuilder(BaseDummyInputsBuilder[QwenVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class QwenVLMultiModalProcessor(BaseMultiModalProcessor[QwenVLProcessingInfo]): ...

class QwenVLForConditionalGeneration(
    QWenBaseModel, SupportsPP, SupportsLoRA, SupportsMultiModal, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    embed_input_ids: Incomplete
    def get_mm_mapping(self) -> MultiModelKeys: ...
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    transformer: QwenVLModel
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        transformer_type: type[QwenVLModel] = ...,
    ) -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
