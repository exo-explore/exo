import abc
import torch
import torch.nn as nn
from .idefics2_vision_model import (
    Idefics2VisionConfig as Idefics2VisionConfig,
    Idefics2VisionTransformer as Idefics3VisionTransformer,
)
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsQuant as SupportsQuant,
)
from .llama import (
    LlamaDecoderLayer as LlamaDecoderLayer,
    LlamaMLP as LlamaMLP,
    LlamaModel as LlamaModel,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    is_pp_missing_parameter as is_pp_missing_parameter,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import (
    AriaConfig,
    AriaTextConfig as AriaTextConfig,
    BatchFeature as BatchFeature,
)
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
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
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class AriaImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    pixel_mask: Annotated[torch.Tensor | None, None]

class AriaVisionTransformer(Idefics3VisionTransformer, SupportsQuant):
    packed_modules_mapping: Incomplete
    post_layernorm: Incomplete
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class AriaProjectorMLP(nn.Module):
    linear_in: Incomplete
    linear_out: Incomplete
    act: Incomplete
    def __init__(
        self, in_features: int, hidden_features: int, output_dim: int, prefix: str = ""
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class AriaProjector(nn.Module):
    patch_to_query_dict: Incomplete
    in_features: Incomplete
    num_heads: Incomplete
    kv_dim: Incomplete
    hidden_features: Incomplete
    output_dim: Incomplete
    query: Incomplete
    cross_attn: Incomplete
    layer_norm: Incomplete
    feed_forward: Incomplete
    def __init__(self, config: AriaConfig, prefix: str = "") -> None: ...
    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class AriaFusedMoE(SharedFusedMoE):
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None: ...

class AriaTextMoELayer(nn.Module):
    config: Incomplete
    router_weight: Incomplete
    shared_experts: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: AriaTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class AriaTextDecoderLayer(LlamaDecoderLayer):
    mlp: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class AriaTextModel(LlamaModel, SupportsQuant):
    packed_modules_mapping: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class AriaProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_vision_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_image_tokens(self) -> int: ...

class AriaDummyInputsBuilder(BaseDummyInputsBuilder[AriaProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class AriaMultiModalProcessor(BaseMultiModalProcessor[AriaProcessingInfo]): ...

class AriaForConditionalGeneration(
    nn.Module, SupportsMultiModal, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
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
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
