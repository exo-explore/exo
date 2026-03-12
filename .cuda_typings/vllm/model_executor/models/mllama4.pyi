import abc
import torch
from .interfaces import (
    MixtureOfExperts as MixtureOfExperts,
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .llama4 import Llama4ForCausalLM as Llama4ForCausalLM
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    StageMissingLayer as StageMissingLayer,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_vision_model as run_dp_sharded_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import (
    BatchFeature as BatchFeature,
    Llama4Config,
    Llama4VisionConfig as Llama4VisionConfig,
)
from transformers.models.llama4 import Llama4Processor
from typing import Annotated, Literal
from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder as should_torch_compile_mm_encoder,
    support_torch_compile as support_torch_compile,
)
from vllm.config import (
    VllmConfig as VllmConfig,
    set_current_vllm_config as set_current_vllm_config,
)
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.model_loader.utils import initialize_model as initialize_model
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    InputProcessingContext as InputProcessingContext,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Llama4ImagePatchInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    patches_per_image: Annotated[torch.Tensor, None]
    aspect_ratios: Annotated[torch.Tensor, None]

class Llama4VisionMLP(nn.Module):
    fc1: Incomplete
    fc2: Incomplete
    activation_fn: Incomplete
    output_activation: Incomplete
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        bias: bool,
        output_activation: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Llama4MultiModalProjector(nn.Module):
    linear_1: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, image_features): ...

def pixel_shuffle(input_tensor, shuffle_ratio): ...

class Llama4VisionPixelShuffleMLP(nn.Module):
    pixel_shuffle_ratio: Incomplete
    inner_dim: Incomplete
    output_dim: Incomplete
    mlp: Incomplete
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None: ...
    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor: ...

class Llama4VisionAttention(nn.Module):
    config: Incomplete
    tp_size: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    num_local_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    attention_dropout: Incomplete
    scaling: Incomplete
    attn: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Llama4VisionEncoderLayer(nn.Module):
    hidden_size: Incomplete
    num_attention_heads: Incomplete
    intermediate_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor): ...

class Llama4VisionEncoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Llama4UnfoldConvolution(nn.Module):
    unfold: Incomplete
    linear: Incomplete
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Llama4VisionModel(nn.Module):
    config: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    hidden_size: Incomplete
    num_channels: Incomplete
    num_patches: Incomplete
    scale: Incomplete
    patch_embedding: Incomplete
    class_embedding: Incomplete
    positional_embedding_vlm: Incomplete
    layernorm_pre: Incomplete
    layernorm_post: Incomplete
    model: Incomplete
    vision_adapter: Incomplete
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, images_flattened: torch.Tensor) -> torch.Tensor: ...

class Mllama4ProcessingInfo(BaseProcessingInfo):
    def __init__(self, ctx: InputProcessingContext) -> None: ...
    def get_hf_config(self) -> Llama4Config: ...
    def get_hf_processor(self, **kwargs: object) -> Llama4Processor: ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    @staticmethod
    def get_patch_per_chunk(vision_config: Llama4VisionConfig) -> int: ...
    def get_max_num_tiles(self) -> int: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class Mllama4MultiModalProcessor(BaseMultiModalProcessor[Mllama4ProcessingInfo]): ...

class Mllama4DummyInputsBuilder(BaseDummyInputsBuilder[Mllama4ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Llama4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    supports_encoder_tp_data: bool
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    use_data_parallel: Incomplete
    vllm_config: Incomplete
    config: Incomplete
    quant_config: Incomplete
    multimodal_config: Incomplete
    vision_model: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    num_expert_groups: int
    num_logical_experts: Incomplete
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_routed_experts: Incomplete
    num_shared_experts: Incomplete
    num_redundant_experts: Incomplete
    moe_layers: Incomplete
    num_moe_layers: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    expert_weights: Incomplete
    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ): ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ): ...
    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def separate_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], prefix: str
    ) -> tuple[
        Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str, torch.Tensor]]
    ]: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...
