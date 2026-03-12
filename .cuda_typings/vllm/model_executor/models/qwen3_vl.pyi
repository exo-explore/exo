import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsMultiModalPruning as SupportsMultiModalPruning,
    SupportsPP as SupportsPP,
)
from .qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs as Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs as Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs as Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs as Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs as Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs as Qwen2_5_VLVideoPixelInputs,
    Qwen2_5_VisionAttention as Qwen2_5_VisionAttention,
)
from .qwen2_vl import (
    Qwen2VLMultiModalDataParser as Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo as Qwen2VLProcessingInfo,
)
from .qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLM, Qwen3Model as Qwen3Model
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    get_vit_attn_backend as get_vit_attn_backend,
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model as run_dp_sharded_mrope_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from transformers.models.qwen2_vl import (
    Qwen2VLImageProcessorFast as Qwen2VLImageProcessorFast,
)
from transformers.models.qwen3_vl import Qwen3VLProcessor, Qwen3VLVideoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLVisionConfig as Qwen3VLVisionConfig,
)
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import (
    BaseDummyOptions as BaseDummyOptions,
    VideoDummyOptions as VideoDummyOptions,
)
from vllm.distributed import (
    get_pp_group as get_pp_group,
    parallel_state as parallel_state,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer as Conv3dLayer
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
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.evs import (
    compute_mrope_for_media as compute_mrope_for_media,
    compute_retained_tokens_count as compute_retained_tokens_count,
    compute_retention_mask as compute_retention_mask,
    recompute_mrope_positions as recompute_mrope_positions,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalFieldElem as MultiModalFieldElem,
    MultiModalKwargsItem as MultiModalKwargsItem,
    MultiModalKwargsItems as MultiModalKwargsItems,
    PlaceholderRange as PlaceholderRange,
    VideoItem as VideoItem,
)
from vllm.multimodal.parse import (
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers.protocol import TokenizerLike as TokenizerLike
from vllm.tokenizers.registry import (
    cached_tokenizer_from_config as cached_tokenizer_from_config,
)
from vllm.utils.collection_utils import is_list_of as is_list_of
from vllm.utils.math_utils import round_up as round_up

logger: Incomplete
DUMMY_VIDEO_NUM_FRAMES: int

class Qwen3_VisionPatchEmbed(nn.Module):
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    hidden_size: Incomplete
    proj: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen3_VisionMLP(nn.Module):
    linear_fc1: Incomplete
    linear_fc2: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor): ...

class Qwen3_VisionBlock(nn.Module):
    norm1: Incomplete
    norm2: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor: ...

class Qwen3_VisionPatchMerger(nn.Module):
    hidden_size: Incomplete
    use_postshuffle_norm: Incomplete
    norm: Incomplete
    linear_fc1: Incomplete
    act_fn: Incomplete
    linear_fc2: Incomplete
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen3_VisionTransformer(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    num_position_embeddings: Incomplete
    patch_size: Incomplete
    spatial_merge_size: Incomplete
    spatial_merge_unit: Incomplete
    temporal_patch_size: Incomplete
    deepstack_visual_indexes: Incomplete
    num_grid_per_side: Incomplete
    tp_size: Incomplete
    out_hidden_size: Incomplete
    patch_embed: Incomplete
    pos_embed: Incomplete
    rotary_pos_emb: Incomplete
    merger: Incomplete
    deepstack_merger_list: Incomplete
    attn_backend: Incomplete
    blocks: Incomplete
    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-06,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    @staticmethod
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor: ...
    def rot_pos_emb(self, grid_thw: list[list[int]]): ...
    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor: ...
    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor | list[list[int]]
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor: ...
    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessorFast: ...
    def get_video_processor(self, **kwargs: object) -> Qwen3VLVideoProcessor: ...
    def get_data_parser(self): ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...
    def get_max_video_tokens(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class Qwen3VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor[Qwen3VLProcessingInfo]):
    @staticmethod
    def get_video_repl(
        *,
        tokens_per_frame: list[int],
        timestamps: list[float | int],
        tokenizer: TokenizerLike,
        vision_start_token_id: int,
        vision_end_token_id: int,
        video_token_id: int,
        select_token_id: bool = False,
    ) -> PromptUpdateDetails[list[int]]: ...

class Qwen3LLMModel(Qwen3Model):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...

class Qwen3LLMForCausalLM(Qwen3ForCausalLM, metaclass=abc.ABCMeta):
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class Qwen3VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    SupportsEagle3,
    SupportsMultiModalPruning,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    supports_encoder_tp_data: bool
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    video_pruning_rate: Incomplete
    is_multimodal_pruning_enabled: Incomplete
    use_deepstack: Incomplete
    deepstack_num_level: Incomplete
    visual_dim: Incomplete
    multiscale_dim: Incomplete
    visual: Incomplete
    deepstack_input_embeds: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[MultiModalEmbeddings, torch.Tensor, int]: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
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
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...
