import abc
import torch
from .interfaces import SupportsLateInteraction as SupportsLateInteraction
from .interfaces_base import default_pooling_type as default_pooling_type
from .qwen2_vl import Qwen2VLMultiModalDataParser as Qwen2VLMultiModalDataParser
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder as Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration as Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor as Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo as Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers.models.qwen3_vl import Qwen3VLProcessor
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.pooler.tokwise import (
    pooler_for_token_embed as pooler_for_token_embed,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY

class ColQwen3ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor: ...
    def get_video_processor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...
    def get_data_parser(self): ...

class ColQwen3Model(
    Qwen3VLForConditionalGeneration, SupportsLateInteraction, metaclass=abc.ABCMeta
):
    is_pooling_model: bool
    hf_to_vllm_mapper: Incomplete
    embed_dim: int | None
    custom_text_proj: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
