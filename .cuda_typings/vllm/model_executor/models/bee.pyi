import abc
import torch
import torch.nn as nn
from .llava_next import (
    LlavaDummyInputsBuilder as LlavaDummyInputsBuilder,
    LlavaNextMultiModalProcessor as LlavaNextMultiModalProcessor,
    LlavaNextProcessingInfo as LlavaNextProcessingInfo,
)
from .llava_onevision import (
    LlavaOnevisionForConditionalGeneration as LlavaOnevisionForConditionalGeneration,
)
from .utils import WeightsMapper as WeightsMapper
from _typeshed import Incomplete
from collections.abc import Mapping
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict as MultiModalDataDict

class BeeProcessingInfo(LlavaNextProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...

class BeeDummyInputsBuilder(LlavaDummyInputsBuilder[BeeProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class BeeMultiModalProjector(nn.Module):
    pre_norm: Incomplete
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, image_feature: torch.Tensor) -> torch.Tensor: ...

class BeeForConditionalGeneration(
    LlavaOnevisionForConditionalGeneration, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    multi_modal_projector: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
