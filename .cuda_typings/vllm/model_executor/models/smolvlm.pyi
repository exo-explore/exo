import abc
from .idefics3 import (
    Idefics3ForConditionalGeneration as Idefics3ForConditionalGeneration,
    Idefics3ProcessingInfo as Idefics3ProcessingInfo,
)
from transformers import SmolVLMProcessor
from vllm.config import VllmConfig as VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY

class SmolVLMProcessingInfo(Idefics3ProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> SmolVLMProcessor: ...

class SmolVLMForConditionalGeneration(
    Idefics3ForConditionalGeneration, metaclass=abc.ABCMeta
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
