import abc
from .audioflamingo3 import (
    AudioFlamingo3DummyInputsBuilder as AudioFlamingo3DummyInputsBuilder,
    AudioFlamingo3ForConditionalGeneration as AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProcessor as AudioFlamingo3MultiModalProcessor,
)
from collections.abc import Mapping
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo as BaseProcessingInfo

class MusicFlamingoProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_feature_extractor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class MusicFlamingoDummyInputsBuilder(AudioFlamingo3DummyInputsBuilder): ...
class MusicFlamingoForConditionalGeneration(
    AudioFlamingo3ForConditionalGeneration, metaclass=abc.ABCMeta
): ...
