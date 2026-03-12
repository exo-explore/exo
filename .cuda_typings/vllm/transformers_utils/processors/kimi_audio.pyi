from _typeshed import Incomplete
from collections.abc import Mapping as Mapping
from transformers import (
    AutoFeatureExtractor as AutoFeatureExtractor,
    BatchFeature,
    ProcessorMixin,
)
from transformers.audio_utils import AudioInput as AudioInput
from transformers.tokenization_utils_base import TextInput as TextInput
from typing import Any
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer as KimiAudioTokenizer

class KimiAudioProcessor(ProcessorMixin):
    attributes: Incomplete
    feature_extractor_class: str
    tokenizer_class: str
    KIMIA_MEDIA_BEGIN: int
    KIMIA_MEDIA_END: int
    KIMIA_TEXT_BLANK: int
    AUDIO_SEQ_LEN: int
    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs) -> None: ...
    def check_argument_for_proper_class(self, attribute_name: str, argument: Any): ...
    def __call__(
        self,
        text: TextInput = None,
        audio: AudioInput = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature: ...
