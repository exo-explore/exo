import numpy as np
from _typeshed import Incomplete
from transformers import AutoProcessor as AutoProcessor
from transformers.audio_utils import AudioInput as AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import TextInput as TextInput

class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False): ...

class Qwen3ASRProcessor(ProcessorMixin):
    attributes: Incomplete
    feature_extractor_class: str
    tokenizer_class: Incomplete
    audio_token: Incomplete
    audio_bos_token: Incomplete
    audio_eos_token: Incomplete
    def __init__(
        self, feature_extractor=None, tokenizer=None, chat_template=None
    ) -> None: ...
    def __call__(
        self, text: TextInput = None, audio: AudioInput = None, **kwargs
    ) -> BatchFeature: ...
    def replace_multimodal_special_tokens(self, text, audio_lengths): ...
    def get_chunked_index(
        self, token_indices: np.ndarray, tokens_per_chunk: int
    ) -> list[tuple[int, int]]: ...
    def apply_chat_template(self, conversations, chat_template=None, **kwargs): ...
    @property
    def model_input_names(self): ...
