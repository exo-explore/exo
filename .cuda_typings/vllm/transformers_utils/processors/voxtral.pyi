from _typeshed import Incomplete
from mistral_common.tokens.tokenizers.audio import AudioEncoder as AudioEncoder
from transformers import BatchFeature, ProcessorMixin, TensorType as TensorType
from transformers.audio_utils import AudioInput as AudioInput
from transformers.image_utils import ImageInput as ImageInput
from transformers.tokenization_utils_base import (
    PreTokenizedInput as PreTokenizedInput,
    TextInput as TextInput,
)
from transformers.video_utils import VideoInput as VideoInput
from vllm.tokenizers.mistral import MistralTokenizer as MistralTokenizer

class MistralCommonFeatureExtractor:
    audio_encoder: Incomplete
    def __init__(self, audio_encoder: AudioEncoder) -> None: ...
    @property
    def sampling_rate(self): ...
    @property
    def frame_rate(self): ...
    def __call__(
        self,
        audios: AudioInput,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature: ...
    def get_num_audio_tokens(self, audio_length: int) -> int: ...

class MistralCommonVoxtralProcessor(ProcessorMixin):
    attributes: Incomplete
    tokenizer: Incomplete
    feature_extractor: Incomplete
    def __init__(self, tokenizer: MistralTokenizer) -> None: ...
    @property
    def audio_token_id(self) -> int: ...
    @property
    def begin_audio_token_id(self) -> int: ...
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs,
    ): ...
