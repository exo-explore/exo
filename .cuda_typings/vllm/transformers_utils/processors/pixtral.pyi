from _typeshed import Incomplete
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder as ImageEncoder
from transformers import BatchFeature, ProcessorMixin, TensorType as TensorType
from transformers.audio_utils import AudioInput as AudioInput
from transformers.image_utils import ImageInput as ImageInput
from transformers.tokenization_utils_base import (
    PreTokenizedInput as PreTokenizedInput,
    TextInput as TextInput,
)
from transformers.video_utils import VideoInput as VideoInput
from vllm.tokenizers.mistral import MistralTokenizer as MistralTokenizer

class MistralCommonImageProcessor:
    mm_encoder: Incomplete
    def __init__(self, mm_encoder: ImageEncoder) -> None: ...
    def __call__(
        self,
        images: ImageInput,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature: ...
    def get_number_of_image_patches(
        self, height: int, width: int
    ) -> tuple[int, int, int]: ...

class MistralCommonPixtralProcessor(ProcessorMixin):
    attributes: Incomplete
    tokenizer: Incomplete
    image_processor: Incomplete
    def __init__(self, tokenizer: MistralTokenizer) -> None: ...
    @property
    def image_break_id(self) -> int: ...
    @property
    def image_token_id(self) -> int: ...
    @property
    def image_end_id(self) -> int: ...
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
