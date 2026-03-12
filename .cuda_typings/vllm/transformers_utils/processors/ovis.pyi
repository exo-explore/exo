import PIL
from _typeshed import Incomplete
from functools import cached_property
from transformers import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

__all__ = ["OvisProcessor"]

class OvisProcessorKwargs(ProcessingKwargs, total=False): ...

class OvisProcessor(ProcessorMixin):
    attributes: Incomplete
    valid_kwargs: Incomplete
    image_processor_class: str
    tokenizer_class: str
    image_token: str
    image_pad_token: Incomplete
    image_segment_len: Incomplete
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_pad_token=None,
        image_segment_len: int = 255,
        **kwargs,
    ) -> None: ...
    @cached_property
    def extra_special_tokens(self): ...
    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        **kwargs: Unpack[OvisProcessorKwargs],
    ) -> BatchFeature: ...
    def get_image_size(self): ...
    def get_token_value(self, tok): ...
    def construct_image_indicators(self, grid): ...
    def construct_image_placeholders(self, grid): ...
    def preprocess_image(
        self,
        image: PIL.Image.Image,
        max_partition,
        covering_threshold,
        do_convert_rgb,
        return_tensors,
    ): ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    def post_process_image_text_to_text(self, generated_outputs): ...
    @property
    def model_input_names(self): ...
