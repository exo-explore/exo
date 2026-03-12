import PIL
import numpy as np
from _typeshed import Incomplete
from functools import cached_property
from transformers import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

__all__ = ["Ovis2_5Processor"]

class Ovis2_5ProcessorKwargs(ProcessingKwargs, total=False): ...

class Ovis2_5Processor(ProcessorMixin):
    attributes: Incomplete
    valid_kwargs: Incomplete
    image_processor_class: str
    tokenizer_class: str
    image_token: Incomplete
    video_token: Incomplete
    image_pad_token: str
    patch_size: Incomplete
    hidden_stride: Incomplete
    temporal_patch_size: Incomplete
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_pad_token=None,
        patch_size: int = 16,
        hidden_stride: int = 2,
        temporal_patch_size: int = 1,
        **kwargs,
    ) -> None: ...
    @cached_property
    def extra_special_tokens(self): ...
    def __call__(
        self,
        images: ImageInput = None,
        videos: np.ndarray | list[ImageInput] = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        **kwargs: Unpack[Ovis2_5ProcessorKwargs],
    ) -> BatchFeature: ...
    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = ...,
        max_pixels: int = ...,
    ): ...
    def get_token_value(self, tok): ...
    def construct_visual_indicators(self, grid, is_video: bool = False): ...
    def construct_visual_placeholders(self, grid, is_video: bool = False): ...
    def preprocess_multidata(
        self,
        images: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        video: list[PIL.Image.Image] | np.ndarray | None = None,
        do_convert_rgb: bool | None = True,
        min_pixels: int = ...,
        max_pixels: int = ...,
        return_tensors: str | None = "pt",
    ): ...
