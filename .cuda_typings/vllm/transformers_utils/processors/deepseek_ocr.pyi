from PIL import Image as Image
from _typeshed import Incomplete
from transformers import LlamaTokenizerFast as LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin
from typing import Literal

BASE_SIZE: int
IMAGE_SIZE: int
CROP_MODE: bool
MIN_CROPS: int
MAX_CROPS: int

def find_closest_aspect_ratio(
    aspect_ratio, target_ratios, width, height, image_size
): ...
def calculate_aspect_ratios(
    min_num: int = ..., max_num: int = ...
) -> list[tuple[int, int]]: ...
def count_tiles(
    orig_width,
    orig_height,
    min_num=...,
    max_num=...,
    image_size: int = 640,
    use_thumbnail: bool = False,
): ...
def dynamic_preprocess(
    image, min_num=..., max_num=..., image_size: int = 640, use_thumbnail: bool = False
): ...

class ImageTransform:
    mean: Incomplete
    std: Incomplete
    normalize: Incomplete
    transform: Incomplete
    def __init__(
        self,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ) -> None: ...
    def __call__(self, pil_img: Image.Image): ...

class DeepseekOCRProcessor(ProcessorMixin):
    tokenizer_class: Incomplete
    attributes: Incomplete
    image_size: Incomplete
    base_size: Incomplete
    strategy: Incomplete
    patch_size: int
    image_mean: Incomplete
    image_std: Incomplete
    normalize: Incomplete
    downsample_ratio: int
    image_transform: Incomplete
    tokenizer: Incomplete
    image_token_id: Incomplete
    image_token: Incomplete
    pad_token: Incomplete
    add_special_token: Incomplete
    sft_format: Incomplete
    mask_prompt: Incomplete
    ignore_id: Incomplete
    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        image_size: int = ...,
        base_size: int = ...,
        strategy: Literal["v1", "v2"] = "v1",
        **kwargs,
    ) -> None: ...
    @property
    def bos_id(self): ...
    @property
    def eos_id(self): ...
    @property
    def pad_id(self): ...
    def encode(self, text: str, bos: bool = True, eos: bool = False): ...
    def decode(self, t: list[int], **kwargs) -> str: ...
    def process_one(
        self, prompt: str, images: list[Image.Image], crop_mode: bool = ...
    ): ...
    def __call__(
        self, *, prompt: str, images: list[Image.Image], crop_mode: bool = ..., **kwargs
    ): ...
    def tokenize_with_images(
        self,
        conversation: str,
        images: list[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ): ...
