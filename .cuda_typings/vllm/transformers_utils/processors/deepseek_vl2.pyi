from PIL import Image as Image
from _typeshed import Incomplete
from transformers import LlamaTokenizerFast as LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin
from typing import Any

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

class DeepseekVLV2Processor(ProcessorMixin):
    tokenizer_class: Incomplete
    attributes: Incomplete
    candidate_resolutions: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    image_mean: Incomplete
    image_std: Incomplete
    normalize: Incomplete
    downsample_ratio: Incomplete
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
        candidate_resolutions: tuple[tuple[int, int]],
        patch_size: int,
        downsample_ratio: int,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ) -> None: ...
    def select_best_resolution(self, image_size): ...
    @property
    def bos_id(self): ...
    @property
    def eos_id(self): ...
    @property
    def pad_id(self): ...
    def encode(self, text: str, bos: bool = True, eos: bool = False): ...
    def decode(self, t: list[int], **kwargs) -> str: ...
    def process_one(
        self,
        prompt: str,
        images: list[Image.Image],
        inference_mode: bool = True,
        **kwargs: Any,
    ): ...
    def __call__(
        self,
        *,
        text: str,
        images: list[Image.Image],
        inference_mode: bool = True,
        **kwargs: Any,
    ): ...
    def tokenize_with_images(
        self,
        conversation: str,
        images: list[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ): ...
