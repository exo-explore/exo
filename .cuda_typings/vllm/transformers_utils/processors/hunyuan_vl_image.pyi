from _typeshed import Incomplete
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import (
    ChannelDimension,
    ImageInput as ImageInput,
    PILImageResampling,
)
from transformers.utils import TensorType as TensorType
from transformers.video_utils import VideoInput as VideoInput

logger: Incomplete

def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = ...,
    max_pixels: int = ...,
): ...

class HunYuanVLImageProcessor(BaseImageProcessor):
    model_input_names: Incomplete
    min_pixels: Incomplete
    max_pixels: Incomplete
    size: Incomplete
    do_resize: Incomplete
    resample: Incomplete
    do_rescale: Incomplete
    rescale_factor: Incomplete
    do_normalize: Incomplete
    image_mean: Incomplete
    image_std: Incomplete
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    merge_size: Incomplete
    do_convert_rgb: Incomplete
    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = ...,
        do_rescale: bool = True,
        rescale_factor: int | float = ...,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        resample: PILImageResampling = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        do_convert_rgb: bool | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = None,
    ): ...
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=None
    ): ...
