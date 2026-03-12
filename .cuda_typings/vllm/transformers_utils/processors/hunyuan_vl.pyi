import torch
from _typeshed import Incomplete
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput as ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PreTokenizedInput as PreTokenizedInput,
    TextInput as TextInput,
)
from transformers.video_utils import VideoInput as VideoInput

class HunYuanVLProcessor(ProcessorMixin):
    attributes: Incomplete
    valid_kwargs: Incomplete
    image_processor_class: str
    tokenizer_class: str
    tokenizer: Incomplete
    image_token_id: int
    image_token: Incomplete
    im_start_token_id: int
    im_start_token: Incomplete
    im_end_token_id: int
    im_end_token: Incomplete
    placeholder_token: Incomplete
    pad_id: int
    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        **kwargs,
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ) -> None: ...
    def apply_chat_template(self, *args, **kwargs): ...
    def get_imgs_pos(self, doc_ids): ...
    @property
    def model_input_names(self): ...

def split_image_into_patch_blocks(
    pixel_values: torch.Tensor, patch_size: int = 16, adaptor_patch_div: int = 4
) -> torch.Tensor: ...
