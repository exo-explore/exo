from _typeshed import Incomplete
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.processing_utils import ProcessorMixin
from vllm.tokenizers.qwen_vl import QwenVLTokenizer as QwenVLTokenizer

class QwenVLImageProcessorFast(BaseImageProcessorFast):
    resample: Incomplete
    image_mean: Incomplete
    image_std: Incomplete
    size: Incomplete
    do_resize: bool
    do_rescale: bool
    do_normalize: bool

class QwenVLProcessor(ProcessorMixin):
    attributes: Incomplete
    tokenizer: Incomplete
    image_processor: Incomplete
    def __init__(
        self,
        tokenizer: QwenVLTokenizer,
        image_size: int,
        image_processor: QwenVLImageProcessorFast | None = None,
    ) -> None: ...
    @property
    def image_start_tag(self) -> str: ...
    @property
    def image_end_tag(self) -> str: ...
    @property
    def image_pad_tag(self) -> str: ...
