from _typeshed import Incomplete
from transformers import PreTrainedTokenizer as PreTrainedTokenizer
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.processing_utils import ProcessorMixin

class GLM4VImageProcessorFast(BaseImageProcessorFast):
    resample: Incomplete
    image_mean: Incomplete
    image_std: Incomplete
    size: Incomplete
    do_resize: bool
    do_rescale: bool
    do_normalize: bool

class GLM4VProcessor(ProcessorMixin):
    attributes: Incomplete
    tokenizer: Incomplete
    image_processor: Incomplete
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_size: int,
        image_processor: GLM4VImageProcessorFast | None = None,
    ) -> None: ...
