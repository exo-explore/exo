from _typeshed import Incomplete
from transformers.image_utils import ImageInput as ImageInput
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack as Unpack,
)
from transformers.tokenization_utils_base import (
    PreTokenizedInput as PreTokenizedInput,
    TextInput as TextInput,
)

class BagelProcessorKwargs(ProcessingKwargs, total=False): ...

class BagelProcessor(ProcessorMixin):
    attributes: Incomplete
    image_processor_class: str
    tokenizer_class: str
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        images: ImageInput = None,
        **kwargs: Unpack[BagelProcessorKwargs],
    ): ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self): ...
