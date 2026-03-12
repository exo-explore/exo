import numpy as np
from _typeshed import Incomplete
from transformers import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType as TensorType
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import LazyLoader as LazyLoader

logger: Incomplete

class CMVN:
    def __init__(self, dim, means, inverse_std_variences) -> None: ...
    def __call__(self, x): ...

class KaldifeatFbank:
    dither: Incomplete
    opts: Incomplete
    def __init__(
        self,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 1.0,
    ) -> None: ...
    def __call__(self, sample_rate, wav_np, is_train: bool = False): ...

class FireRedASR2FeatureExtractor(SequenceFeatureExtractor):
    model_input_names: Incomplete
    chunk_length: Incomplete
    max_length: Incomplete
    dim: Incomplete
    means: Incomplete
    inverse_std_variences: Incomplete
    num_mel_bins: Incomplete
    frame_length: Incomplete
    frame_shift: Incomplete
    dither: Incomplete
    sampling_rate: Incomplete
    downsample_rate: Incomplete
    context: Incomplete
    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        chunk_length: int = 30,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
        dim: int = 80,
        means=None,
        inverse_std_variences=None,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        max_length: int = 3000,
        downsample_rate: int = 2,
        left_context: int = 3,
        right_context: int = 3,
        **kwargs,
    ) -> None: ...
    cmvn: Incomplete
    fbank: Incomplete
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = True,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "max_length",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        do_normalize: bool | None = None,
        **kwargs,
    ) -> BatchFeature: ...

class FireRedASR2Processor(ProcessorMixin):
    feature_extractor_class: str
    tokenizer_class: Incomplete
    current_processor: Incomplete
    audio_token: Incomplete
    audio_token_id: Incomplete
    def __init__(
        self, feature_extractor, tokenizer, audio_token: str = "<|AUDIO|>"
    ) -> None: ...
    def get_decoder_prompt_ids(
        self, task=None, language=None, no_timestamps: bool = True
    ): ...
    def __call__(self, *args, **kwargs): ...
    def get_prompt_ids(self, text: str, return_tensors: str = "np"): ...
