import numpy as np
import torch
import torch.nn as nn
from _typeshed import Incomplete
from transformers import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType as TensorType
from vllm.logger import init_logger as init_logger

logger: Incomplete

def apply_cmvn(inputs, cmvn): ...
def apply_lfr(inputs, lfr_m, lfr_n): ...
def load_cmvn(cmvn_file): ...

class WavFrontend(nn.Module):
    fs: Incomplete
    window: Incomplete
    n_mels: Incomplete
    frame_length: Incomplete
    frame_shift: Incomplete
    filter_length_min: Incomplete
    filter_length_max: Incomplete
    lfr_m: Incomplete
    lfr_n: Incomplete
    cmvn_file: Incomplete
    dither: Incomplete
    snip_edges: Incomplete
    upsacle_samples: Incomplete
    cmvn: Incomplete
    def __init__(
        self,
        cmvn_file: str = "null",
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: int = -1,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        snip_edges: bool = True,
        upsacle_samples: bool = True,
        **kwargs,
    ) -> None: ...
    def output_size(self) -> int: ...
    def forward(
        self, input: torch.Tensor, input_lengths, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_fbank(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_lfr_cmvn(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class FunASRFeatureExtractor(SequenceFeatureExtractor):
    model_input_names: Incomplete
    frontend_conf: Incomplete
    max_length: Incomplete
    n_fft: Incomplete
    hop_length: Incomplete
    chunk_length: Incomplete
    n_samples: Incomplete
    nb_max_frames: Incomplete
    sampling_rate: Incomplete
    dither: Incomplete
    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        dither: float = 0.0,
        max_length: int = 1000,
        return_attention_mask: bool = False,
        **kwargs,
    ) -> None: ...
    def extract_fbank(
        self, data, data_len=None, data_type: str = "sound", frontend=None, **kwargs
    ): ...
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
        device: str | None = "cpu",
        return_token_timestamps: bool | None = None,
        **kwargs,
    ) -> BatchFeature: ...

class FunASRProcessor(ProcessorMixin):
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
