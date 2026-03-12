import librosa
import numpy as np
import numpy.typing as npt
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from enum import Enum
from typing import Literal
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

class ChannelReduction(str, Enum):
    MEAN = "mean"
    FIRST = "first"
    MAX = "max"
    SUM = "sum"

@dataclass
class AudioSpec:
    target_channels: int | None = ...
    channel_reduction: ChannelReduction = ...
    @property
    def needs_normalization(self) -> bool: ...

MONO_AUDIO_SPEC: Incomplete
PASSTHROUGH_AUDIO_SPEC: Incomplete

def normalize_audio(
    audio: npt.NDArray[np.floating] | torch.Tensor, spec: AudioSpec
) -> npt.NDArray[np.floating] | torch.Tensor: ...
def resample_audio_librosa(
    audio: npt.NDArray[np.floating], *, orig_sr: float, target_sr: float
) -> npt.NDArray[np.floating]: ...
def resample_audio_scipy(
    audio: npt.NDArray[np.floating], *, orig_sr: float, target_sr: float
): ...

class AudioResampler:
    target_sr: Incomplete
    method: Incomplete
    def __init__(
        self,
        target_sr: float | None = None,
        method: Literal["librosa", "scipy"] = "librosa",
    ) -> None: ...
    def resample(
        self, audio: npt.NDArray[np.floating], *, orig_sr: float
    ) -> npt.NDArray[np.floating]: ...

def split_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    max_clip_duration_s: float,
    overlap_duration_s: float,
    min_energy_window_size: int,
) -> list[np.ndarray]: ...
def find_split_point(
    wav: np.ndarray, start_idx: int, end_idx: int, min_energy_window: int
) -> int: ...
