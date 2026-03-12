import numpy as np
import numpy.typing as npt
import torch.types
from .hasher import MultiModalHasher as MultiModalHasher
from .inputs import (
    BatchedTensorInputs as BatchedTensorInputs,
    MultiModalFieldElem as MultiModalFieldElem,
    MultiModalKwargsItem as MultiModalKwargsItem,
    MultiModalPlaceholderDict as MultiModalPlaceholderDict,
    MultiModalSharedField as MultiModalSharedField,
)
from .media import (
    AudioMediaIO as AudioMediaIO,
    ImageMediaIO as ImageMediaIO,
    MediaConnector as MediaConnector,
    VideoMediaIO as VideoMediaIO,
)
from PIL import Image as Image
from collections.abc import Generator, Sequence
from typing import Any
from vllm.utils.import_utils import LazyLoader as LazyLoader

def encode_audio_base64(
    audio: np.ndarray, sampling_rate: int, *, format: str = "WAV"
) -> str: ...
def encode_audio_url(
    audio: np.ndarray, sampling_rate: int, *, format: str = "WAV"
) -> str: ...
def encode_image_base64(
    image: Image.Image, *, image_mode: str = "RGB", format: str = "PNG"
) -> str: ...
def encode_image_url(
    image: Image.Image, *, image_mode: str = "RGB", format: str = "PNG"
) -> str: ...
def encode_video_base64(frames: npt.NDArray, *, format: str = "JPEG") -> str: ...
def encode_video_url(frames: npt.NDArray, *, format: str = "JPEG") -> str: ...
def argsort_mm_positions(
    mm_positions: MultiModalPlaceholderDict,
) -> list[tuple[str, int]]: ...
def group_and_batch_mm_items(
    items: Sequence[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[int, BatchedTensorInputs]]: ...
def group_and_batch_mm_kwargs(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]: ...
def group_mm_kwargs_by_modality(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]: ...
def fetch_audio(
    audio_url: str, audio_io_kwargs: dict[str, Any] | None = None
) -> tuple[np.ndarray, int | float]: ...
def fetch_image(
    image_url: str, image_io_kwargs: dict[str, Any] | None = None
) -> Image.Image: ...
def fetch_video(
    video_url: str, video_io_kwargs: dict[str, Any] | None = None
) -> tuple[npt.NDArray, dict[str, Any]]: ...
