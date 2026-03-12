import abc
import cv2
import numpy.typing as npt
from _typeshed import Incomplete
from abc import abstractmethod
from typing import Any, NamedTuple
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule
from vllm.utils.registry import ExtensionManager as ExtensionManager

logger: Incomplete

def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray: ...
def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray: ...
def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray: ...

class VideoTargetMetadata(NamedTuple):
    num_frames: int
    fps: float
    max_duration: float

class VideoSourceMetadata(NamedTuple):
    total_frames_num: int
    original_fps: float
    duration: float

class VideoLoader(metaclass=abc.ABCMeta):
    @classmethod
    def compute_frames_index_to_sample(
        cls, source: VideoSourceMetadata, target: VideoTargetMetadata, **kwargs
    ) -> list[int]: ...
    @classmethod
    @abstractmethod
    def load_bytes(
        cls, data: bytes, **kwargs
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...
    @classmethod
    def create_hf_metadata(
        cls,
        source: VideoSourceMetadata,
        valid_frame_indices: list[int],
        video_backend: str,
    ): ...

VIDEO_LOADER_REGISTRY: Incomplete

class OpenCVVideoBackendMixin:
    @staticmethod
    def get_cv2_video_api(): ...
    @classmethod
    def open_video_capture(cls, data: bytes) -> cv2.VideoCapture: ...
    @staticmethod
    def get_video_metadata(cap: cv2.VideoCapture) -> VideoSourceMetadata: ...
    @classmethod
    def read_frames(
        cls,
        cap: cv2.VideoCapture,
        frame_idx: list[int],
        total_frames_num: int,
        *,
        frame_recovery: bool = False,
    ) -> tuple[npt.NDArray, list[int]]: ...

class OpenCVVideoBackend(VideoLoader, OpenCVVideoBackendMixin):
    @classmethod
    def compute_frames_index_to_sample(
        cls, source: VideoSourceMetadata, target: VideoTargetMetadata, **kwargs
    ) -> list[int]: ...
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...

class OpenCVDynamicVideoBackend(VideoLoader, OpenCVVideoBackendMixin):
    @classmethod
    def compute_frames_index_to_sample(
        cls, source: VideoSourceMetadata, target: VideoTargetMetadata, **kwargs
    ) -> list[int]: ...
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...

class Molmo2VideoBackend(VideoLoader, OpenCVVideoBackendMixin):
    @classmethod
    def get_candidate_target_fps(
        cls, video_fps: float, sampling_fps: float, max_fps: float = 8.0
    ) -> list[float]: ...
    @classmethod
    def get_target_fps(
        cls,
        video_fps: float,
        max_frames: int,
        total_frames: int,
        frame_sample_mode: str,
        candidate_target_fps: list[float],
    ) -> float | None: ...
    @classmethod
    def get_frame_times_and_chosen_fps(
        cls,
        selected_target_fps: float | None,
        total_frames: int,
        max_frames: int,
        video_fps: float,
    ) -> tuple[float | None, npt.NDArray]: ...
    @classmethod
    def sample_times(
        cls,
        duration: float,
        max_frames: int,
        frame_sample_mode: str,
        max_fps: int | None,
        candidate_target_fps: list[float] | None = None,
        **kwargs,
    ) -> npt.NDArray: ...
    @classmethod
    def compute_frames_index_to_sample(
        cls, source: VideoSourceMetadata, target: VideoTargetMetadata, **kwargs
    ): ...
    @classmethod
    def load_bytes_opencv(
        cls,
        data: bytes,
        frame_sample_mode: str | None = None,
        num_frames: int = -1,
        max_fps: int = 2,
        sampling_fps: int = 2,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...
    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...

class NemotronVLVideoBackend(OpenCVVideoBackend):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...

class OpenCVDynamicOpenPanguVideoBackend(VideoLoader, OpenCVVideoBackendMixin):
    @classmethod
    def compute_frames_index_to_sample(
        cls, source: VideoSourceMetadata, target: VideoTargetMetadata, **kwargs
    ) -> list[int]: ...
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...
