import numpy.typing as npt
from ..video import VIDEO_LOADER_REGISTRY as VIDEO_LOADER_REGISTRY
from .base import MediaIO as MediaIO
from .image import ImageMediaIO as ImageMediaIO
from _typeshed import Incomplete
from pathlib import Path
from typing import Any
from vllm import envs as envs

class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):
    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]: ...
    image_io: Incomplete
    num_frames: Incomplete
    kwargs: Incomplete
    video_loader: Incomplete
    def __init__(
        self, image_io: ImageMediaIO, num_frames: int = 32, **kwargs
    ) -> None: ...
    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]: ...
    def load_base64(
        self, media_type: str, data: str
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...
    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]: ...
    def encode_base64(
        self, media: npt.NDArray, *, video_format: str = "JPEG"
    ) -> str: ...
