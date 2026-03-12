import numpy as np
import numpy.typing as npt
import torch
from .audio import (
    AudioEmbeddingMediaIO as AudioEmbeddingMediaIO,
    AudioMediaIO as AudioMediaIO,
)
from .base import MediaIO as MediaIO
from .image import (
    ImageEmbeddingMediaIO as ImageEmbeddingMediaIO,
    ImageMediaIO as ImageMediaIO,
)
from .video import VideoMediaIO as VideoMediaIO
from PIL import Image as Image
from _typeshed import Incomplete
from typing import Any
from urllib3.util import Url as Url
from vllm.connections import (
    HTTPConnection as HTTPConnection,
    global_http_connection as global_http_connection,
)
from vllm.utils.registry import ExtensionManager as ExtensionManager

global_thread_pool: Incomplete
MEDIA_CONNECTOR_REGISTRY: Incomplete
MODALITY_IO_MAP: dict[str, type[MediaIO]]

def merge_media_io_kwargs(
    defaults: dict[str, dict[str, Any]] | None,
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None: ...

class MediaConnector:
    media_io_kwargs: dict[str, dict[str, Any]]
    connection: Incomplete
    allowed_local_media_path: Incomplete
    allowed_media_domains: Incomplete
    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        connection: HTTPConnection = ...,
        *,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> None: ...
    def load_from_url(
        self, url: str, media_io: MediaIO[_M], *, fetch_timeout: int | None = None
    ) -> _M: ...
    async def load_from_url_async(
        self, url: str, media_io: MediaIO[_M], *, fetch_timeout: int | None = None
    ) -> _M: ...
    def fetch_audio(self, audio_url: str) -> tuple[np.ndarray, int | float]: ...
    async def fetch_audio_async(
        self, audio_url: str
    ) -> tuple[np.ndarray, int | float]: ...
    def fetch_image(
        self, image_url: str, *, image_mode: str = "RGB"
    ) -> Image.Image: ...
    async def fetch_image_async(
        self, image_url: str, *, image_mode: str = "RGB"
    ) -> Image.Image: ...
    def fetch_video(
        self, video_url: str, *, image_mode: str = "RGB"
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...
    async def fetch_video_async(
        self, video_url: str, *, image_mode: str = "RGB"
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...
    def fetch_image_embedding(self, data: str) -> torch.Tensor: ...
    def fetch_audio_embedding(self, data: str) -> torch.Tensor: ...
