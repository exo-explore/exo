import numpy.typing as npt
from .base import (
    VLLM_S3_BUCKET_URL as VLLM_S3_BUCKET_URL,
    get_vllm_public_assets as get_vllm_public_assets,
)
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

ASSET_DIR: str
AudioAssetName: Incomplete

@dataclass(frozen=True)
class AudioAsset:
    name: AudioAssetName
    @property
    def filename(self) -> str: ...
    @property
    def audio_and_sample_rate(self) -> tuple[npt.NDArray, float]: ...
    def get_local_path(self) -> Path: ...
    @property
    def url(self) -> str: ...
