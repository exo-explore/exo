import torch
from .base import get_vllm_public_assets as get_vllm_public_assets
from PIL import Image
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path

VLM_IMAGES_DIR: str
ImageAssetName: Incomplete

@dataclass(frozen=True)
class ImageAsset:
    name: ImageAssetName
    def get_path(self, ext: str) -> Path: ...
    @property
    def pil_image(self) -> Image.Image: ...
    def pil_image_ext(self, ext: str) -> Image.Image: ...
    @property
    def image_embeds(self) -> torch.Tensor: ...
    def read_bytes(self, ext: str) -> bytes: ...
