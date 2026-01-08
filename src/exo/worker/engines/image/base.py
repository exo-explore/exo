from collections.abc import Generator
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class ImageGenerator(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def is_first_stage(self) -> bool: ...

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        quality: Literal["low", "medium", "high"],
        seed: int,
        image_path: Path | None = None,
        image_strength: float | None = None,
        partial_images: int = 0,
    ) -> Generator[Image.Image | tuple[Image.Image, int, int], None, None]:
        """Generate an image from a text prompt, or edit an existing image.

        For distributed inference, only the last stage returns images.
        Other stages yield nothing after participating in the pipeline.

        When partial_images > 0, yields intermediate images during diffusion
        as tuples of (image, partial_index, total_partials), then yields
        the final image.

        When partial_images = 0 (default), only yields the final image.

        Args:
            prompt: Text description of the image to generate
            height: Image height in pixels
            width: Image width in pixels
            quality: Generation quality level
            seed: Random seed for reproducibility
            image_path: Optional path to input image for img2img
            image_strength: Optional strength for img2img (0.0-1.0, higher = more change)
            partial_images: Number of intermediate images to yield (0 for none)

        Yields:
            Intermediate images as (Image, partial_index, total_partials) tuples
            Final PIL Image (last stage) or nothing (other stages)
        """
        ...
