from pathlib import Path
from typing import Literal, Optional, Protocol, runtime_checkable

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
    ) -> Optional[Image.Image]:
        """Generate an image from a text prompt, or edit an existing image.

        For distributed inference, only the first stage (rank 0) returns the image.
        Other stages return None after participating in the pipeline.

        Args:
            prompt: Text description of the image to generate
            height: Image height in pixels
            width: Image width in pixels
            quality: Generation quality level
            seed: Random seed for reproducibility
            image_path: Optional path to input image for img2img
            image_strength: Optional strength for img2img (0.0-1.0, higher = more change)

        Returns:
            Generated PIL Image (rank 0) or None (other ranks)
        """
        ...
