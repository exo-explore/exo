import base64
import io
import tempfile
from pathlib import Path
from typing import Generator, Literal

from PIL import Image

from exo.shared.types.api import ImageEditsInternalParams, ImageGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    ImageGenerationResponse,
    PartialImageResponse,
)
from exo.worker.engines.image.base import ImageGenerator


def parse_size(size_str: str | None) -> tuple[int, int]:
    """Parse size parameter like '1024x1024' to (width, height) tuple."""
    if not size_str or size_str == "auto":
        size_str = "1024x1024"

    try:
        parts = size_str.split("x")
        if len(parts) == 2:
            width, height = int(parts[0]), int(parts[1])
            return (width, height)
    except (ValueError, AttributeError):
        pass

    # Default fallback
    return (1024, 1024)


def warmup_image_generator(model: ImageGenerator) -> Image.Image | None:
    """Warmup the image generator with a small image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small dummy image for warmup (needed for edit models)
        dummy_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        dummy_path = Path(tmpdir) / "warmup.png"
        dummy_image.save(dummy_path)

        for result in model.generate(
            prompt="Warmup",
            height=256,
            width=256,
            quality="low",
            seed=2,
            image_path=dummy_path,
        ):
            if not isinstance(result, tuple):
                return result
    return None


def generate_image(
    model: ImageGenerator,
    task: ImageGenerationTaskParams | ImageEditsInternalParams,
) -> Generator[ImageGenerationResponse | PartialImageResponse, None, None]:
    """Generate image(s), optionally yielding partial results.

    When partial_images > 0 or stream=True, yields PartialImageResponse for
    intermediate images, then ImageGenerationResponse for the final image.

    Yields:
        PartialImageResponse for intermediate images (if partial_images > 0)
        ImageGenerationResponse for the final complete image
    """
    width, height = parse_size(task.size)
    quality: Literal["low", "medium", "high"] = task.quality or "medium"
    seed = 2  # TODO(ciaran): Randomise when not testing anymore

    # Handle streaming params for both generation and edit tasks
    partial_images = task.partial_images or (3 if task.stream else 0)

    image_path: Path | None = None
    image_strength: float | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        if isinstance(task, ImageEditsInternalParams):
            # Decode base64 image data and save to temp file
            image_path = Path(tmpdir) / "input.png"
            image_path.write_bytes(base64.b64decode(task.image_data))
            image_strength = task.image_strength

        # Iterate over generator results
        for result in model.generate(
            prompt=task.prompt,
            height=height,
            width=width,
            quality=quality,
            seed=seed,
            image_path=image_path,
            image_strength=image_strength,
            partial_images=partial_images,
        ):
            if isinstance(result, tuple):
                # Partial image: (Image, partial_index, total_partials)
                image, partial_idx, total_partials = result
                buffer = io.BytesIO()
                image_format = task.output_format.upper()
                if image_format == "JPG":
                    image_format = "JPEG"
                image.save(buffer, format=image_format)

                yield PartialImageResponse(
                    image_data=buffer.getvalue(),
                    format=task.output_format,
                    partial_index=partial_idx,
                    total_partials=total_partials,
                )
            else:
                # Final image
                image = result
                buffer = io.BytesIO()
                image_format = task.output_format.upper()
                if image_format == "JPG":
                    image_format = "JPEG"
                image.save(buffer, format=image_format)

                yield ImageGenerationResponse(
                    image_data=buffer.getvalue(),
                    format=task.output_format,
                )
