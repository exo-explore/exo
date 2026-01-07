import io
from typing import Generator, Literal

from PIL import Image

from exo.shared.types.api import ImageGenerationTaskParams
from exo.shared.types.worker.runner_response import ImageGenerationResponse
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
    return model.generate(
        prompt="Warmup",
        height=256,
        width=256,
        quality="low",
        seed=2,
    )


def generate_image(
    model: ImageGenerator,
    task: ImageGenerationTaskParams,
) -> Generator[ImageGenerationResponse, None, None]:
    # Parse parameters
    width, height = parse_size(task.size)
    quality: Literal["low", "medium", "high"] = task.quality or "medium"
    seed = 2  # TODO(ciaran): Randomise when not testing anymore

    # Generate using the model's generate method
    image = model.generate(
        prompt=task.prompt,
        height=height,
        width=width,
        quality=quality,
        seed=seed,
    )

    # Only rank 0 returns the image
    if image is None:
        return

    buffer = io.BytesIO()
    image_format = task.output_format.upper()
    if image_format == "JPG":
        image_format = "JPEG"

    image.save(buffer, format=image_format)
    image_bytes = buffer.getvalue()

    # Send complete image as single response (no artificial chunking)
    yield ImageGenerationResponse(
        image_data=image_bytes,
        format=task.output_format,
    )
