import io
from typing import Generator

import mlx.core as mx
from PIL import Image, ImageDraw, ImageFont

from exo.shared.types.api import ImageGenerationTaskParams
from exo.shared.types.worker.runner_response import ImageGenerationResponse
from exo.worker.engines.mlx import Model
from exo.worker.runner.bootstrap import logger

image_generation_stream = mx.new_stream(mx.default_device())


def parse_size(size_str: str | None) -> tuple[int, int]:
    """
    Parse size parameter like '1024x1024' to (width, height) tuple.

    Args:
        size_str: Size string in format 'WIDTHxHEIGHT' or 'auto'

    Returns:
        Tuple of (width, height) in pixels

    Examples:
        >>> parse_size("1024x1024")
        (1024, 1024)
        >>> parse_size("1536x1024")
        (1536, 1024)
        >>> parse_size("auto")
        (1024, 1024)
        >>> parse_size(None)
        (1024, 1024)
    """
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


def create_mock_image(
    prompt: str,
    width: int,
    height: int,
    image_index: int,
    total_images: int,
    quality: str | None,
    output_format: str,
) -> bytes:
    """
    Create a simple mock image for testing purposes.

    Generates a solid-color image with text overlay showing the request
    parameters. Uses different colors for variety when generating multiple images.

    Args:
        prompt: Text description to display on image
        width: Image width in pixels
        height: Image height in pixels
        image_index: Index of this image (0-based)
        total_images: Total number of images being generated
        quality: Quality setting to display
        output_format: Image format (png/jpeg/webp)

    Returns:
        Image data as bytes in the requested format
    """
    image = Image.open("./dashboard/exo-logo.png")
    # Convert to bytes in requested format
    buffer = io.BytesIO()
    image_format = output_format.upper()
    if image_format == "JPG":
        image_format = "JPEG"
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def mlx_generate_image(
    model: Model,
    task: ImageGenerationTaskParams,
) -> Generator[ImageGenerationResponse]:
    """
    Generate mock images for testing purposes.

    This is a placeholder implementation that generates simple solid-color
    images with text overlays showing request parameters. In production, this
    would use MLX diffusion models to generate images from prompts.

    Args:
        model: The loaded MLX diffusion model (unused in mock implementation)
        task: Image generation task parameters (contains prompt as string)

    Yields:
        ImageGenerationResponse: Generated image data with metadata

    Note:
        This mock implementation respects key parameters (n, size, output_format)
        to enable realistic API testing without requiring actual diffusion models.
    """
    logger.info(f"Generating mock images for prompt: {task.prompt[:50]}")

    # Parse parameters
    width, height = parse_size(task.size)
    num_images = task.n or 1
    output_format = task.output_format or "png"
    quality = task.quality or "medium"

    logger.info("generating mock image")

    # Generate each requested image
    for i in range(num_images):
        logger.info(f"Generating mock image {i + 1}/{num_images}")

        image_bytes = create_mock_image(
            prompt=task.prompt,
            width=width,
            height=height,
            image_index=i,
            total_images=num_images,
            quality=quality,
            output_format=output_format,
        )

        # Determine finish reason (stop on last image)
        finish_reason = "stop" if i == num_images - 1 else None

        # Send complete image as single response (no artificial chunking)
        yield ImageGenerationResponse(
            image_data=image_bytes,
            format=output_format,
            finish_reason=finish_reason,
        )

    logger.info(f"Completed generating {num_images} mock images")
