import io
from typing import Generator

import mlx.core as mx
from mflux.config.config import Config
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.api import ImageGenerationTaskParams
from exo.shared.types.worker.runner_response import ImageGenerationResponse

image_generation_stream = mx.new_stream(mx.default_device())


def parse_size(size_str: str | None) -> tuple[int, int]:
    """
    Parse size parameter like '1024x1024' to (width, height) tuple.
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


def mflux_generate(
    model: Flux1,
    task: ImageGenerationTaskParams,
) -> Generator[ImageGenerationResponse]:
    # Parse parameters
    width, height = parse_size(task.size)
    num_images = task.n or 1
    output_format = task.output_format or "png"
    quality = task.quality or "medium"

    image = model.generate_image(
        seed=2,
        prompt=task.prompt,
        config=Config(num_inference_steps=2, height=height, width=width),
    )

    buffer = io.BytesIO()
    image_format = output_format.upper()
    if image_format == "JPG":
        image_format = "JPEG"

    image.image.save(buffer, format=image_format)
    image_bytes = buffer.getvalue()

    finish_reason = "stop"

    # Send complete image as single response (no artificial chunking)
    yield ImageGenerationResponse(
        image_data=image_bytes,
        format=output_format,
        finish_reason=finish_reason,
    )
