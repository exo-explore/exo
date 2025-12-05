import io
from typing import Generator

import mlx.core as mx
from mflux.models.flux.variants.txt2img.flux import Flux1
from PIL import Image

from exo.shared.types.api import ImageGenerationTaskParams
from exo.shared.types.worker.runner_response import ImageGenerationResponse
from exo.worker.engines.mflux.distributed_flux import DistributedFlux1
from exo.worker.engines.mflux.generator.flux1 import generate_image

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


def warmup_mflux(model: Flux1 | DistributedFlux1) -> Image.Image:
    # Extract underlying model if wrapped
    underlying_model = model.model if isinstance(model, DistributedFlux1) else model
    return generate_image(
        model=underlying_model,
        prompt="Warmup",
        height=256,
        width=256,
        quality="low",
        seed=2,
    )


def mflux_generate(
    model: Flux1 | DistributedFlux1,
    task: ImageGenerationTaskParams,
) -> Generator[ImageGenerationResponse]:
    # Parse parameters
    width, height = parse_size(task.size)
    quality = task.quality or "medium"

    seed = 2  # TODO: not in OAI API?

    # Extract underlying model if wrapped
    # TODO: In future, use model.group for async pipeline when distributed
    underlying_model = model.model if isinstance(model, DistributedFlux1) else model

    image = generate_image(
        model=underlying_model,
        prompt=task.prompt,
        height=height,
        width=width,
        quality=quality,
        seed=seed,
    )

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
