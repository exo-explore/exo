import io
from typing import Generator, Literal

import mlx.core as mx
from PIL import Image

from exo.shared.types.api import ImageGenerationTaskParams
from exo.shared.types.worker.runner_response import ImageGenerationResponse
from exo.worker.engines.mflux.distributed_flux import DistributedFlux1

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


def warmup_mflux(model: DistributedFlux1) -> Image.Image | None:
    """
    Warmup the model by generating a small image.

    Returns the image for rank 0, None for other ranks in distributed mode.
    """
    return model.generate(
        prompt="Warmup",
        height=256,
        width=256,
        quality="low",
        seed=2,
    )


def mflux_generate(
    model: DistributedFlux1,
    task: ImageGenerationTaskParams,
) -> Generator[ImageGenerationResponse, None, None]:
    """
    Generate an image using the DistributedFlux1 model.

    For distributed inference, only rank 0 yields the response.
    Other ranks participate in the pipeline but yield nothing.
    """
    # Parse parameters
    width, height = parse_size(task.size)
    quality: Literal["low", "medium", "high"] = task.quality or "medium"
    seed = 2  # TODO: not in OAI API?

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
