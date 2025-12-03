import io
from typing import Generator

import mlx.core as mx
from mflux.config.config import Config
from mflux.models.flux.variants.txt2img.flux import Flux1
from PIL import Image

from exo.shared.types.api import ImageGenerationTaskParams
from exo.shared.types.worker.runner_response import ImageGenerationResponse
from exo.worker.engines.mflux.generator.flux1 import generate_image

image_generation_stream = mx.new_stream(mx.default_device())


def warmup_mflux(model: Flux1) -> Image.Image:
    prompt = "Warmup"
    image = model.generate_image(
        seed=2,
        prompt=prompt,
        config=Config(num_inference_steps=1, height=256, width=256),
    )

    return image.image


def mflux_generate(
    model: Flux1,
    task: ImageGenerationTaskParams,
) -> Generator[ImageGenerationResponse]:
    image = generate_image(model=model, task=task)

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
