from mflux.config.config import Config
from mflux.models.flux.variants.txt2img.flux import Flux1
from PIL import Image

from exo.shared.types.api import ImageGenerationTaskParams


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


def generate_image(
    model: Flux1,
    task: ImageGenerationTaskParams,
) -> Image.Image:
    # Parse parameters
    width, height = parse_size(task.size)
    quality = task.quality or "medium"

    # TODO: Flux1 only
    steps = 2
    if quality == "low":
        steps = 1
    elif quality == "high":
        steps = 4

    image = model.generate_image(
        seed=2,
        prompt=task.prompt,
        config=Config(num_inference_steps=steps, height=height, width=width),
    )

    return image.image
