import base64
import io
import tempfile
from pathlib import Path
from typing import Generator, Literal

from PIL import Image

from exo.shared.types.api import ImageEditsInternalParams, ImageGenerationTaskParams
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
    task: ImageGenerationTaskParams | ImageEditsInternalParams,
) -> Generator[ImageGenerationResponse, None, None]:
    width, height = parse_size(task.size)
    quality: Literal["low", "medium", "high"] = task.quality or "medium"
    seed = 2  # TODO(ciaran): Randomise when not testing anymore

    image_path: Path | None = None
    image_strength: float | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        if isinstance(task, ImageEditsInternalParams):
            # Decode base64 image data and save to temp file
            image_path = Path(tmpdir) / "input.png"
            image_path.write_bytes(base64.b64decode(task.image_data))
            image_strength = task.image_strength

        image = model.generate(
            prompt=task.prompt,
            height=height,
            width=width,
            quality=quality,
            seed=seed,
            image_path=image_path,
            image_strength=image_strength,
        )

        # Only final rank returns the image
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
