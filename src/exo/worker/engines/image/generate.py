import base64
import io
import random
import tempfile
import time
from pathlib import Path
from typing import Generator, Literal

import mlx.core as mx
from PIL import Image

from exo.shared.types.api import (
    AdvancedImageParams,
    ImageEditsInternalParams,
    ImageGenerationStats,
    ImageGenerationTaskParams,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import (
    ImageGenerationResponse,
    PartialImageResponse,
)
from exo.worker.engines.image.distributed_model import DistributedImageModel


def parse_size(size_str: str | None) -> tuple[int, int]:
    """Parse size parameter like '1024x1024' to (width, height) tuple."""
    if not size_str:
        return (1024, 1024)

    try:
        parts = size_str.split("x")
        if len(parts) == 2:
            width, height = int(parts[0]), int(parts[1])
            if width > 0 and height > 0:
                return (width, height)
    except (ValueError, AttributeError):
        pass

    raise ValueError(
        f"Invalid size format: '{size_str}'. Expected 'WIDTHxHEIGHT' (e.g., '1024x1024')"
    )


def warmup_image_generator(model: DistributedImageModel) -> Image.Image | None:
    """Warmup the image generator with a small image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small dummy image for warmup (needed for edit models)
        dummy_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        dummy_path = Path(tmpdir) / "warmup.png"
        dummy_image.save(dummy_path)

        warmup_params = AdvancedImageParams(num_inference_steps=2)

        for result in model.generate(
            prompt="Warmup",
            height=256,
            width=256,
            quality="low",
            image_path=dummy_path,
            advanced_params=warmup_params,
        ):
            if not isinstance(result, tuple):
                return result
    return None


def generate_image(
    model: DistributedImageModel,
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

    advanced_params = task.advanced_params
    if advanced_params is not None and advanced_params.seed is not None:
        seed = advanced_params.seed
    else:
        seed = random.randint(0, 2**32 - 1)

    is_bench = getattr(task, "bench", False)

    generation_start_time: float = 0.0

    if is_bench:
        mx.reset_peak_memory()
        generation_start_time = time.perf_counter()

    partial_images = task.partial_images or (3 if task.stream else 0)

    image_path: Path | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        if isinstance(task, ImageEditsInternalParams):
            # Decode base64 image data and save to temp file
            image_path = Path(tmpdir) / "input.png"
            image_path.write_bytes(base64.b64decode(task.image_data))

        # Iterate over generator results
        for result in model.generate(
            prompt=task.prompt,
            height=height,
            width=width,
            quality=quality,
            seed=seed,
            image_path=image_path,
            partial_images=partial_images,
            advanced_params=advanced_params,
        ):
            if isinstance(result, tuple):
                # Partial image: (Image, partial_index, total_partials)
                image, partial_idx, total_partials = result
                buffer = io.BytesIO()
                image_format = task.output_format.upper()
                if image_format == "JPG":
                    image_format = "JPEG"
                if image_format == "JPEG" and image.mode == "RGBA":
                    image = image.convert("RGB")
                image.save(buffer, format=image_format)

                yield PartialImageResponse(
                    image_data=buffer.getvalue(),
                    format=task.output_format,
                    partial_index=partial_idx,
                    total_partials=total_partials,
                )
            else:
                image = result

                stats: ImageGenerationStats | None = None
                if is_bench:
                    generation_end_time = time.perf_counter()
                    total_generation_time = generation_end_time - generation_start_time

                    num_inference_steps = model.get_steps_for_quality(quality)

                    seconds_per_step = (
                        total_generation_time / num_inference_steps
                        if num_inference_steps > 0
                        else 0.0
                    )

                    peak_memory_gb = mx.get_peak_memory() / (1024**3)

                    stats = ImageGenerationStats(
                        seconds_per_step=seconds_per_step,
                        total_generation_time=total_generation_time,
                        num_inference_steps=num_inference_steps,
                        num_images=task.n or 1,
                        image_width=width,
                        image_height=height,
                        peak_memory_usage=Memory.from_gb(peak_memory_gb),
                    )

                buffer = io.BytesIO()
                image_format = task.output_format.upper()
                if image_format == "JPG":
                    image_format = "JPEG"
                if image_format == "JPEG" and image.mode == "RGBA":
                    image = image.convert("RGB")
                image.save(buffer, format=image_format)

                yield ImageGenerationResponse(
                    image_data=buffer.getvalue(),
                    format=task.output_format,
                    stats=stats,
                )
