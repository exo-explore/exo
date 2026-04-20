import base64
import io
import random
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Generator, Literal

import mlx.core as mx
from PIL import Image

from exo.api.types import (
    AdvancedImageParams,
    ImageEditsTaskParams,
    ImageGenerationStats,
    ImageGenerationTaskParams,
    ImageSize,
)
from exo.shared.constants import EXO_MAX_CHUNK_SIZE
from exo.shared.types.chunks import ImageChunk
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.worker.engines.image.distributed_model import DistributedImageModel


def parse_size(size_str: ImageSize) -> tuple[int, int]:
    """Parse size parameter like '1024x1024' to (width, height) tuple."""
    if size_str == "auto":
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
    task: ImageGenerationTaskParams | ImageEditsTaskParams,
    cancel_checker: Callable[[], bool] | None = None,
) -> Generator[ImageChunk, None, None]:
    """Generate image(s), optionally yielding partial results."""
    width, height = parse_size(task.size)
    quality: Literal["low", "medium", "high"] = task.quality or "medium"

    advanced_params = task.advanced_params
    if advanced_params is not None and advanced_params.seed is not None:
        base_seed = advanced_params.seed
    else:
        base_seed = random.randint(0, 2**32 - 1)

    is_bench = getattr(task, "bench", False)
    num_images = task.n or 1

    generation_start_time: float = 0.0

    if is_bench:
        mx.reset_peak_memory()
        generation_start_time = time.perf_counter()

    partial_images = (
        task.partial_images
        if task.partial_images is not None and task.stream is not None and task.stream
        else 0
    )

    image_path: Path | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        if isinstance(task, ImageEditsTaskParams):
            # Decode base64 image data and save to temp file
            image_path = Path(tmpdir) / "input.png"
            image_path.write_bytes(base64.b64decode(task.image_data))
            if task.size == "auto":
                with Image.open(image_path) as img:
                    width, height = img.size

        for image_num in range(num_images):
            # Increment seed for each image to ensure unique results
            current_seed = base_seed + image_num

            for result in model.generate(
                prompt=task.prompt,
                height=height,
                width=width,
                quality=quality,
                seed=current_seed,
                image_path=image_path,
                partial_images=partial_images,
                advanced_params=advanced_params,
                cancel_checker=cancel_checker,
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

                    yield from _process_image_response(
                        image_data=buffer.getvalue(),
                        image_format=task.output_format,
                        partial_index=partial_idx,
                        total_partials=total_partials,
                        image_index=image_num,
                        model_id=model.model_id,
                        stats=None,
                    )
                else:
                    image = result

                    # Only include stats on the final image
                    stats: ImageGenerationStats | None = None
                    if is_bench and image_num == num_images - 1:
                        generation_end_time = time.perf_counter()
                        total_generation_time = (
                            generation_end_time - generation_start_time
                        )

                        num_inference_steps = model.get_steps_for_quality(quality)
                        total_steps = num_inference_steps * num_images

                        seconds_per_step = (
                            total_generation_time / total_steps
                            if total_steps > 0
                            else 0.0
                        )

                        peak_memory = Memory.from_bytes(mx.get_peak_memory())

                        stats = ImageGenerationStats(
                            seconds_per_step=seconds_per_step,
                            total_generation_time=total_generation_time,
                            num_inference_steps=num_inference_steps,
                            num_images=num_images,
                            image_width=width,
                            image_height=height,
                            peak_memory_usage=peak_memory,
                        )

                    buffer = io.BytesIO()
                    image_format = task.output_format.upper()
                    if image_format == "JPG":
                        image_format = "JPEG"
                    if image_format == "JPEG" and image.mode == "RGBA":
                        image = image.convert("RGB")
                    image.save(buffer, format=image_format)

                    yield from _process_image_response(
                        image_data=buffer.getvalue(),
                        image_format=task.output_format,
                        stats=stats,
                        image_index=image_num,
                        model_id=model.model_id,
                        partial_index=None,
                        total_partials=None,
                    )


def _process_image_response(
    image_data: bytes,
    image_index: int,
    image_format: Literal["png", "jpeg", "webp"],
    partial_index: int | None,
    total_partials: int | None,
    stats: ImageGenerationStats | None,
    model_id: ModelId,
) -> Iterator[ImageChunk]:
    """Process a single image response and send chunks."""
    is_partial = partial_index is not None
    encoded_data = base64.b64encode(image_data).decode("utf-8")
    # Extract stats from final ImageGenerationResponse if available
    data_chunks = [
        encoded_data[i : i + EXO_MAX_CHUNK_SIZE]
        for i in range(0, len(encoded_data), EXO_MAX_CHUNK_SIZE)
    ]
    total_chunks = len(data_chunks)

    def _data_to_chunk(item: tuple[int, str]) -> ImageChunk:
        chunk_index, chunk_data = item
        # Only include stats on the last chunk of the final image
        chunk_stats = (
            stats if chunk_index == total_chunks - 1 and not is_partial else None
        )

        return ImageChunk(
            model=model_id,
            data=chunk_data,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            image_index=image_index,
            is_partial=is_partial,
            partial_index=partial_index,
            total_partials=total_partials,
            stats=chunk_stats,
            format=image_format,
        )

    return map(_data_to_chunk, enumerate(data_chunks))
