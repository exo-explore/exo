from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Generator

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse

"""
The annotation frozen=True attribute makes the class immutable, preventing
side-effects during runtime. This is critical since we are working with
heterogenuous consumer-grade devices.
"""

@dataclass(frozen=True)
class Engine:
    # (BoundInstance) -> context
    initialize: Callable[[BoundInstance], Any]

    # (BoundInstance, context) -> model, tokenizer
    load: Callable[..., tuple[Any, Tokenizer]]

    # (model, tokenizer, task, prompt) -> Generator[GenerationResponse]
    generate: Callable[..., Generator[GenerationResponse | ToolCallResponse]]

    apply_chat_template: Callable[[Tokenizer, TextGenerationTaskParams], str]
    detect_thinking_prompt_suffix: Callable[[str, Tokenizer], bool]

    # (model, tokenizer) -> initialize
    warmup: Callable[..., int]
    cleanup: Callable[[], None]

def create_engine(bound_instance: BoundInstance) -> Engine:
    from exo.shared.types.worker.instances import (
        MlxRingInstance,
        MlxJacclInstance,
        TinygradInstance,
    )

    match bound_instance.instance:
        case MlxRingInstance() | MlxJacclInstance():
            # Lazy import - MLX must be loaded only on MacOS.
            from exo.worker.engines.mlx.utils_mlx import (
                    initialize_mlx, load_mlx_items,
                    apply_chat_template, detect_thinking_prompt_suffix
            )
            from exo.worker.engines.mlx.generator.generate import (
                    mlx_generate_with_postprocessing, warmup_inference
            )

            return Engine(
                initialize = initialize_mlx,
                load = load_mlx_items,
                generate=mlx_generate_with_postprocessing,
                apply_chat_template=apply_chat_template,
                detect_thinking_prompt_suffix=detect_thinking_prompt_suffix,
                warmup = warmup_inference,
                cleanup = _mlx_cleanup,
            )

        case TinygradInstance():
            # Lazy import - Tinygrad must be loaded on non-Apple systems
            from exo.worker.engines.tinygrad.utils_tinygrad import initialize_tinygrad, load_tinygrad_items
            from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate, warmup_inference

            return Engine(
                initialize = initialize_tinygrad,
                load = load_tinygrad_items,
                warmup = warmup_inference,
                cleanup = lambda: None,
            )

        case _:
            raise ValueError(f"Unsupported Instance: {type(bound_instance.instance)}")

def _mlx_cleanup() -> None:
    from mlx.core import clear_cache
    clear_cache()
