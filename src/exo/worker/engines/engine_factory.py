from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

from pydantic import BaseModel

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse

"""
The annotation frozen=True attribute makes the class immutable, preventing
side-effects during runtime. This is critical since we are working with
heterogenuous consumer-grade devices.
"""

class Engine(BaseModel, frozen=True):
    # (BoundInstance) -> context
    initialize: Callable[[BoundInstance], Any]

    # (BoundInstance, context) -> model, tokenizer
    load: Callable[..., tuple[Any, Any]]

    # (model, tokenizer, task, prompt) -> Generator[GenerationResponse]
    generate: Callable[..., Generator[GenerationResponse | ToolCallResponse]]

    apply_chat_template: Callable[..., str]
    detect_thinking_prompt_suffix: Callable[..., bool]

    # (model, tokenizer) -> initialize
    warmup: Callable[..., int]
    cleanup: Callable[[], None]

def create_engine(bound_instance: BoundInstance) -> Engine:
    from exo.shared.types.worker.instances import (
        MlxJacclInstance,
        MlxRingInstance,
        TinygradInstance,
    )

    match bound_instance.instance:
        case MlxRingInstance() | MlxJacclInstance():
            # Lazy import - MLX must be loaded only on MacOS.
            from exo.worker.engines.mlx.generator.generate import (
                mlx_generate_with_postprocessing,
                warmup_inference,
            )
            from exo.worker.engines.mlx.utils_mlx import (
                apply_chat_template,
                detect_thinking_prompt_suffix,
                initialize_mlx,
                load_mlx_items,
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
            from exo.shared.tokenizer.chat_template import (
                apply_chat_template,
            )
            from exo.worker.engines.tinygrad.generator.generate import (
                tinygrad_generate,
                warmup_inference,
            )
            from exo.worker.engines.tinygrad.utils_tinygrad import (
                initialize_tinygrad,
                load_tinygrad_items,
            )

            return Engine(
                initialize=initialize_tinygrad,
                load=load_tinygrad_items,
                generate=tinygrad_generate,
                apply_chat_template=apply_chat_template,
                detect_thinking_prompt_suffix=_tinygrad_detect_thinking,
                warmup=warmup_inference,
                cleanup=_tinygrad_cleanup,
            )

        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(f"Unsupported Instance: {type(bound_instance.instance)}")


def _mlx_cleanup() -> None:
    from mlx.core import clear_cache
    clear_cache()


def _tinygrad_cleanup() -> None:
    from exo.worker.engines.tinygrad.generator.generate import cleanup_jit_state
    cleanup_jit_state()


def _tinygrad_detect_thinking(prompt: str, tokenizer: Any) -> bool:  # pyright: ignore[reportAny]
    return False
