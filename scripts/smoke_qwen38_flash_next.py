#!/usr/bin/env python3
"""Run a short full-weight Qwen3.8 Flash Next generation through EXO."""

import argparse
import os
import time
from pathlib import Path
from typing import cast

import mlx.core as mx

# The smoke harness does not serve the dashboard, but EXO resolves its path when
# importing API response types used by the generator.
os.environ.setdefault(
    "EXO_DASHBOARD_DIR",
    str(Path(__file__).resolve().parents[1] / "dashboard" / "build"),
)

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    load_language_model,
    load_tokenizer_for_model_id,
)

DEFAULT_MODEL_ID = ModelId("sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--model-id", type=ModelId, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: EXO QWEN38 READY",
    )
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the same request with one shared EXO prefix cache.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = cast(Path, args.model_path).expanduser().resolve()
    model_id = cast(ModelId, args.model_id)
    user_prompt = cast(str, args.prompt)
    max_output_tokens = cast(int, args.max_output_tokens)
    repeat = cast(int, args.repeat)
    thinking = cast(bool, args.thinking)
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"No model config found at {model_path}")
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    load_started = time.perf_counter()
    loaded_model, config = load_language_model(model_path, lazy=True, strict=False)
    model = cast(Model, loaded_model)
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    print(
        f"loaded {config.get('model_type')} in "
        f"{time.perf_counter() - load_started:.1f}s"
    )

    task = TextGenerationTaskParams(
        model=model_id,
        input=[
            InputMessage(role="user", content=InputMessageContent(user_prompt)),
        ],
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        enable_thinking=thinking,
    )
    prompt = apply_chat_template(tokenizer, task)

    prefix_cache = KVPrefixCache(None)
    for run_number in range(1, repeat + 1):
        generation_started = time.perf_counter()
        text_parts: list[str] = []
        final_stats = None
        final_usage = None
        for response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=prefix_cache,
            group=None,
        ):
            text_parts.append(response.text)
            if response.stats is not None:
                final_stats = response.stats
            if response.usage is not None:
                final_usage = response.usage

        mx.synchronize()
        print(f"run {run_number} response:", "".join(text_parts).strip())
        print(
            f"run {run_number} elapsed: {time.perf_counter() - generation_started:.1f}s"
        )
        if final_stats is not None:
            print(f"run {run_number} stats:", final_stats.model_dump_json())
        if final_usage is not None:
            print(f"run {run_number} usage:", final_usage.model_dump_json())


if __name__ == "__main__":
    main()
