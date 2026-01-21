# type: ignore
import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.utils_mlx import apply_chat_template, shard_and_load


class MockLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.custom_attr = "test_value"
        self.use_sliding = True

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x * 2


@dataclass(frozen=True)
class PipelineTestConfig:
    model_path: Path
    total_layers: int
    base_port: int
    max_tokens: int


def create_hostfile(world_size: int, base_port: int) -> tuple[str, list[str]]:
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    return hostfile_path, hosts


# Use GPT OSS 20b to test as it is a model with a lot of strange behaviour

DEFAULT_GPT_OSS_CONFIG = PipelineTestConfig(
    model_path=EXO_MODELS_DIR / "mlx-community--gpt-oss-20b-MXFP4-Q8",
    total_layers=24,
    base_port=29600,
    max_tokens=200,
)


DEFAULT_GPT_OSS_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"


def run_gpt_oss_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    layer_splits: list[tuple[int, int]],
    prompt_tokens: int,
    prefill_step_size: int,
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 200,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        start_layer, end_layer = layer_splits[rank]

        shard_meta = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId(DEFAULT_GPT_OSS_MODEL_ID),
                storage_size=Memory.from_gb(12),
                n_layers=24,
                hidden_size=2880,
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=24,
        )

        model, tokenizer = shard_and_load(shard_meta, group)
        model = cast(Model, model)

        # Generate a prompt of exact token length
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        base_len = len(base_tokens)

        # Build prompt with approximate target length
        repeats = (prompt_tokens // base_len) + 2
        long_text = base_text * repeats
        tokens = tokenizer.encode(long_text)
        # Truncate to exact target length
        tokens = tokens[:prompt_tokens]
        prompt_text = tokenizer.decode(tokens)

        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content=prompt_text)],
            max_tokens=max_tokens,
        )

        prompt = apply_chat_template(tokenizer, task)

        generated_text = ""

        for response in mlx_generate(
            model=model, tokenizer=tokenizer, task=task, prompt=prompt
        ):
            generated_text += response.text
            if response.finish_reason is not None:
                break

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]


def run_gpt_oss_tensor_parallel_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    prompt_tokens: int,
    prefill_step_size: int,
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 10,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        # For tensor parallelism, all devices run all layers
        shard_meta = TensorShardMetadata(
            model_card=ModelCard(
                model_id=ModelId(DEFAULT_GPT_OSS_MODEL_ID),
                storage_size=Memory.from_gb(12),
                n_layers=24,
                hidden_size=2880,
                supports_tensor=True,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=0,
            end_layer=24,
            n_layers=24,
        )

        model, tokenizer = shard_and_load(shard_meta, group)
        model = cast(Model, model)

        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        base_len = len(base_tokens)

        repeats = (prompt_tokens // base_len) + 2
        long_text = base_text * repeats
        tokens = tokenizer.encode(long_text)
        tokens = tokens[:prompt_tokens]
        prompt_text = tokenizer.decode(tokens)

        task = ChatCompletionTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content=prompt_text)],
            max_tokens=max_tokens,
        )

        prompt = apply_chat_template(tokenizer, task)

        generated_text = ""
        for response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
        ):
            generated_text += response.text
            if response.finish_reason is not None:
                break

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]
