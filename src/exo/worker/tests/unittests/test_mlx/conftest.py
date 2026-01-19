# type: ignore
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from exo.shared.constants import EXO_MODELS_DIR
from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
)


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
    import json
    import tempfile

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


def run_gpt_oss_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    model_path: Path,
    layer_splits: list[tuple[int, int]],
    prompt_tokens: int,
    prefill_step_size: int,
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 200,
) -> None:
    import os
    import traceback

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    import mlx.core as mlx_core
    from mlx_lm import load, stream_generate

    from exo.shared.types.memory import Memory
    from exo.shared.types.models import ModelId, ModelMetadata
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel

    try:
        group = mlx_core.distributed.init(backend="ring", strict=True)

        model, tokenizer = load(str(model_path))

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

        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

        start_layer, end_layer = layer_splits[rank]

        shard_meta = PipelineShardMetadata(
            model_meta=ModelMetadata(
                model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
                pretty_name="GPT-OSS 20B",
                storage_size=Memory.from_gb(12),
                n_layers=24,
                hidden_size=2880,
                supports_tensor=False,
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=24,
        )

        model = pipeline_auto_parallel(model, group, shard_meta)

        # Barrier before generation
        barrier = mlx_core.distributed.all_sum(mlx_core.array([1.0]), group=group)
        mlx_core.eval(barrier)

        generated_text = ""
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
        ):
            generated_text += response.text

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]


def run_gpt_oss_tensor_parallel_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    model_path: Path,
    prompt_tokens: int,
    prefill_step_size: int,
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 10,
) -> None:
    import os
    import traceback

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    import mlx.core as mlx_core
    from mlx_lm import load, stream_generate

    from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

    try:
        group = mlx_core.distributed.init(backend="ring", strict=True)

        model, tokenizer = load(str(model_path))

        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        base_len = len(base_tokens)

        repeats = (prompt_tokens // base_len) + 2
        long_text = base_text * repeats
        tokens = tokenizer.encode(long_text)
        tokens = tokens[:prompt_tokens]
        prompt_text = tokenizer.decode(tokens)

        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

        model = tensor_auto_parallel(model, group)

        barrier = mlx_core.distributed.all_sum(mlx_core.array([1.0]), group=group)
        mlx_core.eval(barrier)

        generated_text = ""
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
        ):
            generated_text += response.text

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]
