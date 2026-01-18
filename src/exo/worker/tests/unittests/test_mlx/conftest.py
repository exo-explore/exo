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


def run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    """Worker function for pipeline parallel tests. Runs in a spawned process."""
    import os

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    import mlx.core as mlx_core
    import mlx.nn as mlx_nn

    class MockLayerInner(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom_attr = "test_value"

        def __call__(
            self, x: mlx_core.array, *args: object, **kwargs: object
        ) -> mlx_core.array:
            return x * 2

    try:
        group = mlx_core.distributed.init(backend="ring", strict=True)

        mock = MockLayerInner()
        first = PipelineFirstLayer(mock, r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)

        x = mlx_core.ones((1, 4))
        result = composed(x)
        mlx_core.eval(result)

        success = result.shape == x.shape
        result_queue.put((rank, success, result))  # pyright: ignore[reportAny]
    except Exception as e:
        result_queue.put((rank, False, str(e)))  # pyright: ignore[reportAny]


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
    prefill_step_size: int,  # noqa: ARG001 - kept for API compatibility
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 200,
) -> None:
    import os
    import traceback

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    import mlx.core as mlx_core
    import mlx.nn as mlx_nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import load_model

    from exo.shared.types.api import ChatCompletionMessage
    from exo.shared.types.memory import Memory
    from exo.shared.types.models import ModelId, ModelMetadata
    from exo.shared.types.tasks import ChatCompletionTaskParams
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel
    from exo.worker.engines.mlx.generator.generate import mlx_generate
    from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

    try:
        group = mlx_core.distributed.init(backend="ring", strict=True)

        model: mlx_nn.Module
        model, _ = load_model(model_path, lazy=False, strict=False)
        tokenizer: TokenizerWrapper = load_tokenizer_for_model_id(
            "mlx-community/gpt-oss-20b-MXFP4-Q8", model_path
        )

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

        # Evaluate model parameters (required to avoid GPU timeout with distributed)
        mlx_core.eval(model.parameters())
        mlx_core.eval(model)

        # Barrier before generation
        barrier = mlx_core.distributed.all_sum(mlx_core.array([1.0]), group=group)
        mlx_core.eval(barrier)

        # Create task params for mlx_generate
        task = ChatCompletionTaskParams(
            model="mlx-community/gpt-oss-20b-MXFP4-Q8",
            messages=[
                ChatCompletionMessage(role="user", content=prompt_text),
            ],
            max_tokens=max_tokens,
        )

        # Use mlx_generate which has token broadcasting built in
        generated_text = ""
        for response in mlx_generate(model, tokenizer, task):  # type: ignore[arg-type]
            generated_text += response.text

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]


def run_gpt_oss_tensor_parallel_device(
    rank: int,
    world_size: int,  # noqa: ARG001 - kept for API compatibility
    hostfile_path: str,
    model_path: Path,
    prompt_tokens: int,
    prefill_step_size: int,  # noqa: ARG001 - kept for API compatibility
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 10,
) -> None:
    import os
    import traceback

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    import mlx.core as mlx_core
    import mlx.nn as mlx_nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import load_model

    from exo.shared.types.api import ChatCompletionMessage
    from exo.shared.types.tasks import ChatCompletionTaskParams
    from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel
    from exo.worker.engines.mlx.generator.generate import mlx_generate
    from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

    try:
        group = mlx_core.distributed.init(backend="ring", strict=True)

        model: mlx_nn.Module
        model, _ = load_model(model_path, lazy=False, strict=False)
        tokenizer: TokenizerWrapper = load_tokenizer_for_model_id(
            "mlx-community/gpt-oss-20b-MXFP4-Q8", model_path
        )

        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        base_len = len(base_tokens)

        repeats = (prompt_tokens // base_len) + 2
        long_text = base_text * repeats
        tokens = tokenizer.encode(long_text)
        tokens = tokens[:prompt_tokens]
        prompt_text = tokenizer.decode(tokens)

        model = tensor_auto_parallel(model, group)

        # Evaluate model parameters (required to avoid GPU timeout with distributed)
        mlx_core.eval(model.parameters())
        mlx_core.eval(model)

        barrier = mlx_core.distributed.all_sum(mlx_core.array([1.0]), group=group)
        mlx_core.eval(barrier)

        # Create task params for mlx_generate
        task = ChatCompletionTaskParams(
            model="mlx-community/gpt-oss-20b-MXFP4-Q8",
            messages=[
                ChatCompletionMessage(role="user", content=prompt_text),
            ],
            max_tokens=max_tokens,
        )

        # Use mlx_generate
        generated_text = ""
        for response in mlx_generate(model, tokenizer, task):  # type: ignore[arg-type]
            generated_text += response.text

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]
