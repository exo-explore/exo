"""The fused forward pass on sharded Laguna models: fused after pipeline or tensor
sharding (as the post_load hook does), two ranks over MLX's ring backend on localhost.

- Pipeline: every split is bit-identical to a single device, fused or not.
- Tensor: the fused model gives bit-identical logits to the unfused tensor-parallel one.
"""

import json
import multiprocessing as mp
import os
import tempfile
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.laguna import Model

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.models.laguna import fast
from exo.worker.engines.mlx.models.laguna.tests.test_laguna_fast import (
    _N_LAYERS,  # pyright: ignore[reportPrivateUsage]
    _model,  # pyright: ignore[reportPrivateUsage]
)
from exo.worker.engines.mlx.sharding.pipeline import get_inner_model, get_layers

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused kernels are Metal kernels"
)

_PROMPT = [[(7 * i + 3) % 256 for i in range(600)]]
_DECODE_STEPS = 4


def _drain(
    loader: Any,  # pyright: ignore[reportAny]
) -> nn.Module:
    while True:
        try:
            response = cast(ModelLoadingResponse, next(loader))  # pyright: ignore[reportAny]
            assert response.total > 0
        except StopIteration as stop:
            return cast(nn.Module, stop.value)


def _fuse(model: nn.Module) -> None:
    layers = cast(list[nn.Module], get_layers(get_inner_model(model)))
    fast.fuse_for_inference(model, layers)


def _generate(model: nn.Module) -> list[list[float]]:
    cache = cast(Model, model).make_cache()
    logits = model(mx.array(_PROMPT), cache=cache)[:, -1, :]
    steps = [logits]
    for _ in range(_DECODE_STEPS):
        next_id = mx.argmax(logits, axis=-1, keepdims=True)
        logits = model(next_id, cache=cache)[:, -1, :]
        steps.append(logits)
    mx.eval(*steps)
    return [cast(list[float], step[0].astype(mx.float32).tolist()) for step in steps]


def _card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/laguna-tiny-fast"),
        storage_size=Memory.from_mb(1),
        n_layers=_N_LAYERS,
        hidden_size=256,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _run_rank(
    mode: str,
    split: int,
    quantization: str | None,
    fused: bool,
    rank: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        model: nn.Module = _model(quantization)
        if mode == "pipeline":
            shard = PipelineShardMetadata(
                model_card=_card(),
                device_rank=rank,
                world_size=2,
                start_layer=0 if rank == 0 else split,
                end_layer=split if rank == 0 else _N_LAYERS,
                n_layers=_N_LAYERS,
            )
            model = _drain(pipeline_auto_parallel(model, group, shard))
        else:
            model = _drain(tensor_auto_parallel(model, group))
        if fused:
            _fuse(model)
        result_queue.put((rank, True, _generate(model)))  # pyright: ignore[reportAny]
    except Exception as error:
        result_queue.put((rank, False, repr(error)))  # pyright: ignore[reportAny]


def _sharded_logits(
    mode: str, split: int, quantization: str | None, fused: bool, base_port: int
) -> list[list[float]]:
    context = mp.get_context("spawn")
    hosts = [f"127.0.0.1:{base_port + rank}" for rank in range(2)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name
    try:
        result_queue: Any = context.Queue()
        processes = [
            context.Process(
                target=_run_rank,
                args=(
                    mode,
                    split,
                    quantization,
                    fused,
                    rank,
                    hostfile_path,
                    result_queue,
                ),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        results: dict[int, tuple[bool, object]] = {}
        for _ in processes:
            rank, ok, value = cast(
                tuple[int, bool, object],
                result_queue.get(timeout=180),  # pyright: ignore[reportAny]
            )
            results[rank] = (ok, value)
        for process in processes:
            process.join(timeout=30)
        assert all(ok for ok, _ in results.values()), results
        assert results[0][1] == results[1][1]
        return cast(list[list[float]], results[0][1])
    finally:
        os.unlink(hostfile_path)


@pytest.mark.parametrize(
    "split,quantization,base_port",
    [(2, None, 29760), (1, "nvfp4", 29762), (4, "affine", 29764)],
)
def test_fused_pipeline_split_matches_a_single_device(
    split: int, quantization: str | None, base_port: int
) -> None:
    single = _generate(_model(quantization))
    sharded = _sharded_logits("pipeline", split, quantization, True, base_port)
    assert np.array_equal(np.array(sharded), np.array(single))


@pytest.mark.parametrize(
    "quantization,base_port", [(None, 29770), ("nvfp4", 29774), ("affine", 29778)]
)
def test_fused_tensor_split_matches_the_unfused_one(
    quantization: str | None, base_port: int
) -> None:
    unfused = _sharded_logits("tensor", 0, quantization, False, base_port)
    fused = _sharded_logits("tensor", 0, quantization, True, base_port + 2)
    assert np.array_equal(np.array(fused), np.array(unfused))
