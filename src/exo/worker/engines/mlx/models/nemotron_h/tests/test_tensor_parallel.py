"""Tensor parallelism for Nemotron-H on quantized weights.

Every published Nemotron-H conversion that exo serves is quantized, so the Mamba-2
`in_proj` is a `QuantizedLinear`: a packed weight plus per-row scales and biases.
Sharding has to slice all of them, not only the weight.
"""

import json
import multiprocessing as mp
import os
import tempfile
from collections.abc import Generator
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nemotron_h import Model, ModelArgs, NemotronHMamba2Mixer

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.models.nemotron_h.tests.tiny import CONFIG, PROMPT


def _quantized_model() -> Model:
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(CONFIG))
    for layer in model.layers:
        mixer = layer.mixer
        # The Mamba parameters a checkpoint provides (mlx-lm initialises them to constants).
        if isinstance(mixer, NemotronHMamba2Mixer):
            heads = mixer.num_heads
            mixer.update(
                {
                    "A_log": mx.random.normal((heads,)),
                    "D": mx.random.normal((heads,)),
                    "dt_bias": mx.random.normal((heads,)),
                }
            )
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


def _drain(
    loading: Generator[ModelLoadingResponse, None, nn.Module],
) -> nn.Module:
    while True:
        try:
            next(loading)
        except StopIteration as done:
            return cast(nn.Module, done.value)


def _tensor_parallel_rank(
    rank: int,
    hostfile: str,
    results: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

        group = mx.distributed.init(backend="ring", strict=True)
        inputs = mx.array(PROMPT)
        single = _quantized_model()(inputs)
        sharded = _drain(tensor_auto_parallel(_quantized_model(), group))
        parallel = sharded(inputs)
        difference = mx.abs(parallel - single).max() / mx.abs(single).max()
        results.put((rank, float(difference.item()), None))  # pyright: ignore[reportAny]
    except Exception as error:
        results.put((rank, None, repr(error)))  # pyright: ignore[reportAny]


def test_quantized_tensor_parallel_matches_single_device() -> None:
    world_size = 2
    hosts = [f"127.0.0.1:{29760 + rank}" for rank in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        json.dump(hosts, file)
        hostfile = file.name
    context = mp.get_context("spawn")
    results: Any = context.Queue()
    try:
        processes = [
            context.Process(
                target=_tensor_parallel_rank, args=(rank, hostfile, results)
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=120) for _ in range(world_size)]  # pyright: ignore[reportAny]
        for process in processes:
            process.join(timeout=30)
    finally:
        os.unlink(hostfile)

    for rank, difference, error in cast(
        list[tuple[int, float | None, str | None]], outcomes
    ):
        assert error is None, f"rank {rank}: {error}"
        assert difference is not None
        # Only the order of the partial sums differs between the two runs.
        assert difference < 1e-4, f"rank {rank}: relative logit difference {difference}"
