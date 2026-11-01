"""Pipeline shards of Nemotron-H that hold no attention layer."""

from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel
from exo.worker.engines.mlx.cache import copy_snapshot_entry
from exo.worker.engines.mlx.models.nemotron_h.model import Model, ModelArgs
from exo.worker.engines.mlx.models.nemotron_h.tests.tiny import CONFIG, PROMPT


def _drain(
    loading: Generator[ModelLoadingResponse, None, nn.Module],
) -> nn.Module:
    while True:
        try:
            next(loading)
        except StopIteration as done:
            return cast(nn.Module, done.value)


def _shard(start_layer: int, end_layer: int) -> Model:
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(CONFIG))
    n_layers = len(model.layers)
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("test/tiny-nemotron-h"),
            storage_size=Memory.from_mb(1),
            n_layers=n_layers,
            hidden_size=cast(int, CONFIG["hidden_size"]),
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=0,
        world_size=1,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
    )
    return cast(
        Model, _drain(pipeline_auto_parallel(model, mx.distributed.init(), shard))
    )


def test_a_shard_without_attention_runs_on_restored_ssm_states() -> None:
    # Layers 2-4 of the tiny model are MoE, Mamba and MLP: no attention layer, so the
    # attention mask is built from an ArraysCache.
    model = _shard(2, 5)
    inputs = mx.array(PROMPT)
    expected = model(inputs, cache=model.make_cache())

    # exo's prefill restores the SSM states it snapshots into new ArraysCache objects
    # (exo's make_kv_cache is the model's make_cache).
    cache = model.make_cache()
    restored = [
        copy_snapshot_entry(entry) if isinstance(entry, ArraysCache) else entry
        for entry in cache
    ]
    assert any(isinstance(entry, ArraysCache) for entry in restored)

    assert mx.array_equal(model(inputs, cache=restored), expected)
