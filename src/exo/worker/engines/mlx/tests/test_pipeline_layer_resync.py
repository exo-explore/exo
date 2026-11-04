# type: ignore
"""Pipeline sharding must let a model re-derive state from its sharded layers.

A hybrid stack (linear-attention layers interleaved with full-attention ones)
records which cache entry holds the first layer of each kind. Those indices are
derived from the *full* stack at construction time, but pipeline sharding
replaces the layer list afterwards, so on every shard but the first they point
at the wrong entry -- and the model then builds its masks from the wrong cache
without raising.
"""

import mlx.nn as nn

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel


class _Layer(nn.Module):
    def __init__(self, is_linear: bool):
        super().__init__()
        self.is_linear = is_linear

    def __call__(self, x, *args, **kwargs):
        return x


class _Inner(nn.Module):
    """Caches layer indices in __init__, the way hybrid stacks do."""

    def __init__(self, kinds: list[bool]):
        super().__init__()
        self.layers = [_Layer(k) for k in kinds]
        self._recompute()
        self.resynced = False

    def _recompute(self) -> None:
        self.ssm_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), None
        )
        self.fa_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear),
            None,
        )

    def resync_sharded_layers(self) -> None:
        self.resynced = True
        self._recompute()


class _Model(nn.Module):
    def __init__(self, kinds: list[bool]):
        super().__init__()
        self.model = _Inner(kinds)

    def __call__(self, x, cache=None):
        return x


def _shard(model, start, end, rank, world):
    meta = PipelineShardMetadata.model_construct(
        start_layer=start, end_layer=end, device_rank=rank, world_size=world
    )
    gen = pipeline_auto_parallel(model, group=None, model_shard_meta=meta)
    try:
        while True:
            next(gen)
    except StopIteration:
        pass


# layer_types repeating [linear, linear, linear, full]
KINDS = [(i % 4) != 3 for i in range(12)]


def test_resync_hook_runs_and_fixes_indices() -> None:
    model = _Model(KINDS)
    # Before sharding, indices describe the full stack.
    assert (model.model.ssm_idx, model.model.fa_idx) == (0, 3)

    # Shard 3 holds global layers 9..12 -> [linear, linear, full] locally,
    # so the first full-attention layer is now at local index 2, not 3.
    _shard(model, 9, 12, rank=2, world=3)

    assert model.model.resynced, "resync_sharded_layers was never called"
    assert model.model.fa_idx == 2
    assert model.model.ssm_idx == 0


class _InnerNoHook(_Inner):
    resync_sharded_layers = None  # model opts out; sharding must still work


class _ModelNoHook(nn.Module):
    def __init__(self, kinds: list[bool]):
        super().__init__()
        self.model = _InnerNoHook(kinds)

    def __call__(self, x, cache=None):
        return x


def test_model_without_hook_is_untouched() -> None:
    """A model that does not define the hook shards exactly as before."""
    model = _ModelNoHook(KINDS)
    _shard(model, 0, 4, rank=0, world=3)  # must not raise
    assert len(model.model.layers) == 4
