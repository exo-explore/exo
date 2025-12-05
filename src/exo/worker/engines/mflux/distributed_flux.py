from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.shards import PipelineShardMetadata


class DistributedFlux1:
    """
    Wrapper for Flux1 that attaches distributed group and shard metadata.

    This wrapper enables the generation runtime to access distributed context
    (group, rank, world_size, shard boundaries).
    """

    __slots__ = ("_model", "_group", "_shard_metadata")

    _model: Flux1
    _group: mx.distributed.Group
    _shard_metadata: PipelineShardMetadata

    def __init__(
        self,
        model: Flux1,
        group: mx.distributed.Group,
        shard_metadata: PipelineShardMetadata,
    ) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_shard_metadata", shard_metadata)

    @property
    def model(self) -> Flux1:
        """The underlying Flux1 model."""
        return self._model

    @property
    def group(self) -> mx.distributed.Group:
        """The MLX distributed group for this model."""
        return self._group

    @property
    def shard_metadata(self) -> PipelineShardMetadata:
        """Shard metadata containing layer assignments and device info."""
        return self._shard_metadata

    @property
    def rank(self) -> int:
        """This device's rank in the distributed group."""
        return self._shard_metadata.device_rank

    @property
    def world_size(self) -> int:
        """Total number of devices in the distributed group."""
        return self._shard_metadata.world_size

    @property
    def is_first_stage(self) -> bool:
        """True if this device is the first stage in the pipeline."""
        return self._shard_metadata.device_rank == 0

    @property
    def is_last_stage(self) -> bool:
        """True if this device is the last stage in the pipeline."""
        return self._shard_metadata.device_rank == self._shard_metadata.world_size - 1

    @property
    def is_distributed(self) -> bool:
        """True if running in distributed mode (world_size > 1)."""
        return self._shard_metadata.world_size > 1

    # Delegate attribute access to the underlying model.
    # Guarded with TYPE_CHECKING to prevent type checker complaints
    # while still providing full delegation at runtime.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            return getattr(self._model, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in ("_model", "_group", "_shard_metadata"):
                object.__setattr__(self, name, value)
            else:
                setattr(self._model, name, value)
