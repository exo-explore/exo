from typing import Any

import mlx.core as mx
import mlx.nn as nn

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
