"""Type stubs for mlx_lm.models.pipeline"""

from typing import Any, Optional

import mlx.core as mx

class PipelineMixin:
    pipeline_rank: int
    pipeline_size: int
    start_idx: int
    end_idx: Optional[int]

    def __init__(self) -> None: ...
    @property
    def pipeline_layers(self) -> list[Any]: ...
    def pipeline(
        self, group: mx.distributed.Group, split: Optional[list[int]] = None
    ) -> None: ...
