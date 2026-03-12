from cutlass import Boolean as Boolean, Int32
from cutlass.pipeline import (
    PipelineState as PipelineState,
    PipelineTmaAsync as PipelineTmaAsyncOg,
    PipelineTmaUmma as PipelineTmaUmmaOg,
    PipelineUserType,
)
from dataclasses import dataclass

class PipelineStateSimple:
    def __init__(self, stages: int, phase_index: Int32) -> None: ...
    def clone(self) -> PipelineStateSimple: ...
    @property
    def stages(self) -> int: ...
    @property
    def index(self) -> Int32: ...
    @property
    def phase(self) -> Int32: ...
    def advance(self) -> None: ...
    def __extract_mlir_values__(self): ...
    def __new_from_mlir_values__(self, values): ...

def make_pipeline_state(type: PipelineUserType, stages: int): ...
@dataclass(frozen=True)
class PipelineTmaAsync(PipelineTmaAsyncOg):
    @staticmethod
    def create(*args, **kwargs): ...
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Boolean | None = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ): ...

@dataclass(frozen=True)
class PipelineTmaUmma(PipelineTmaUmmaOg):
    @staticmethod
    def create(*args, **kwargs): ...
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Boolean | None = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ): ...
