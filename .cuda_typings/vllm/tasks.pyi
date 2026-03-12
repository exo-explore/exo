from _typeshed import Incomplete
from typing import Literal

GenerationTask: Incomplete
GENERATION_TASKS: tuple[GenerationTask, ...]
PoolingTask: Incomplete
POOLING_TASKS: tuple[PoolingTask, ...]
ScoreType: Incomplete
FrontendTask: Incomplete
FRONTEND_TASKS: tuple[FrontendTask, ...]
SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]
