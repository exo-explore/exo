from collections.abc import Callable as Callable
from typing import Any
from vllm import PoolingParams as PoolingParams, SamplingParams as SamplingParams
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData as CachedRequestData,
    GrammarOutput as GrammarOutput,
    NewRequestData as NewRequestData,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.request import Request as Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunner

def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None: ...
