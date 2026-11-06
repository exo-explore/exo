"""Heterogeneous NPU and GPU pipelining engine for low-power edge acceleration."""

from exo.worker.engines.npu.pipeline_coordinator import PipelinedInferenceCoordinator
from exo.worker.engines.npu.pipelined_scheduler import HeterogeneousPipelinedScheduler

__all__ = [
    "HeterogeneousPipelinedScheduler",
    "PipelinedInferenceCoordinator",
]
