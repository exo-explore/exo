from typing import Literal

ParallelisationStrategyType = Literal[
    "auto",
    "pipeline",
    "tensor",
    "tensor_rdma",
    "pipeline_rdma",
]


def strategy_error() -> ValueError:
    return ValueError("Unexpected strategy")
