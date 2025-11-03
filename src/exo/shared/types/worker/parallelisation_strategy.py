from typing import Literal

ParallelisationStrategyType = Literal[
    "auto",
    "pipeline",
    "tensor",
]


def strategy_error() -> ValueError:
    return ValueError("Unexpected strategy")
