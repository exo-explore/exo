import argparse
from typing import Any, Literal
from vllm.benchmarks.datasets import (
    MultiModalConversationDataset as MultiModalConversationDataset,
    VisionArenaDataset as VisionArenaDataset,
)
from vllm.benchmarks.throughput import get_requests as get_requests
from vllm.engine.arg_utils import EngineArgs as EngineArgs
from vllm.utils.gc_utils import freeze_gc_heap as freeze_gc_heap
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule
from vllm.v1.engine.llm_engine import LLMEngine as LLMEngine

def get_timing_stats_from_engine(
    llm_engine: LLMEngine,
) -> dict[str, dict[str, float]]: ...
def collect_mm_processor_stats(llm_engine: LLMEngine) -> dict[str, list[float]]: ...
def calculate_mm_processor_metrics(
    stats_by_stage: dict[str, list[float]],
    selected_percentiles: list[float],
    *,
    unit: Literal["us", "ms", "s"] = "ms",
) -> dict[str, dict[str, float]]: ...
def validate_args(args) -> None: ...
def benchmark_multimodal_processor(args: argparse.Namespace) -> dict[str, Any]: ...
def add_cli_args(parser: argparse.ArgumentParser) -> None: ...
def main(args: argparse.Namespace) -> None: ...
