import argparse
from typing import Any
from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format as convert_to_pytorch_benchmark_format,
    write_to_json as write_to_json,
)
from vllm.engine.arg_utils import EngineArgs as EngineArgs
from vllm.inputs import PromptType as PromptType
from vllm.sampling_params import BeamSearchParams as BeamSearchParams

def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None: ...
def add_cli_args(parser: argparse.ArgumentParser): ...
def main(args: argparse.Namespace): ...
