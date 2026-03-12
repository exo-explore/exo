import argparse
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format as convert_to_pytorch_benchmark_format,
    write_to_json as write_to_json,
)
from vllm.engine.arg_utils import EngineArgs as EngineArgs

@contextmanager
def cold_startup() -> Generator[None]: ...
def run_startup_in_subprocess(engine_args, result_queue) -> None: ...
def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None: ...
def add_cli_args(parser: argparse.ArgumentParser): ...
def main(args: argparse.Namespace): ...
