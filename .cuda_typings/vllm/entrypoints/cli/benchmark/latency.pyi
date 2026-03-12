import argparse
from vllm.benchmarks.latency import add_cli_args as add_cli_args, main as main
from vllm.entrypoints.cli.benchmark.base import (
    BenchmarkSubcommandBase as BenchmarkSubcommandBase,
)

class BenchmarkLatencySubcommand(BenchmarkSubcommandBase):
    name: str
    help: str
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None: ...
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
