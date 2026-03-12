import argparse
from vllm.entrypoints.cli.types import CLISubcommand as CLISubcommand

class BenchmarkSubcommandBase(CLISubcommand):
    help: str
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None: ...
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
