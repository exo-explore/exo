import argparse
from vllm.entrypoints.cli.benchmark.base import (
    BenchmarkSubcommandBase as BenchmarkSubcommandBase,
)
from vllm.entrypoints.cli.types import CLISubcommand as CLISubcommand
from vllm.entrypoints.utils import (
    VLLM_SUBCMD_PARSER_EPILOG as VLLM_SUBCMD_PARSER_EPILOG,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

class BenchmarkSubcommand(CLISubcommand):
    name: str
    help: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    def validate(self, args: argparse.Namespace) -> None: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...

def cmd_init() -> list[CLISubcommand]: ...
