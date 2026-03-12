import argparse
from _typeshed import Incomplete
from vllm.entrypoints.cli.types import CLISubcommand as CLISubcommand
from vllm.entrypoints.utils import (
    VLLM_SUBCMD_PARSER_EPILOG as VLLM_SUBCMD_PARSER_EPILOG,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

logger: Incomplete

class RunBatchSubcommand(CLISubcommand):
    name: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...

def cmd_init() -> list[CLISubcommand]: ...
