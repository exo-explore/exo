import argparse
from vllm.entrypoints.cli.types import CLISubcommand as CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

class CollectEnvSubcommand(CLISubcommand):
    name: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...

def cmd_init() -> list[CLISubcommand]: ...
