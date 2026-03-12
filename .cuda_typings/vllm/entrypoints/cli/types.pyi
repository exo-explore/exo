import argparse
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

class CLISubcommand:
    name: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    def validate(self, args: argparse.Namespace) -> None: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...
