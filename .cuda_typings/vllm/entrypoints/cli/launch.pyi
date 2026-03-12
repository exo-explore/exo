import argparse
from _typeshed import Incomplete
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.entrypoints.cli.types import CLISubcommand as CLISubcommand
from vllm.entrypoints.openai.api_server import (
    build_and_serve_renderer as build_and_serve_renderer,
    setup_server as setup_server,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser as make_arg_parser,
    validate_parsed_serve_args as validate_parsed_serve_args,
)
from vllm.entrypoints.utils import (
    VLLM_SUBCMD_PARSER_EPILOG as VLLM_SUBCMD_PARSER_EPILOG,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

logger: Incomplete
DESCRIPTION: str

class LaunchSubcommandBase(CLISubcommand):
    help: str
    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None: ...
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...

class RenderSubcommand(LaunchSubcommandBase):
    name: str
    help: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...

class LaunchSubcommand(CLISubcommand):
    name: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    def validate(self, args: argparse.Namespace) -> None: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...

def cmd_init() -> list[CLISubcommand]: ...
async def run_launch_fastapi(args: argparse.Namespace) -> None: ...
