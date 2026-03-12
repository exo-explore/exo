import argparse
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam as ChatCompletionMessageParam
from vllm.entrypoints.cli.types import CLISubcommand as CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

def chat(system_prompt: str | None, model_name: str, client: OpenAI) -> None: ...

class ChatCommand(CLISubcommand):
    name: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...

class CompleteCommand(CLISubcommand):
    name: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...

def cmd_init() -> list[CLISubcommand]: ...
