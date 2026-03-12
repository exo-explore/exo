import argparse
from _typeshed import Incomplete
from vllm.entrypoints.cli.types import CLISubcommand as CLISubcommand
from vllm.entrypoints.openai.api_server import (
    run_server as run_server,
    run_server_worker as run_server_worker,
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
from vllm.usage.usage_lib import UsageContext as UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser
from vllm.utils.network_utils import get_tcp_uri as get_tcp_uri
from vllm.utils.system_utils import (
    decorate_logs as decorate_logs,
    set_process_title as set_process_title,
)
from vllm.v1.engine.core import EngineCoreProc as EngineCoreProc
from vllm.v1.engine.utils import (
    CoreEngineProcManager as CoreEngineProcManager,
    launch_core_engines as launch_core_engines,
)
from vllm.v1.executor import Executor as Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor as MultiprocExecutor
from vllm.v1.metrics.prometheus import (
    setup_multiprocess_prometheus as setup_multiprocess_prometheus,
)
from vllm.v1.utils import (
    APIServerProcessManager as APIServerProcessManager,
    wait_for_completion_or_failure as wait_for_completion_or_failure,
)

logger: Incomplete
DESCRIPTION: str

class ServeSubcommand(CLISubcommand):
    name: str
    @staticmethod
    def cmd(args: argparse.Namespace) -> None: ...
    def validate(self, args: argparse.Namespace) -> None: ...
    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser: ...

def cmd_init() -> list[CLISubcommand]: ...
def run_headless(args: argparse.Namespace): ...
def run_multi_api_server(args: argparse.Namespace): ...
def run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None: ...
