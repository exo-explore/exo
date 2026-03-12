from _typeshed import Incomplete
from argparse import Namespace
from fastapi import Request as Request
from http import HTTPStatus
from logging import Logger
from vllm import envs as envs
from vllm.engine.arg_utils import EngineArgs as EngineArgs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo as ErrorInfo,
    ErrorResponse as ErrorResponse,
    GenerationError as GenerationError,
    StreamOptions as StreamOptions,
)
from vllm.entrypoints.openai.models.protocol import LoRAModulePath as LoRAModulePath
from vllm.logger import (
    current_formatter_type as current_formatter_type,
    init_logger as init_logger,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser

logger: Incomplete
VLLM_SUBCMD_PARSER_EPILOG: str

async def listen_for_disconnect(request: Request) -> None: ...
def with_cancellation(handler_func): ...
def decrement_server_load(request: Request): ...
def load_aware_call(func): ...
def cli_env_setup() -> None: ...
def get_max_tokens(
    max_model_len: int,
    max_tokens: int | None,
    input_length: int,
    default_sampling_params: dict,
    override_max_tokens: int | None = None,
) -> int: ...
def log_non_default_args(args: Namespace | EngineArgs): ...
def should_include_usage(
    stream_options: StreamOptions | None, enable_force_include_usage: bool
) -> tuple[bool, bool]: ...
def process_lora_modules(
    args_lora_modules: list[LoRAModulePath], default_mm_loras: dict[str, str] | None
) -> list[LoRAModulePath]: ...
def sanitize_message(message: str) -> str: ...
def log_version_and_model(lgr: Logger, version: str, model_name: str) -> None: ...
def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = ...,
    param: str | None = None,
) -> ErrorResponse: ...
