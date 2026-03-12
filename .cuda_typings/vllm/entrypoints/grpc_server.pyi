import argparse
from _typeshed import Incomplete
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.entrypoints.utils import log_version_and_model as log_version_and_model
from vllm.logger import init_logger as init_logger
from vllm.usage.usage_lib import UsageContext as UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLM

logger: Incomplete

async def serve_grpc(args: argparse.Namespace): ...
def main() -> None: ...
