import asyncio
import traceback

from loguru import logger

from exo.shared.global_conn import AsyncConnection, get_conn
from exo.shared.types.worker.commands_runner import (
    ErrorResponse,
    PrintResponse,
    RunnerMessage,
    RunnerResponse,
)

### Utils - Runner Prints


def runner_print(text: str) -> None:
    obj = PrintResponse(
        text=text,
    )

    conn: AsyncConnection[RunnerResponse, RunnerMessage] = get_conn()
    conn.send_sync(obj)


def runner_write_error(error: Exception) -> None:
    error_response: ErrorResponse = ErrorResponse(
        error_type=type(error).__name__,
        error_message=str(error),
        traceback=traceback.format_exc(),
    )

    conn = get_conn()
    asyncio.create_task(conn.send(error_response))
    logger.opt(exception=error).exception("Critical Runner error")


## TODO: To make this cleaner, it seems like we should have only one writer.
# This is fine in runner_supervisor but there's a risk in runner.py that we overlap things
# We can guarantee this by enqueueing messages and have a writing thread.
