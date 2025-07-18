import asyncio
import sys
import traceback

from shared.types.worker.commands_runner import (
    ErrorResponse,
    PrintResponse,
    RunnerMessage,
    RunnerMessageTypeAdapter,
    RunnerResponse,
    RunnerResponseType,
    RunnerResponseTypeAdapter,
)

### Utils - MESSAGE TO RUNNER


async def supervisor_write_message(
    proc: asyncio.subprocess.Process, message: RunnerMessage
) -> None:
    assert proc.stdin is not None, (
        "proc.stdin should not be None when created with stdin=PIPE"
    )

    encoded: bytes = message.model_dump_json().encode("utf-8") + b"\n"
    print(f"message: {message}")
    # print(f"encoded: {encoded}")
    proc.stdin.write(encoded)
    await proc.stdin.drain()


async def runner_read_message() -> RunnerMessage:
    loop = asyncio.get_running_loop()

    line: bytes = await loop.run_in_executor(None, sys.stdin.buffer.readline)
    if not line: # This seems to be what triggers when we don't clean up the runner neatly and leave the process dangling.
        raise EOFError("No more data to read")
    line = line.strip()

    try:
        return RunnerMessageTypeAdapter.validate_json(line)
    except Exception as e:
        raise ValueError(f"Error validating message: {line}") from e


### Utils - RESPONSE FROM RUNNER


def runner_write_response(obj: RunnerResponse) -> None:
    encoded: bytes = obj.model_dump_json().encode("utf-8") + b"\n"
    _ = sys.stdout.buffer.write(encoded)
    _ = sys.stdout.buffer.flush()


async def supervisor_read_response(
    proc: asyncio.subprocess.Process,
) -> RunnerResponse | None:
    assert proc.stdout is not None, (
        "proc.stdout should not be None when created with stdout=PIPE"
    )
    line_bytes: bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    line: str = line_bytes.decode("utf-8").strip()

    if not line:
        raise EOFError("No more data to read")

    try:
        return RunnerResponseTypeAdapter.validate_json(line)
    except Exception as err:
        raise ValueError(f"Error validating response: {line}") from err


### Utils - Runner Prints


def runner_print(text: str) -> None:
    obj = PrintResponse(
        type=RunnerResponseType.PrintResponse,
        text=text,
    )

    runner_write_response(obj)


def runner_write_error(error: Exception) -> None:
    error_response: ErrorResponse = ErrorResponse(
        type=RunnerResponseType.ErrorResponse,
        error_type=type(error).__name__,
        error_message=str(error),
        traceback=traceback.format_exc(),
    )
    runner_write_response(error_response)
