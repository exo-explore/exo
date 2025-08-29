import asyncio
import json
import struct
import sys
import traceback
from typing import Any, BinaryIO, Dict, Tuple, Union, cast

from loguru import logger

from exo.shared.types.worker.commands_runner import (
    ErrorResponse,
    PrintResponse,
    RunnerMessage,
    RunnerMessageTypeAdapter,
    RunnerResponse,
    RunnerResponseType,
    RunnerResponseTypeAdapter,
)

### Utils - SAFE LENGTH READ/WRITE

MAGIC = b"EXO1"
HDR_FMT = "!I"  # 4-byte big-endian length


async def write_frame(stream: Union[asyncio.StreamWriter, Any], obj: Union[Dict[str, Any], bytes]) -> None:
    """Write a length-prefixed frame to a stream."""
    payload = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
    header = MAGIC + struct.pack(HDR_FMT, len(payload))
    stream.write(header + payload)
    if hasattr(stream, 'drain'):
        await stream.drain()


async def read_frame(stream: Union[asyncio.StreamReader, Any]) -> Dict[str, Any]:
    """Read a length-prefixed frame from a stream."""
    # Read 8 bytes: 4-byte magic + 4-byte length
    header: bytes = await stream.readexactly(8)
    if header[:4] != MAGIC:
        # Fallback to legacy newline mode for backward compatibility
        # Reconstruct the partial line and read the rest
        remaining: bytes = await stream.readline()
        line = header + remaining
        return cast(Dict[str, Any], json.loads(line.strip().decode('utf-8')))
    
    (length,) = cast(Tuple[int], struct.unpack(HDR_FMT, header[4:]))
    data: bytes = await stream.readexactly(length)
    return cast(Dict[str, Any], json.loads(data.decode('utf-8')))


def write_frame_sync(stream: BinaryIO, obj: Union[Dict[str, Any], bytes]) -> None:
    """Synchronous version of write_frame for use in runner."""
    payload = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
    header = MAGIC + struct.pack(HDR_FMT, len(payload))
    stream.write(header + payload)
    stream.flush()


def read_frame_sync(stream: BinaryIO) -> Dict[str, Any]:
    """Synchronous version of read_frame for use in runner."""
    # Read 8 bytes: 4-byte magic + 4-byte length
    header: bytes = stream.read(8)
    if not header or len(header) < 8:
        raise EOFError("No more data to read")
    
    if header[:4] != MAGIC:
        # Fallback to legacy newline mode for backward compatibility
        # Reconstruct the partial line and read the rest
        remaining: bytes = stream.readline()
        if not remaining:
            raise EOFError("No more data to read")
        line = header + remaining
        return cast(Dict[str, Any], json.loads(line.strip().decode('utf-8')))
    
    (length,) = cast(Tuple[int], struct.unpack(HDR_FMT, header[4:]))
    data: bytes = stream.read(length)
    if len(data) < length:
        raise EOFError(f"Expected {length} bytes, got {len(data)}")
    return cast(Dict[str, Any], json.loads(data.decode('utf-8')))



### Utils - MESSAGE TO RUNNER

async def supervisor_write_message(
    proc: asyncio.subprocess.Process, message: RunnerMessage
) -> None:
    assert proc.stdin is not None, (
        "proc.stdin should not be None when created with stdin=PIPE"
    )

    # Use model_dump_json to get proper JSON encoding for Pydantic types like IPv4Address
    await write_frame(proc.stdin, message.model_dump_json().encode('utf-8'))


async def runner_read_message() -> RunnerMessage:
    loop = asyncio.get_running_loop()
    
    # Use executor to avoid blocking the event loop
    data: Dict[str, Any] = await loop.run_in_executor(None, read_frame_sync, sys.stdin.buffer)
    
    try:
        return RunnerMessageTypeAdapter.validate_python(data)
    except Exception as e:
        raise ValueError(f"Error validating message: {data}") from e


### Utils - RESPONSE FROM RUNNER

def runner_write_response(obj: RunnerResponse) -> None:
    try:
        # Use model_dump_json to get proper JSON encoding
        write_frame_sync(sys.stdout.buffer, obj.model_dump_json().encode('utf-8'))
    except BrokenPipeError:
        # Supervisor has closed the pipe, silently exit
        sys.exit(0)


async def supervisor_read_response(
    proc: asyncio.subprocess.Process,
) -> RunnerResponse:
    assert proc.stdout is not None, (
        "proc.stdout should not be None when created with stdout=PIPE"
    )
    
    data: Dict[str, Any]
    try:
        data = await read_frame(proc.stdout)
        return RunnerResponseTypeAdapter.validate_python(data)
    except EOFError:
        raise EOFError('No more data to read when reading response from runner.') from None
    except Exception as err:
        raise ValueError(f"Error validating response: {err}") from err


### Utils - Runner Prints


def runner_print(text: str) -> None:
    obj = PrintResponse(
        type=RunnerResponseType.PrintResponse,
        text=text,
    )

    runner_write_response(obj)


def runner_write_error(error: Exception) -> None:
    # Skip writing error if it's a BrokenPipeError - supervisor is already gone
    if isinstance(error, BrokenPipeError):
        sys.exit(0)

    error_response: ErrorResponse = ErrorResponse(
        type=RunnerResponseType.ErrorResponse,
        error_type=type(error).__name__,
        error_message=str(error),
        traceback=traceback.format_exc(),
    )
    runner_write_response(error_response)
    logger.opt(exception=error).exception("Critical Runner error")
