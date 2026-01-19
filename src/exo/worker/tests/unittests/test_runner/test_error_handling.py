# pyright: reportAny=false
from unittest.mock import MagicMock

from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import ChunkGenerated
from exo.worker.runner.runner import send_error_chunk_on_exception
from exo.worker.tests.constants import MODEL_A_ID


def test_send_error_chunk_on_exception_no_error() -> None:
    event_sender = MagicMock()
    command_id = CommandId()

    with send_error_chunk_on_exception(
        event_sender, command_id, MODEL_A_ID, device_rank=0
    ):
        _ = 1 + 1

    event_sender.send.assert_not_called()


def test_send_error_chunk_on_exception_catches_error() -> None:
    event_sender = MagicMock()
    command_id = CommandId()

    with send_error_chunk_on_exception(
        event_sender, command_id, MODEL_A_ID, device_rank=0
    ):
        raise ValueError("test error")

    event_sender.send.assert_called_once()
    call_args = event_sender.send.call_args[0][0]
    assert isinstance(call_args, ChunkGenerated)
    assert call_args.command_id == command_id
    assert isinstance(call_args.chunk, TokenChunk)
    assert call_args.chunk.finish_reason == "error"
    assert call_args.chunk.error_message == "test error"


def test_send_error_chunk_on_exception_skips_non_rank_zero() -> None:
    event_sender = MagicMock()
    command_id = CommandId()

    with send_error_chunk_on_exception(
        event_sender, command_id, MODEL_A_ID, device_rank=1
    ):
        raise ValueError("test error")

    event_sender.send.assert_not_called()
