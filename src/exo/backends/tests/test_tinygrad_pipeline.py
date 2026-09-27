"""Pipeline hops between tinygrad ranks, in memory and over TCP."""

from __future__ import annotations

import threading
import time

from exo.backends.tinygrad_hidden_state import HiddenStateBuffer
from exo.backends.tinygrad_pipeline import (
    PipelineTokenResult,
    TcpPipelineTransport,
    connect_memory_pipeline,
)
from exo.shared.types.common import Host
from exo.utils.ports import random_ephemeral_port


def test_tcp_pipeline_round_trip() -> None:
    first_port = random_ephemeral_port()
    second_port = random_ephemeral_port()
    first = TcpPipelineTransport(
        device_rank=0,
        hosts=(
            Host(ip="0.0.0.0", port=first_port),
            Host(ip="127.0.0.1", port=second_port),
        ),
    )
    second = TcpPipelineTransport(
        device_rank=1,
        hosts=(
            Host(ip="127.0.0.1", port=first_port),
            Host(ip="0.0.0.0", port=second_port),
        ),
    )
    errors: list[BaseException] = []

    def open_second() -> None:
        try:
            second.open()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=open_second)
    thread.start()
    try:
        first.open()
        thread.join(5)
        assert not thread.is_alive()
        assert errors == []
        hidden_state = HiddenStateBuffer(
            dtype="float32",
            shape=(1, 1, 1),
            data=b"\x00\x00\x80\x3f",
        )
        first.send_hidden_state("task", hidden_state)
        assert second.receive_hidden_state("task") == hidden_state
        result = PipelineTokenResult(
            token_id=3,
            text="a",
            finish_reason="length",
            prompt_token_count=1,
            completion_token_count=1,
        )
        second.send_token_result("task", result)
        assert first.receive_token_result("task") == result
    finally:
        first.close()
        second.close()
        thread.join(5)


def test_cancel_frame_unblocks_a_waiting_receive() -> None:
    transports = connect_memory_pipeline(2)
    delivered: list[HiddenStateBuffer | None] = []
    errors: list[BaseException] = []
    started = threading.Event()

    def wait_for_hidden() -> None:
        started.set()
        try:
            delivered.append(transports[1].receive_hidden_state("task"))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=wait_for_hidden)
    thread.start()
    assert started.wait(2)
    time.sleep(0.1)
    transports[0].signal_cancellation("task")
    thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert delivered == [None]
