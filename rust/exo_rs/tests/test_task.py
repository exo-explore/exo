import asyncio
import os
import uuid

import pytest
from exo_rs import SessionHandle

ZENOH_PORT = 52416
DISCOVERY_PORT = 52413


@pytest.fixture(scope="module")
def session_handle():
    node_id = os.urandom(16).hex().lstrip("0")

    session_handle, _nh = SessionHandle.new(
        node_id,
        ZENOH_PORT,
        DISCOVERY_PORT,
    )

    return session_handle


@pytest.mark.asyncio
async def test_task_requester_responder_round_trip(session_handle):
    instance_id = f"tests-task-instance-{uuid.uuid4().hex}"
    command_id = f"tests-task-command-{uuid.uuid4().hex}"
    command = '{"kind":"command"}'
    chunk = '{"kind":"chunk","finish_reason":"stop"}'

    requester = session_handle.task_requester()
    responder = session_handle.task_responder(instance_id)

    async def respond_to_submission():
        received = await responder.recv()
        assert received is not None
        request, chunk_sender, payload = received
        assert payload == command
        request.reply(command_id)
        await chunk_sender.send(chunk)

    stream, _ = await asyncio.gather(
        requester.submit(instance_id, command_id, command),
        respond_to_submission(),
    )

    assert await stream.recv() == chunk


@pytest.mark.asyncio
async def test_task_requester_interrupt_round_trip(session_handle):
    instance_id = f"tests-task-instance-{uuid.uuid4().hex}"
    command_id = f"tests-task-command-{uuid.uuid4().hex}"
    command = '{"kind":"interrupt"}'

    requester = session_handle.task_requester()
    responder = session_handle.task_responder(instance_id)

    async def respond_to_interrupt():
        received = await responder.recv()
        assert received is not None
        request, _chunk_sender, payload = received
        assert payload == command
        request.reply(command_id)

    await asyncio.gather(
        requester.interrupt(instance_id, command_id, command),
        respond_to_interrupt(),
    )
