# pyright: reportPrivateUsage=false

import hashlib

import anyio
import pytest
import zstandard

from exo.api.main import API
from exo.routing.event_router import EventRouter
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.commands import ForwarderCommand, RequestSnapshot
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    TestEvent,
    TransientEvent,
)
from exo.shared.types.snapshots import SnapshotChunk, SnapshotTransferId
from exo.shared.types.state import State
from exo.utils.channels import Receiver, Sender, channel


class _FakeEventLog:
    def __init__(self) -> None:
        self.appended: list[Event] = []

    def append(self, event: Event) -> None:
        self.appended.append(event)


def _snapshot_chunk(
    state: State, *, requester_node_id: NodeId, session_id: SessionId
) -> SnapshotChunk:
    body = zstandard.ZstdCompressor().compress(state.model_dump_json().encode("utf-8"))
    return SnapshotChunk.from_data(
        data=body,
        transfer_id=SnapshotTransferId("transfer-1"),
        requester_node_id=requester_node_id,
        session_id=session_id,
        schema_version=state.schema_version,
        last_event_applied_idx=state.last_event_applied_idx,
        chunk_index=0,
        total_chunks=1,
        sha256_hex=hashlib.sha256(body).hexdigest(),
    )


def _api(
    node_id: NodeId, session_id: SessionId
) -> tuple[
    API,
    EventRouter,
    Receiver[ForwarderCommand],
    Sender[SnapshotChunk],
    Sender[TransientEvent],
    Sender[IndexedEvent],
    _FakeEventLog,
]:
    router_command_sender, _router_command_receiver = channel[ForwarderCommand]()
    _global_event_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    local_event_sender, _local_event_receiver = channel[LocalForwarderEvent]()
    event_router = EventRouter(
        session_id=session_id,
        command_sender=router_command_sender,
        external_inbound=global_event_receiver,
        external_outbound=local_event_sender,
    )

    event_sender, event_receiver = channel[IndexedEvent]()
    transient_sender, transient_receiver = channel[TransientEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    snapshot_sender, snapshot_receiver = channel[SnapshotChunk]()

    api = object.__new__(API)
    api.node_id = node_id
    api.session_id = session_id
    api.event_router = event_router
    api.event_receiver = event_receiver
    api.transient_event_receiver = transient_receiver
    api.snapshot_chunk_receiver = snapshot_receiver
    api.command_sender = command_sender
    api._system_id = SystemId("api-system")
    api.state = State()
    event_log = _FakeEventLog()
    api._event_log = event_log  # pyright: ignore[reportAttributeAccessIssue]
    api._image_generation_queues = {}
    api._text_generation_queues = {}
    return (
        api,
        event_router,
        command_receiver,
        snapshot_sender,
        transient_sender,
        event_sender,
        event_log,
    )


@pytest.mark.asyncio
async def test_api_fetch_snapshot_applies_state_and_fast_forwards_router() -> None:
    node_id = NodeId("api")
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=1)
    (
        api,
        event_router,
        command_receiver,
        snapshot_sender,
        _transient_sender,
        _event_sender,
        _event_log,
    ) = _api(node_id, session_id)
    state = State(last_event_applied_idx=7)

    async with anyio.create_task_group() as tg:
        tg.start_soon(api._fetch_snapshot)
        command = await command_receiver.receive()
        assert isinstance(command.command, RequestSnapshot)
        assert command.command.requester_node_id == node_id

        await snapshot_sender.send(
            _snapshot_chunk(state, requester_node_id=node_id, session_id=session_id)
        )

    assert api.state.last_event_applied_idx == 7
    assert event_router.event_buffer.next_idx_to_release == 8


@pytest.mark.asyncio
async def test_api_apply_state_ignores_events_covered_by_snapshot() -> None:
    node_id = NodeId("api")
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=1)
    (
        api,
        _event_router,
        _command_receiver,
        _snapshot_sender,
        _transient_sender,
        event_sender,
        event_log,
    ) = _api(node_id, session_id)
    api.state = State(last_event_applied_idx=7)

    async with anyio.create_task_group() as tg:
        tg.start_soon(api._apply_state)
        await event_sender.send(IndexedEvent(idx=7, event=TestEvent()))
        await event_sender.send(IndexedEvent(idx=8, event=TestEvent()))

        while api.state.last_event_applied_idx != 8:
            await anyio.sleep(0.001)
        tg.cancel_scope.cancel()

    assert len(event_log.appended) == 1


@pytest.mark.asyncio
async def test_api_apply_transient_dispatches_generated_chunks() -> None:
    node_id = NodeId("api")
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=1)
    (
        api,
        _event_router,
        _command_receiver,
        _snapshot_sender,
        transient_sender,
        _event_sender,
        _event_log,
    ) = _api(node_id, session_id)
    command_id = CommandId("cmd-a")
    chunk_sender, chunk_receiver = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[command_id] = chunk_sender

    chunk = ErrorChunk(model=ModelId("test-model"), error_message="test chunk")
    async with anyio.create_task_group() as tg:
        tg.start_soon(api._apply_transient)
        await transient_sender.send(ChunkGenerated(command_id=command_id, chunk=chunk))

        assert await chunk_receiver.receive() == chunk
        tg.cancel_scope.cancel()
