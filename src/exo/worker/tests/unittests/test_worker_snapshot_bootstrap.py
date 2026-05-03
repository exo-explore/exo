# pyright: reportPrivateUsage=false

import hashlib

import anyio
import pytest
import zstandard

from exo.routing.event_router import EventRouter
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestSnapshot,
)
from exo.shared.types.common import CommandId, ModelId, NodeId, SessionId
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
from exo.worker.main import Worker


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


def _worker(
    node_id: NodeId, session_id: SessionId
) -> tuple[
    Worker,
    EventRouter,
    Receiver[ForwarderCommand],
    Sender[SnapshotChunk],
    Sender[TransientEvent],
    Sender[IndexedEvent],
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
    local_event_output_sender, _local_event_output_receiver = channel[Event]()
    transient_sender, transient_receiver = channel[TransientEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_command_sender, _download_command_receiver = channel[
        ForwarderDownloadCommand
    ]()
    snapshot_sender, snapshot_receiver = channel[SnapshotChunk]()

    worker = Worker(
        node_id,
        session_id,
        event_router=event_router,
        event_receiver=event_receiver,
        event_sender=local_event_output_sender,
        transient_event_receiver=transient_receiver,
        transient_event_sender=transient_sender,
        snapshot_chunk_receiver=snapshot_receiver,
        command_sender=command_sender,
        download_command_sender=download_command_sender,
        api_port=52415,
    )
    return (
        worker,
        event_router,
        command_receiver,
        snapshot_sender,
        transient_sender,
        event_sender,
    )


@pytest.mark.asyncio
async def test_worker_fetch_snapshot_applies_state_and_fast_forwards_router() -> None:
    node_id = NodeId("worker")
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=1)
    (
        worker,
        event_router,
        command_receiver,
        snapshot_sender,
        _transient_sender,
        _event_sender,
    ) = _worker(node_id, session_id)
    state = State(last_event_applied_idx=7)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker._fetch_snapshot)
        command = await command_receiver.receive()
        assert isinstance(command.command, RequestSnapshot)
        assert command.command.requester_node_id == node_id

        await snapshot_sender.send(
            _snapshot_chunk(state, requester_node_id=node_id, session_id=session_id)
        )

    assert worker.state.last_event_applied_idx == 7
    assert event_router.event_buffer.next_idx_to_release == 8


@pytest.mark.asyncio
async def test_worker_event_applier_ignores_events_covered_by_snapshot() -> None:
    node_id = NodeId("worker")
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=1)
    (
        worker,
        _event_router,
        _command_receiver,
        _snapshot_sender,
        _transient_sender,
        event_sender,
    ) = _worker(node_id, session_id)
    worker.state = State(last_event_applied_idx=7)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker._event_applier)
        await event_sender.send(IndexedEvent(idx=7, event=TestEvent()))
        await event_sender.send(IndexedEvent(idx=8, event=TestEvent()))

        while worker.state.last_event_applied_idx != 8:
            await anyio.sleep(0.001)
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_worker_transient_handler_drains_ignored_events() -> None:
    node_id = NodeId("worker")
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=1)
    (
        worker,
        _event_router,
        _command_receiver,
        _snapshot_sender,
        transient_sender,
        _event_sender,
    ) = _worker(node_id, session_id)
    done = anyio.Event()

    async def run_handler() -> None:
        await worker._transient_event_handler()
        done.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_handler)
        await transient_sender.send(
            ChunkGenerated(
                command_id=CommandId("cmd-a"),
                chunk=ErrorChunk(
                    model=ModelId("test-model"),
                    error_message="ignored transient",
                ),
            )
        )
        transient_sender.close()

        with anyio.fail_after(1):
            await done.wait()
        tg.cancel_scope.cancel()
