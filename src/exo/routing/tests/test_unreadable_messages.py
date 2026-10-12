"""A message a node can't read is dropped instead of stopping the node's receive loop."""

import pytest

from exo.routing.router import TopicRouter
from exo.routing.topics import LOCAL_EVENTS
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import ChunkGenerated, LocalForwarderEvent
from exo.utils.channels import Receiver, channel
from exo.worker.runner.diagnostics import (
    KnownRunnerDiagnostic,
    RunnerMetalGpuTimeout,
    RunnerRingSocketReceivingError,
    RunnerRingTransportError,
)


def failed_request(diagnostic: KnownRunnerDiagnostic) -> LocalForwarderEvent:
    return LocalForwarderEvent(
        origin=SystemId(),
        origin_idx=0,
        session=SessionId(master_node_id=NodeId("master"), election_clock=0),
        event=ChunkGenerated(
            command_id=CommandId(),
            chunk=ErrorChunk(
                model=ModelId("test-model"),
                error_message="Runner shutdown before completing command",
                diagnostics=[diagnostic],
            ),
        ),
    )


def router() -> tuple[TopicRouter[LocalForwarderEvent], Receiver[LocalForwarderEvent]]:
    networking_sender, _ = channel[tuple[str, bytes]]()
    topic_router = TopicRouter[LocalForwarderEvent](LOCAL_EVENTS, networking_sender)
    subscriber, received = channel[LocalForwarderEvent]()
    topic_router.senders.add(subscriber)
    return topic_router, received


@pytest.mark.parametrize(
    "diagnostic",
    [
        RunnerRingTransportError(message="aborted", evidence=("a", "b")),
        RunnerRingSocketReceivingError(
            message="failed",
            evidence=("c",),
            error_number=54,
            error_name="ECONNRESET",
            error_description="Connection reset by peer",
        ),
        RunnerMetalGpuTimeout(message="timeout"),
    ],
)
async def test_a_failed_request_with_diagnostics_reaches_other_nodes(
    diagnostic: KnownRunnerDiagnostic,
) -> None:
    sent = failed_request(diagnostic)
    topic_router, received = router()

    await topic_router.publish_bytes(LOCAL_EVENTS.serialize(sent))

    assert received.collect() == [sent]


@pytest.mark.parametrize(
    "data",
    [b"not json", b'{"origin": "x"}', "\xff".encode("latin-1")],
)
async def test_an_unreadable_message_is_dropped(data: bytes) -> None:
    topic_router, received = router()

    await topic_router.publish_bytes(data)

    assert received.collect() == []
