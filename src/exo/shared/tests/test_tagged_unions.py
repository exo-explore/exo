"""Events and chunks are validated by their tag, not by trying every member of their union.

A follower parses every event of the cluster, and every token of every reply is one. Each member
of the Event union is a TaggedModel with a Python validator, and pydantic tried the members in
turn: parsing a token's message took about ten times as long as parsing its event as its own
type, enough for a node streaming replies to fall behind the cluster under load. The bytes on the
wire are the same either way.
"""

import time
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from exo.routing.topics import GLOBAL_EVENTS
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    NodeTimedOut,
    TestEvent,
)

MODEL = ModelId("test-model")
SESSION = SessionId(master_node_id=NodeId("master"), election_clock=1)


def token() -> ChunkGenerated:
    return ChunkGenerated(
        command_id=CommandId(),
        chunk=TokenChunk(model=MODEL, text="hello", token_id=1, usage=None),
    )


def message(event: Event) -> bytes:
    return GLOBAL_EVENTS.serialize(
        GlobalForwarderEvent(
            origin_idx=7, origin=NodeId("master"), session=SESSION, event=event
        )
    )


def test_an_event_is_checked_only_against_the_types_its_tags_name() -> None:
    """A message whose token is broken is checked against ChunkGenerated and TokenChunk only:
    one error, at the field. Trying every member of the unions gave 70."""
    data = message(token()).replace(b'"text":"hello"', b'"text":7')

    with pytest.raises(ValidationError) as raised:
        GLOBAL_EVENTS.deserialize(data)

    assert [error["loc"] for error in raised.value.errors()] == [
        ("event", "ChunkGenerated", "chunk", "TokenChunk", "text")
    ]


class TokenMessage(GlobalForwarderEvent):
    """A GLOBAL_EVENTS message whose event is known to be a ChunkGenerated: the same bytes, read
    without looking anything up in the Event union."""

    event: ChunkGenerated  # pyright: ignore[reportIncompatibleVariableOverride]


def test_parsing_a_token_costs_about_what_parsing_it_as_its_own_type_does() -> None:
    """The same message parsed through the Event union, and with its event's type known, in the
    same process at the same time, so a busy machine slows both alike. It took several times
    as long through the union."""
    data = message(token())

    def seconds(parse: Callable[[], object]) -> float:
        began = time.perf_counter()
        for _ in range(200):
            parse()
        return time.perf_counter() - began

    through_union: list[float] = []
    as_its_type: list[float] = []
    for _ in range(7):
        through_union.append(seconds(lambda: GLOBAL_EVENTS.deserialize(data)))
        as_its_type.append(seconds(lambda: TokenMessage.model_validate_json(data)))

    assert min(through_union) < 3 * min(as_its_type)


@pytest.mark.parametrize(
    "event",
    [
        TestEvent(),
        NodeTimedOut(node_id=NodeId("gone")),
        token(),
        ChunkGenerated(
            command_id=CommandId(), chunk=ErrorChunk(model=MODEL, error_message="no")
        ),
        ChunkGenerated(
            command_id=CommandId(),
            chunk=ToolCallChunk(model=MODEL, tool_calls=[], usage=None),
        ),
        ChunkGenerated(
            command_id=CommandId(),
            chunk=PrefillProgressChunk(model=MODEL, processed_tokens=1, total_tokens=2),
        ),
    ],
    ids=[
        "TestEvent",
        "NodeTimedOut",
        "TokenChunk",
        "ErrorChunk",
        "ToolCallChunk",
        "PrefillProgressChunk",
    ],
)
def test_an_event_comes_back_from_its_bytes_as_it_was(event: Event) -> None:
    """The serialized form is the class name over the fields, as before, so nodes of either
    version read each other's messages."""
    data = message(event)
    assert data.decode().count(f'"{type(event).__name__}":') == 1
    parsed = GLOBAL_EVENTS.deserialize(data)
    assert parsed.event == event
    assert GLOBAL_EVENTS.serialize(parsed) == data


def test_an_event_object_is_validated_as_its_own_class() -> None:
    event = token()
    assert IndexedEvent(idx=0, event=event).event is event


def test_an_unknown_tag_is_refused_by_name() -> None:
    data = message(TestEvent()).replace(b'"TestEvent"', b'"EventFromANewerVersion"')
    with pytest.raises(ValidationError, match="EventFromANewerVersion"):
        GLOBAL_EVENTS.deserialize(data)
