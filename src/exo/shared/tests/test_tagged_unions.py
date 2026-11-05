"""Events and serve frames are validated by their tag, not by trying every member of their union.

A follower parses every event of the cluster, and an API node every line of the replies it
streams, one per token. Each member of these unions is a TaggedModel with a Python validator,
and pydantic tried the members in turn: parsing a message took about ten times as long as
parsing it as its own type. The bytes on the wire are the same either way.
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
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    NodeTimedOut,
    TestEvent,
)
from exo.shared.types.serving import Accepted, Refused, ServeFrame, serve_frames

MODEL = ModelId("test-model")
SESSION = SessionId(master_node_id=NodeId("master"), election_clock=1)


def token() -> TokenChunk:
    return TokenChunk(model=MODEL, text="hello", token_id=1, usage=None)


def message(event: Event) -> bytes:
    return GLOBAL_EVENTS.serialize(
        GlobalForwarderEvent(
            origin_idx=7, origin=NodeId("master"), session=SESSION, event=event
        )
    )


def frame(value: ServeFrame) -> bytes:
    return serve_frames.dump_json(value)


def test_an_event_is_checked_only_against_the_type_its_tag_names() -> None:
    """A message whose event is broken is checked against that event's type only: one error,
    at the field."""
    data = message(NodeTimedOut(node_id=NodeId("gone"))).replace(
        b'"node_id":"gone"', b'"node_id":7'
    )

    with pytest.raises(ValidationError) as raised:
        GLOBAL_EVENTS.deserialize(data)

    assert [error["loc"] for error in raised.value.errors()] == [
        ("event", "NodeTimedOut", "node_id")
    ]


def test_a_serve_frame_is_checked_only_against_the_type_its_tag_names() -> None:
    data = frame(token()).replace(b'"text":"hello"', b'"text":7')

    with pytest.raises(ValidationError) as raised:
        serve_frames.validate_json(data)

    assert [error["loc"] for error in raised.value.errors()] == [("TokenChunk", "text")]


def test_parsing_a_token_costs_about_what_parsing_it_as_its_own_type_does() -> None:
    """The same line parsed through the ServeFrame union, and as a TokenChunk, in the same
    process at the same time, so a busy machine slows both alike."""
    data = frame(token())

    def seconds(parse: Callable[[], object]) -> float:
        began = time.perf_counter()
        for _ in range(200):
            parse()
        return time.perf_counter() - began

    through_union: list[float] = []
    as_its_type: list[float] = []
    for _ in range(7):
        through_union.append(seconds(lambda: serve_frames.validate_json(data)))
        as_its_type.append(seconds(lambda: TokenChunk.model_validate_json(data)))

    assert min(through_union) < 3 * min(as_its_type)


@pytest.mark.parametrize(
    "event",
    [TestEvent(), NodeTimedOut(node_id=NodeId("gone"))],
    ids=["TestEvent", "NodeTimedOut"],
)
def test_an_event_comes_back_from_its_bytes_as_it_was(event: Event) -> None:
    """The serialized form is the class name over the fields, as before, so nodes of either
    version read each other's messages."""
    data = message(event)
    assert data.decode().count(f'"{type(event).__name__}":') == 1
    parsed = GLOBAL_EVENTS.deserialize(data)
    assert parsed.event == event
    assert GLOBAL_EVENTS.serialize(parsed) == data


@pytest.mark.parametrize(
    "value",
    [
        Accepted(),
        Refused(message="no"),
        token(),
        ErrorChunk(model=MODEL, error_message="no"),
        ToolCallChunk(model=MODEL, tool_calls=[], usage=None),
        PrefillProgressChunk(model=MODEL, processed_tokens=1, total_tokens=2),
    ],
    ids=[
        "Accepted",
        "Refused",
        "TokenChunk",
        "ErrorChunk",
        "ToolCallChunk",
        "PrefillProgressChunk",
    ],
)
def test_a_serve_frame_comes_back_from_its_bytes_as_it_was(value: ServeFrame) -> None:
    data = frame(value)
    assert data.decode().count(f'"{type(value).__name__}":') == 1
    parsed = serve_frames.validate_json(data)
    assert parsed == value
    assert frame(parsed) == data


def test_an_event_object_is_validated_as_its_own_class() -> None:
    event = NodeTimedOut(node_id=NodeId("gone"))
    assert IndexedEvent(idx=0, event=event).event is event


def test_an_unknown_tag_is_refused_by_name() -> None:
    data = message(TestEvent()).replace(b'"TestEvent"', b'"EventFromANewerVersion"')
    with pytest.raises(ValidationError, match="EventFromANewerVersion"):
        GLOBAL_EVENTS.deserialize(data)
