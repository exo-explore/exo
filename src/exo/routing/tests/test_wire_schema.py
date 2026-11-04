"""What nodes send each other may only change together with WIRE_VERSION (exo/routing/wire.py)."""

from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_serializer,
    field_validator,
)

from exo.routing import topics
from exo.routing import wire_schema as wire_schema_module
from exo.routing.wire import WIRE_VERSION
from exo.routing.wire_schema import (
    LedgerEntry,
    changes,
    dump_ledger,
    load_ledger,
    load_snapshot,
    network_topics,
    schema_hash,
    schema_of,
    wire_schema,
)
from exo.utils.pydantic_ext import FrozenModel, TaggedModel

# The checkout's Cargo.lock, found from this file: exo itself may be installed elsewhere, as in
# CI's test environment, where it is in the nix store without Cargo.lock
CARGO_LOCK = Path(__file__).parents[4] / "Cargo.lock"

_WHAT_TO_DO = (
    "Nodes on the old schema can't read the new one, so they must not connect. Set "
    "WIRE_VERSION in src/exo/routing/wire.py to the last version in "
    "src/exo/routing/wire_versions.txt plus 1, then run\n"
    "  uv run python -m exo.routing.wire_schema --write\n"
    "Devices then join only devices on the same version. Take the next version even when "
    "the bytes on the wire didn't change (a pydantic upgrade renders schemas differently, a "
    "model that is never sent by name was renamed): a spare version costs nothing."
)


def test_the_wire_schema_matches_its_snapshot() -> None:
    saved = load_snapshot()
    current = wire_schema(CARGO_LOCK)
    if current != saved:
        pytest.fail(
            "What exo nodes send each other changed:\n"
            + "".join(f"  {line}\n" for line in changes(saved, current))
            + _WHAT_TO_DO,
            pytrace=False,
        )


def test_the_ledger_ends_with_this_wire_version_and_schema() -> None:
    ledger = load_ledger()
    versions = [entry.wire_version for entry in ledger]
    # Append only: a version that two changes both took shows up twice, or out of order
    assert versions == list(range(1, len(versions) + 1)), (
        f"wire_versions.txt must list each version once, in order, from 1: {versions}. "
        "Two changes took the same version: give the later one the next version."
    )
    assert ledger[-1].wire_version == WIRE_VERSION, (
        f"WIRE_VERSION is {WIRE_VERSION} but wire_versions.txt ends at "
        f"{ledger[-1].wire_version}: run uv run python -m exo.routing.wire_schema --write"
    )
    if schema_hash(wire_schema(CARGO_LOCK)) != ledger[-1].schema_hash:
        pytest.fail(
            f"The wire schema isn't the one recorded for wire version {WIRE_VERSION}. "
            + _WHAT_TO_DO,
            pytrace=False,
        )


def test_writing_adds_the_new_versions_line_and_refuses_without_a_bump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, snapshot = tmp_path / "wire_versions.txt", tmp_path / "wire_schema.json"
    monkeypatch.setattr(wire_schema_module, "LEDGER_PATH", ledger)
    monkeypatch.setattr(wire_schema_module, "SNAPSHOT_PATH", snapshot)
    # Version 1 had another schema
    ledger.write_text(dump_ledger([LedgerEntry(1, "0" * 16)]))
    snapshot.write_text("{}")

    monkeypatch.setattr(wire_schema_module, "WIRE_VERSION", 1)
    assert wire_schema_module.main(["--write"], CARGO_LOCK) == 1
    monkeypatch.setattr(wire_schema_module, "WIRE_VERSION", 3)
    assert wire_schema_module.main(["--write"], CARGO_LOCK) == 1, (
        "versions go up one at a time"
    )
    assert load_ledger(ledger) == [LedgerEntry(1, "0" * 16)]

    monkeypatch.setattr(wire_schema_module, "WIRE_VERSION", 2)
    assert wire_schema_module.main(["--write"], CARGO_LOCK) == 0
    assert load_ledger(ledger) == [
        LedgerEntry(1, "0" * 16),
        LedgerEntry(2, schema_hash(wire_schema(CARGO_LOCK))),
    ]
    assert wire_schema_module.main([], CARGO_LOCK) == 0


def test_no_version_is_ever_rewritten() -> None:
    # There is no flag to save a changed schema under the version it already has: settling
    # a ledger conflict that way would re-point a version that may be on main already
    with pytest.raises(SystemExit):
        wire_schema_module.main(["--write", "--keep-version"])


def test_every_topic_nodes_send_is_in_the_schema() -> None:
    assert set(network_topics()) == {
        topics.GLOBAL_EVENTS.topic,
        topics.LOCAL_EVENTS.topic,
        topics.STATE_SNAPSHOTS.topic,
        topics.COMMANDS.topic,
        topics.ELECTION_MESSAGES.topic,
        topics.DOWNLOAD_COMMANDS.topic,
    }
    # Membership comes from this node's own networking, never from the network
    assert topics.MEMBERSHIP.topic not in network_topics()


def _model(
    name: str, base: type[BaseModel], doc: str = "", **fields: object
) -> type[BaseModel]:
    return create_model(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        name,
        __base__=base,
        __doc__=doc,
        **fields,  # pyright: ignore[reportArgumentType]
    )


def _message(doc: str, **fields: object) -> type[BaseModel]:
    return _model("Message", FrozenModel, doc, **fields)


def test_documentation_is_not_part_of_the_wire() -> None:
    before = _message("One thing.", count=(int, Field(description="how many")))
    after = _message("Another thing.", count=(int, Field(description="how much")))
    assert schema_of({"t": before}) == schema_of({"t": after})


def test_a_new_field_changes_the_wire_even_with_a_default() -> None:
    # Old nodes forbid fields they don't know, and every field is always sent
    before = _message("", count=(int, ...))
    after = _message("", count=(int, ...), label=(str, ""))
    assert schema_of({"t": before}) != schema_of({"t": after})


def test_a_new_constraint_changes_the_wire() -> None:
    before = _message("", count=(int, ...))
    after = _message("", count=(int, Field(ge=0)))
    assert schema_of({"t": before}) != schema_of({"t": after})


def test_fields_are_named_as_they_are_sent() -> None:
    message = _message("", origin_index=(int, ...), title=(str, ...))
    sent = message.model_validate({"origin_index": 1, "title": "x"}).model_dump_json()
    assert '"origin_index"' in sent
    properties = schema_of({"t": message})["$defs"]["Message"]["properties"]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
    # A field called like a schema keyword is still a field
    assert set(properties) == {"origin_index", "title"}  # pyright: ignore[reportUnknownArgumentType]


def _event_holder(tag: str) -> type[BaseModel]:
    event = _model(tag, TaggedModel, value=(int, ...))
    return _message("", event=(event, ...))


def test_a_tagged_model_is_sent_under_its_class_name() -> None:
    holder = _event_holder("Ping")
    sent = holder.model_validate({"event": {"Ping": {"value": 1}}}).model_dump_json()
    assert sent == '{"event":{"Ping":{"value":1}}}'
    schema = schema_of({"t": holder})
    assert schema["$defs"]["Ping"]["required"] == ["Ping"]  # pyright: ignore[reportIndexIssue]
    # So renaming it changes the wire
    assert schema_of({"t": _event_holder("Pong")}) != schema


def test_changes_name_what_differs() -> None:
    before = schema_of({"t": _message("", count=(int, ...))})
    after = schema_of({"t": _message("", count=(str, ...))})
    assert changes(before, after) == ["changed Message"]
    assert changes(before, before) == []


def _at_most_one() -> type[BaseModel]:
    class Message(FrozenModel):
        count: int

        @field_validator("count")
        @classmethod
        def _at_most(cls, value: int) -> int:
            if value > 1:
                raise ValueError("too many")
            return value

    return Message


def _at_most_one_documented() -> type[BaseModel]:
    class Message(FrozenModel):
        count: int

        @field_validator("count")
        @classmethod
        def _at_most(cls, value: int) -> int:
            """At most one."""
            if value > 1:  # the same check, with a comment
                raise ValueError("too many")
            return value

    return Message


def _at_most_two() -> type[BaseModel]:
    class Message(FrozenModel):
        count: int

        @field_validator("count")
        @classmethod
        def _at_most(cls, value: int) -> int:
            if value > 2:
                raise ValueError("too many")
            return value

    return Message


def _strict_message(strict: bool) -> type[BaseModel]:
    class Message(BaseModel):
        model_config = ConfigDict(strict=strict, extra="forbid")
        count: int

    return Message


def _extra_message(extra: Literal["forbid", "ignore"]) -> type[BaseModel]:
    class Message(BaseModel):
        model_config = ConfigDict(strict=True, extra=extra)
        count: int

    return Message


def _sent_doubled() -> type[BaseModel]:
    class Message(FrozenModel):
        count: int

        @field_serializer("count")
        def _send(self, value: int) -> int:
            return value * 2

    return Message


def _sent_as_is() -> type[BaseModel]:
    class Message(FrozenModel):
        count: int

        @field_serializer("count")
        def _send(self, value: int) -> int:
            return value

    return Message


def _code_hashes(model: type[BaseModel]) -> list[str]:
    """The hashes of a message's validators and serializers. (The test's variants live in
    differently named functions, so their names differ; real ones keep their names.)"""
    return list(cast(dict[str, str], schema_of({"t": model})["code"]).values())


def test_a_changed_validator_changes_the_wire() -> None:
    # The receiver would reject what the sender sends: JSON schema can't show it
    assert _code_hashes(_at_most_one()) != _code_hashes(_at_most_two())


def test_a_validators_comments_and_docstring_are_not_the_wire() -> None:
    assert _code_hashes(_at_most_one()) == _code_hashes(_at_most_one_documented())


def test_strictness_and_extra_fields_are_part_of_the_wire() -> None:
    assert schema_of({"t": _strict_message(True)}) != schema_of(
        {"t": _strict_message(False)}
    )
    assert schema_of({"t": _extra_message("forbid")}) != schema_of(
        {"t": _extra_message("ignore")}
    )


def test_a_changed_serializer_changes_the_wire() -> None:
    assert _code_hashes(_sent_doubled()) != _code_hashes(_sent_as_is())


def test_the_order_fields_are_declared_in_is_not_the_wire() -> None:
    first = _message("", count=(int, ...), label=(str, ...))
    second = _message("", label=(str, ...), count=(int, ...))
    assert schema_of({"t": first}) == schema_of({"t": second})


def test_the_snapshot_covers_the_encoding_the_task_gather_and_zenoh() -> None:
    schema = wire_schema(CARGO_LOCK)
    assert set(cast(dict[str, str], schema["encoding"])) == {
        "TypedTopic.serialize",
        "TypedTopic.deserialize",
    }
    assert set(cast(dict[str, str], schema["collectives"])) == {"mx_all_gather_tasks"}
    zenoh_version, revision = cast(str, schema["zenoh"]).split()
    assert zenoh_version.count(".") == 2 and revision
    # The validators that real messages run are recorded
    code = cast(dict[str, str], schema["code"])
    assert any(
        name.endswith("ShardAssignments.validate_runners_exist") for name in code
    )
    assert any(name.endswith("State._deserialize_topology") for name in code)
