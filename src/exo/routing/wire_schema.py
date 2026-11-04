"""The schema of every message exo nodes send each other, its snapshot, and its ledger.

``wire_schema.json`` is the schema at ``WIRE_VERSION`` (``wire.py``), and
``wire_versions.txt`` lists every wire version with a hash of its schema, one line each,
appended to and never rewritten. ``tests/test_wire_schema.py`` fails when the schema
generated from the code differs from the snapshot, or its hash from the ledger's last line.
When it does:

1. bump ``WIRE_VERSION`` in ``src/exo/routing/wire.py`` to the ledger's last version + 1;
2. run ``uv run python -m exo.routing.wire_schema --write``, which adds that version's line.

Two changes that each take the next version both add a line at the end of the ledger, so git
makes whichever merges second take a new one, instead of letting two different schemas share
a version.

A change that leaves the bytes on the wire as they were (a pydantic upgrade that renders
schemas differently, a model that is never sent by name renamed) takes the next version too: a
spare version costs nothing, and no line is ever rewritten. Settle a conflict in the ledger the
same way, by taking the next version.

Besides each message's JSON schema (with ``required`` sorted, since field order isn't on the
wire), the snapshot records what JSON schema can't show:

- each model's ``strict`` and ``extra`` settings;
- a hash of the source of every exo validator and serializer in a message, so tightening a
  validator trips the test;
- how messages are encoded (``TypedTopic.serialize`` and ``deserialize``);
- how the ranks of an instance agree on tasks (``mx_all_gather_tasks``), read from its source
  without importing MLX;
- the zenoh revision in ``Cargo.lock``.

Discovery also compares builds (``build_id()`` in ``wire.py``), so different app releases never
connect, nor do source builds whose package, MLX or mlx-lm versions differ. Not covered, so
bump by hand when they change:

- discovery's packets (a test in ``rust/networking/src/discovery.rs`` pins them);
- the zenoh key expressions in ``rust/networking/src/swarm.rs``;
- ``GET /node_id``, which nodes call on each other to check reachability;
- between source builds of one package version: what an instance's ranks send each other
  through MLX (the activations and top-k indices in each model's sharding code, and the shape
  of what ``mx_all_gather_tasks`` gathers). A source build is named by its package and runner
  versions, not its commit, so two checkouts that differ only in that code still connect.
"""

import argparse
import ast
import hashlib
import inspect
import json
import sys
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast, final, override

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue, models_json_schema
from pydantic_core import core_schema

import exo
from exo.routing import topics
from exo.routing.topics import PublishPolicy, TypedTopic
from exo.routing.wire import WIRE_VERSION
from exo.utils.pydantic_ext import FrozenModel, TaggedModel

SNAPSHOT_PATH: Final = Path(__file__).with_name("wire_schema.json")
LEDGER_PATH: Final = Path(__file__).with_name("wire_versions.txt")
_LEDGER_HEADER: Final = """\
# exo's wire versions: one line each, the version, then a hash of its wire schema
# (src/exo/routing/wire_schema.py). Append only: two changes that each take the next version
# both add a line here, so git makes the second to merge take a new one. Never edit a line,
# to settle a conflict or otherwise: take the next version.
"""

# What a schema says that doesn't change which messages a node accepts. Defaults don't
# matter because every field is always sent (TypedTopic.serialize).
_NOT_ON_THE_WIRE = frozenset({"title", "description", "default", "examples"})
# Schema keywords whose value maps names (of fields, or of definitions) to schemas
_NAMED_SCHEMAS = frozenset({"properties", "$defs", "patternProperties"})
# Read as text: importing it needs MLX
_GATHER_SOURCE: Final = Path(exo.__file__).parent / "worker/engines/mlx/utils_mlx.py"
_GATHER_FUNCTION: Final = "mx_all_gather_tasks"
# Cargo.lock of the checkout exo runs from, as `uv sync` installs it. An installed exo has none
# (CI's test environment installs it into the nix store): the tests pass the checkout's.
CARGO_LOCK_PATH: Final = Path(exo.__file__).parents[2] / "Cargo.lock"


@final
class _WireSchemaGenerator(GenerateJsonSchema):
    """The JSON schema of messages as they are on the wire: a tagged model is sent as
    ``{"ClassName": {...}}``, so its class name is part of what a receiver must know."""

    @override
    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        inner = super().model_schema(schema)
        model = schema["cls"]
        config = cast(Mapping[str, object], getattr(model, "model_config", {}))
        # Not in JSON schema, but what a receiver accepts
        inner["x-strict"] = bool(config.get("strict", False))
        inner["x-extra"] = str(config.get("extra", "ignore"))
        if not issubclass(model, TaggedModel):
            return inner
        return {
            "type": "object",
            "properties": {model.__name__: inner},
            "required": [model.__name__],
            "additionalProperties": False,
        }


def network_topics() -> dict[str, type[BaseModel]]:
    """Every topic that nodes send each other, with its message model."""
    found: dict[str, type[BaseModel]] = {}
    for value in list[object](vars(topics).values()):
        if not isinstance(value, TypedTopic):
            continue
        topic = cast(TypedTopic[FrozenModel], value)
        if topic.publish_policy is not PublishPolicy.Never:
            found[topic.topic] = topic.model_type
    return found


def schema_of(models_by_topic: Mapping[str, type[BaseModel]]) -> dict[str, object]:
    """The wire schema of these topics: what each topic carries, and every model in it."""
    # By field name, as TypedTopic.serialize sends them (aliases are only accepted)
    roots, schema = models_json_schema(
        [(model, "serialization") for model in models_by_topic.values()],
        by_alias=False,
        schema_generator=_WireSchemaGenerator,
    )
    code: dict[str, str] = {}
    for model in models_by_topic.values():
        code.update(_custom_functions(model.__pydantic_core_schema__))
    return {
        "topics": {
            topic: cast(str, roots[(model, "serialization")]["$ref"])
            for topic, model in sorted(models_by_topic.items())
        },
        "$defs": _without_annotations(cast(object, schema.get("$defs", {}))),
        "code": dict(sorted(code.items())),
    }


def wire_schema(cargo_lock: Path = CARGO_LOCK_PATH) -> dict[str, object]:
    return {
        **schema_of(network_topics()),
        "encoding": {
            f"TypedTopic.{name}": _source_hash(cast(object, getattr(TypedTopic, name)))
            for name in ("serialize", "deserialize")
        },
        "collectives": {_GATHER_FUNCTION: _gather_hash()},
        "zenoh": _zenoh_revision(cargo_lock),
    }


def _custom_functions(node: object) -> dict[str, str]:
    """The validators and serializers written for these models (pydantic's own are left
    out) in a pydantic core schema, by name, with a hash of each one's source. Default
    factories are left out too: every field is always sent."""
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in cast(dict[str, object], node).items():
            function = cast(object, getattr(value, "__func__", value))
            module = cast(str, getattr(function, "__module__", "") or "")
            if (
                key != "default_factory"
                and hasattr(function, "__code__")
                and not module.startswith("pydantic")
            ):
                name = f"{module}:{cast(str, getattr(function, '__qualname__', '?'))}"
                found[name] = _source_hash(function)
            found.update(_custom_functions(value))
    elif isinstance(node, (list, tuple)):
        for item in cast(list[object], node):
            found.update(_custom_functions(item))
    return found


def _source_hash(function: object) -> str:
    """A short hash of a function's code, unchanged by its comments, docstring and layout."""
    try:
        source = textwrap.dedent(
            inspect.getsource(cast(Callable[..., object], function))
        )
    except (OSError, TypeError):
        return "unavailable"
    try:
        return _code_hash(ast.parse(source))
    except SyntaxError:
        # A lambda's source is the lines around it, which may not parse alone
        return hashlib.sha256(source.strip().encode()).hexdigest()[:12]


def _code_hash(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        body = cast(list[ast.stmt] | None, getattr(node, "body", None))
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:12]


def _gather_hash() -> str:
    tree = ast.parse(_GATHER_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == _GATHER_FUNCTION
    )
    return _code_hash(function)


def _zenoh_revision(cargo_lock: Path) -> str:
    """The zenoh version and source revision ``Cargo.lock`` pins."""
    lock = cargo_lock.read_text()
    block = lock[lock.index('\nname = "zenoh"\n') :]
    block = block[: block.index("\n\n")]
    fields = dict(
        line.split(" = ", 1) for line in block.strip().splitlines() if " = " in line
    )
    version = fields["version"].strip('"')
    source = fields.get("source", '"registry"').strip('"')
    return f"{version} {source.rsplit('#', 1)[-1]}"


def _without_annotations(node: object, names: bool = False) -> object:
    """``node`` without what doesn't change which messages are accepted. ``names``: its
    keys are field or definition names, kept even when one is called "title"."""
    if isinstance(node, dict):
        return {
            key: (
                # The order fields are declared in isn't on the wire
                sorted(cast(list[str], value))
                if key == "required" and not names
                else _without_annotations(
                    value, names=key in _NAMED_SCHEMAS and not names
                )
            )
            for key, value in cast(dict[str, object], node).items()
            if names or key not in _NOT_ON_THE_WIRE
        }
    if isinstance(node, list):
        return [_without_annotations(item) for item in cast(list[object], node)]
    return node


def schema_hash(schema: Mapping[str, object]) -> str:
    """A short hash of a wire schema, as the ledger records it."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def dump_snapshot(schema: Mapping[str, object]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


@final
@dataclass(frozen=True)
class LedgerEntry:
    wire_version: int
    schema_hash: str


def load_ledger(path: Path = LEDGER_PATH) -> list[LedgerEntry]:
    return parse_ledger(path.read_text())


def parse_ledger(text: str) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        version, hash_ = line.split()
        entries.append(LedgerEntry(int(version), hash_))
    return entries


def dump_ledger(entries: list[LedgerEntry]) -> str:
    return _LEDGER_HEADER + "".join(
        f"{entry.wire_version} {entry.schema_hash}\n" for entry in entries
    )


def changes(old: Mapping[str, object], new: Mapping[str, object]) -> list[str]:
    """What differs between two wire schemas, one line per topic, model, function or setting."""
    lines: list[str] = []
    for section in sorted(set(old) | set(new)):
        before, after = old.get(section), new.get(section)
        if not isinstance(before, dict) or not isinstance(after, dict):
            if before != after:
                lines.append(f"changed {section}")
            continue
        before = cast(Mapping[str, object], before)
        after = cast(Mapping[str, object], after)
        for name in sorted(set(before) | set(after)):
            if name not in after:
                lines.append(f"removed {name}")
            elif name not in before:
                lines.append(f"added {name}")
            elif before[name] != after[name]:
                lines.append(f"changed {name}")
    return lines


def main(argv: list[str] | None = None, cargo_lock: Path = CARGO_LOCK_PATH) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m exo.routing.wire_schema",
        description="Check or write the snapshot and ledger of exo's wire schema.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write {SNAPSHOT_PATH.name}, and the new version's line in {LEDGER_PATH.name}",
    )
    args = parser.parse_args(argv)
    write = cast(bool, args.write)

    current = wire_schema(cargo_lock)
    current_hash = schema_hash(current)
    ledger = load_ledger(LEDGER_PATH)
    last = ledger[-1]
    saved = load_snapshot(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else None
    if (
        saved == current
        and last.wire_version == WIRE_VERSION
        and last.schema_hash == current_hash
    ):
        print(f"{SNAPSHOT_PATH.name} is up to date (wire version {WIRE_VERSION}).")
        return 0
    if saved is not None:
        for line in changes(saved, current):
            print(f"  {line}")
    if not write:
        print(
            "The wire schema or its version changed; see the docstring of src/exo/routing/wire_schema.py."
        )
        return 1

    if last.wire_version + 1 == WIRE_VERSION:
        ledger.append(LedgerEntry(WIRE_VERSION, current_hash))
    elif last.wire_version == WIRE_VERSION and current_hash == last.schema_hash:
        pass  # only the snapshot's layout is out of date
    else:
        print(
            f"WIRE_VERSION is {WIRE_VERSION} and the ledger's last version is "
            f"{last.wire_version}. Set WIRE_VERSION to {last.wire_version + 1} in "
            "src/exo/routing/wire.py: a changed schema always takes the next version.",
            file=sys.stderr,
        )
        return 1
    SNAPSHOT_PATH.write_text(dump_snapshot(current))
    LEDGER_PATH.write_text(dump_ledger(ledger))
    print(
        f"Wrote {SNAPSHOT_PATH.name} and {LEDGER_PATH.name} for wire version {WIRE_VERSION}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
