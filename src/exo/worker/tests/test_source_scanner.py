"""Tests for the worker's SourceScanner — diff/emission against fake sources."""

from collections.abc import Iterable
from pathlib import Path

import anyio
import pytest

from exo.shared.types.common import ModelSourceKind, NodeId
from exo.shared.types.events import Event, LocalModelsScanned
from exo.shared.types.memory import Memory
from exo.shared.types.worker.local_models import LocalModelEntry
from exo.utils.channels import channel
from exo.worker.source_scanner import SourceScanner


class FakeSource:
    def __init__(
        self,
        kind: ModelSourceKind,
        entries: list[LocalModelEntry],
        *,
        available: bool = True,
        raise_on_scan: bool = False,
    ) -> None:
        self.kind: ModelSourceKind = kind
        self.display_name: str = kind
        self._available: bool = available
        self._entries: list[LocalModelEntry] = entries
        self._raise: bool = raise_on_scan
        self.scan_count: int = 0

    def is_available(self) -> bool:
        return self._available

    def scan(self, node_id: NodeId) -> Iterable[LocalModelEntry]:
        self.scan_count += 1
        if self._raise:
            raise RuntimeError("boom")
        # Return the configured entries with node_id stamped on.
        return [
            entry.model_copy(update={"node_id": node_id}) for entry in self._entries
        ]

    def resolve_path(
        self, external_id: str
    ) -> Path | None:  # pragma: no cover - unused
        return None


def _entry(source: ModelSourceKind, external_id: str) -> LocalModelEntry:
    return LocalModelEntry(
        node_id=NodeId("placeholder"),
        source=source,
        external_id=external_id,
        display_name=external_id,
        path=f"/fake/{external_id}",
        format="safetensors",
        size_bytes=Memory(in_bytes=4096),
        loadable_with_mlx=True,
    )


@pytest.mark.asyncio
async def test_scan_once_emits_one_event_per_source() -> None:
    sender, receiver = channel[Event]()
    node = NodeId("worker-A")
    sources = [
        FakeSource("huggingface", [_entry("huggingface", "org/model-1")]),
        FakeSource("lmstudio", [_entry("lmstudio", "Pub/Model-2")]),
    ]
    scanner = SourceScanner(node, sender, sources=sources, interval_seconds=999)
    await scanner.scan_once()

    events = receiver.collect()
    assert len(events) == 2
    by_kind = {e.source: e for e in events if isinstance(e, LocalModelsScanned)}
    assert by_kind["huggingface"].entries[0].external_id == "org/model-1"
    assert by_kind["lmstudio"].entries[0].external_id == "Pub/Model-2"


@pytest.mark.asyncio
async def test_unavailable_source_emits_empty_list() -> None:
    """Stale catalog must be flushed when a source disappears."""
    sender, receiver = channel[Event]()
    sources = [
        FakeSource("ollama", [_entry("ollama", "llama3:8b")], available=False),
    ]
    scanner = SourceScanner(NodeId("n"), sender, sources=sources, interval_seconds=999)
    await scanner.scan_once()

    events = [e for e in receiver.collect() if isinstance(e, LocalModelsScanned)]
    assert len(events) == 1
    assert events[0].source == "ollama"
    assert events[0].entries == []
    assert sources[0].scan_count == 0  # is_available() short-circuited the scan


@pytest.mark.asyncio
async def test_failing_source_does_not_block_others() -> None:
    sender, receiver = channel[Event]()
    sources = [
        FakeSource("huggingface", [_entry("huggingface", "ok/1")], raise_on_scan=True),
        FakeSource("lmstudio", [_entry("lmstudio", "ok/2")]),
    ]
    scanner = SourceScanner(NodeId("n"), sender, sources=sources, interval_seconds=999)
    await scanner.scan_once()

    events = [e for e in receiver.collect() if isinstance(e, LocalModelsScanned)]
    # Only the working source emits; the failing one is silenced.
    assert [e.source for e in events] == ["lmstudio"]


@pytest.mark.asyncio
async def test_run_emits_immediately_then_loops() -> None:
    sender, receiver = channel[Event]()
    sources = [FakeSource("exo", [_entry("exo", "x/y")])]
    scanner = SourceScanner(NodeId("n"), sender, sources=sources, interval_seconds=0.05)
    async with anyio.create_task_group() as tg:
        tg.start_soon(scanner.run)
        await anyio.sleep(0.12)  # 1 immediate + at least 1 interval-driven scan
        tg.cancel_scope.cancel()
    events = [e for e in receiver.collect() if isinstance(e, LocalModelsScanned)]
    assert len(events) >= 2
    assert sources[0].scan_count >= 2
