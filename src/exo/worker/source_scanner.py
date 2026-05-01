"""Per-worker scan of locally-installed models across all configured sources.

Runs as one of the worker's TaskGroup tasks. On each tick we walk every
:class:`~exo.sources.base.ModelSource`, collect its current entries, and emit a
single :class:`LocalModelsScanned` event per source so the apply handler atomically
swaps that (node, source) view.

A failed scan emits nothing for that source — last-known-good state wins. Empty
results emit an explicit empty list, so removed entries propagate.
"""

from collections.abc import Sequence

import anyio
from anyio import to_thread
from loguru import logger

from exo.shared.types.common import ModelSourceKind, NodeId
from exo.shared.types.events import Event, LocalModelsScanned
from exo.shared.types.worker.local_models import LocalModelEntry
from exo.sources import default_sources
from exo.sources.base import ModelSource
from exo.utils.channels import Sender


class SourceScanner:
    def __init__(
        self,
        node_id: NodeId,
        event_sender: Sender[Event],
        *,
        sources: Sequence[ModelSource] | None = None,
        interval_seconds: float = 60.0,
    ):
        self.node_id: NodeId = node_id
        self.event_sender: Sender[Event] = event_sender
        self.sources: Sequence[ModelSource] = (
            sources if sources is not None else default_sources()
        )
        self.interval_seconds: float = interval_seconds

    async def run(self) -> None:
        # Scan once on boot so the dashboard has data before the first interval tick.
        await self.scan_once()
        while True:
            await anyio.sleep(self.interval_seconds)
            await self.scan_once()

    async def scan_once(self) -> None:
        """Run every available source and emit one event per source."""
        for source in self.sources:
            try:
                if not source.is_available():
                    # Empty entries flush any stale catalog left from a previous availability.
                    await self._emit(source.kind, [])
                    continue
                entries = await to_thread.run_sync(_scan_sync, source, self.node_id)
            except Exception as exc:  # noqa: BLE001 — third-party caches can throw anything
                logger.warning(
                    f"Source {source.kind} ({source.display_name}) scan failed: {exc!r}"
                )
                continue
            await self._emit(source.kind, entries)

    async def _emit(
        self, source_kind: ModelSourceKind, entries: list[LocalModelEntry]
    ) -> None:
        await self.event_sender.send(
            LocalModelsScanned(
                node_id=self.node_id,
                source=source_kind,
                entries=entries,
            )
        )


def _scan_sync(source: ModelSource, node_id: NodeId) -> list[LocalModelEntry]:
    """Module-level helper so ``to_thread.run_sync`` gets a plain function reference."""
    return list(source.scan(node_id))
