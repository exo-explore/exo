import asyncio
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from exo.shared.types.moe_dispatch import (
    AllToAllTelemetry,
    DispatchBatch,
    TokenPacket,
)


class AsyncAllToAllDispatcher:
    """Asynchronous All-to-All Token Dispatcher with double-buffering and batch packing."""

    def __init__(self, local_node_id: str, default_max_batch_size: int = 16) -> None:
        self._local_node_id = local_node_id
        self._max_batch_size = default_max_batch_size
        self._expert_to_node_map: dict[int, str] = {}
        self._dispatch_sequence: int = 0

    def register_expert_placement(self, expert_id: int, node_id: str) -> None:
        """Registers the hosting node location for an MoE expert."""
        self._expert_to_node_map[expert_id] = node_id

    def get_node_for_expert(self, expert_id: int) -> str:
        """Retrieves node hosting the specified expert; defaults to local node if unregistered."""
        return self._expert_to_node_map.get(expert_id, self._local_node_id)

    def pack_tokens(
        self,
        packets: Sequence[TokenPacket],
        max_batch_size: int | None = None,
    ) -> Sequence[DispatchBatch]:
        """Groups individual token routing packets into packed batches by (target_node, expert)."""
        limit = max_batch_size or self._max_batch_size
        grouped: dict[tuple[str, int], list[TokenPacket]] = defaultdict(list)

        for p in packets:
            target_node = self.get_node_for_expert(p.target_expert_id)
            grouped[(target_node, p.target_expert_id)].append(p)

        batches: list[DispatchBatch] = []
        for (target_node, expert_id), node_packets in grouped.items():
            for i in range(0, len(node_packets), limit):
                chunk = node_packets[i : i + limit]
                batch_id = f"batch-{uuid.uuid4().hex[:8]}-{expert_id}"
                batches.append(
                    DispatchBatch(
                        batch_id=batch_id,
                        target_node_id=target_node,
                        target_expert_id=expert_id,
                        packets=tuple(chunk),
                    )
                )

        return tuple(batches)

    async def _simulate_or_send_network_batches(
        self,
        batches: Sequence[DispatchBatch],
        custom_send_fn: Callable[[DispatchBatch], Awaitable[None]] | None = None,
    ) -> float:
        """Transmits packed batches asynchronously, measuring network transmission duration."""
        start_time = time.perf_counter()

        # If custom network send coroutine provided, execute concurrently
        if custom_send_fn is not None:
            tasks = [custom_send_fn(batch) for batch in batches]
            if tasks:
                await asyncio.gather(*tasks)
        else:
            # Simulated realistic network flight over WiFi
            # Emulates 1ms serialization per batch plus base network flight time
            simulated_delay = 0.001 * max(1, len(batches))
            await asyncio.sleep(simulated_delay)

        return (time.perf_counter() - start_time) * 1000.0

    async def dispatch_and_overlap(
        self,
        local_computation_fn: Callable[[], Any],
        packets: Sequence[TokenPacket],
        custom_send_fn: Callable[[DispatchBatch], Awaitable[None]] | None = None,
    ) -> tuple[Any, AllToAllTelemetry]:
        """Executes double-buffering pipeline: overlaps local expert compute with network dispatch."""
        self._dispatch_sequence += 1
        dispatch_id = f"dispatch-{self._local_node_id}-{self._dispatch_sequence}"

        # 1. Separate packets into local vs remote
        remote_packets: list[TokenPacket] = []
        for p in packets:
            target_node = self.get_node_for_expert(p.target_expert_id)
            if target_node != self._local_node_id:
                remote_packets.append(p)

        # 2. Pack remote packets into grouped batches
        batches = self.pack_tokens(remote_packets)

        # 3. Launch asynchronous background transmission for remote batches
        send_task = asyncio.create_task(
            self._simulate_or_send_network_batches(batches, custom_send_fn)
        )

        # 4. Concurrently run local expert computation on this worker
        comp_start = time.perf_counter()
        computation_result = local_computation_fn()
        computation_time_ms = (time.perf_counter() - comp_start) * 1000.0

        # 5. Await completion of remote network transfer
        transmission_time_ms = await send_task

        # 6. Calculate overlapping statistics
        overlap_ms = min(computation_time_ms, transmission_time_ms)
        total_payload_bytes = sum(b.payload_bytes_estimate for b in batches)
        bandwidth_mbps = (
            (total_payload_bytes * 8.0 / (transmission_time_ms * 1e-3)) / 1e6
            if transmission_time_ms > 0
            else 0.0
        )

        telemetry = AllToAllTelemetry(
            dispatch_id=dispatch_id,
            tokens_dispatched=len(packets),
            batches_count=len(batches),
            total_transmission_time_ms=transmission_time_ms,
            computation_overlap_time_ms=overlap_ms,
            effective_bandwidth_mbps=bandwidth_mbps,
        )

        return computation_result, telemetry
