import asyncio
import time

import numpy as np
import pytest

from exo.master.moe import AsyncAllToAllDispatcher
from exo.shared.types.moe_dispatch import (
    DispatchBatch,
    TokenPacket,
)
from exo.worker.engines.moe import ZeroCopyTensorRingBuffer


def test_token_packet_and_dispatch_batch() -> None:
    p1 = TokenPacket(
        token_id=1,
        sequence_id="seq-1",
        hidden_states=(0.1, 0.2, 0.3, 0.4),
        target_expert_id=3,
        source_node_id="node-a",
    )
    p2 = TokenPacket(
        token_id=2,
        sequence_id="seq-1",
        hidden_states=(0.5, 0.6, 0.7, 0.8),
        target_expert_id=3,
        source_node_id="node-a",
    )

    batch = DispatchBatch(
        batch_id="batch-test-1",
        target_node_id="node-b",
        target_expert_id=3,
        packets=(p1, p2),
    )

    assert batch.token_count == 2
    assert batch.payload_bytes_estimate > 0
    assert batch.target_node_id == "node-b"


def test_zero_copy_ring_buffer_read_write() -> None:
    capacity = 4
    slot_shape = (2, 8)
    ring = ZeroCopyTensorRingBuffer(capacity=capacity, slot_shape=slot_shape, dtype=np.float32)

    assert ring.capacity == 4
    assert ring.count == 0
    assert ring.is_empty
    assert not ring.is_full

    test_data = np.ones(slot_shape, dtype=np.float32) * 42.0
    slot_idx = ring.write(test_data, metadata={"expert_id": 5})

    assert slot_idx == 0
    assert ring.count == 1
    assert not ring.is_empty

    view, meta = ring.read()
    assert meta == {"expert_id": 5}
    np.testing.assert_array_equal(view, test_data)
    assert ring.count == 0
    assert ring.is_empty


def test_zero_copy_ring_buffer_fifo_wrap() -> None:
    capacity = 3
    slot_shape = (1, 4)
    ring = ZeroCopyTensorRingBuffer(capacity=capacity, slot_shape=slot_shape)

    for i in range(capacity):
        data = np.full(slot_shape, fill_value=float(i), dtype=np.float32)
        ring.write(data, metadata={"val": i})

    assert ring.is_full

    # Should raise TimeoutError when writing to full buffer with tiny timeout
    with pytest.raises(TimeoutError):
        ring.write(np.zeros(slot_shape, dtype=np.float32), timeout=0.01)

    # Read all and verify FIFO
    for i in range(capacity):
        view, meta = ring.read()
        assert meta["val"] == i
        assert view[0, 0] == float(i)

    assert ring.is_empty

    # Should raise TimeoutError when reading from empty buffer
    with pytest.raises(TimeoutError):
        ring.read(timeout=0.01)


def test_dispatcher_batch_packing() -> None:
    dispatcher = AsyncAllToAllDispatcher(local_node_id="local-node", default_max_batch_size=2)
    dispatcher.register_expert_placement(expert_id=1, node_id="remote-node-1")
    dispatcher.register_expert_placement(expert_id=2, node_id="remote-node-2")

    packets = [
        TokenPacket(
            token_id=i,
            sequence_id="seq-test",
            hidden_states=(0.1, 0.2),
            target_expert_id=1,
            source_node_id="local-node",
        )
        for i in range(5)
    ]

    batches = dispatcher.pack_tokens(packets, max_batch_size=2)
    # 5 packets with max_batch_size=2 should form 3 batches (sizes 2, 2, 1)
    assert len(batches) == 3
    assert batches[0].token_count == 2
    assert batches[1].token_count == 2
    assert batches[2].token_count == 1
    for b in batches:
        assert b.target_node_id == "remote-node-1"
        assert b.target_expert_id == 1


@pytest.mark.asyncio
async def test_dispatcher_dispatch_and_overlap() -> None:
    dispatcher = AsyncAllToAllDispatcher(local_node_id="local-worker", default_max_batch_size=4)
    dispatcher.register_expert_placement(expert_id=0, node_id="local-worker")
    dispatcher.register_expert_placement(expert_id=1, node_id="remote-worker-1")

    # Mixed local and remote packets
    packets = [
        TokenPacket(
            token_id=1,
            sequence_id="seq-1",
            hidden_states=(0.1, 0.2, 0.3),
            target_expert_id=0,  # Local
            source_node_id="local-worker",
        ),
        TokenPacket(
            token_id=2,
            sequence_id="seq-1",
            hidden_states=(0.4, 0.5, 0.6),
            target_expert_id=1,  # Remote
            source_node_id="local-worker",
        ),
        TokenPacket(
            token_id=3,
            sequence_id="seq-1",
            hidden_states=(0.7, 0.8, 0.9),
            target_expert_id=1,  # Remote
            source_node_id="local-worker",
        ),
    ]

    # Local computation workload
    def local_compute() -> str:
        time.sleep(0.02)  # 20ms simulated local expert compute
        return "local_compute_done"

    sent_batches = []

    async def custom_network_send(batch: DispatchBatch) -> None:
        await asyncio.sleep(0.01)  # 10ms simulated remote transmission
        sent_batches.append(batch.batch_id)

    res, telemetry = await dispatcher.dispatch_and_overlap(
        local_computation_fn=local_compute,
        packets=packets,
        custom_send_fn=custom_network_send,
    )

    assert res == "local_compute_done"
    assert len(sent_batches) == 1  # 2 remote packets packed into 1 batch
    assert telemetry.tokens_dispatched == 3
    assert telemetry.batches_count == 1
    assert telemetry.computation_overlap_time_ms > 0
    assert 0.0 <= telemetry.overlap_ratio <= 1.0
