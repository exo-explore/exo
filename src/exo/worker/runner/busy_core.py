"""One busy core while a tensor-parallel model generates over TCP.

Each step of a tensor-parallel model makes many small collectives between its devices, and a
collective over the ring waits in a blocking call. With little else running on a Mac, every step
then takes longer: measured on two Mac Studios with Llama 3.2 1B, 4 concurrent chats ran at
240-255 tok/s, and at 270-280 with one core kept busy; with Llama 3.1 8B, at 82-86 and 116. Less
than a whole core (a duty cycle, or an efficiency core) gained nothing.

The work is hashing, which runs without the GIL. Generation at these rates isn't held up by it
(about the same as with a busy loop in another process). A pipeline instance is: its steps are
far shorter, so more of its time is Python, and it ran 15-25% slower with this thread, so it
doesn't get one. Over RDMA (MlxJaccl), a busy loop in another process made no difference.
"""

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from exo.shared.types.worker.instances import Instance, MlxRingInstance
from exo.shared.types.worker.shards import ShardMetadata, TensorShardMetadata

# About 0.3 ms of hashing: how long the thread takes to notice it should stop
_BLOCK: Final = bytes(1 << 20)


def wants_a_busy_core(instance: Instance, shard: ShardMetadata) -> bool:
    return (
        isinstance(instance, MlxRingInstance)
        and isinstance(shard, TensorShardMetadata)
        and shard.world_size > 1
    )


@contextmanager
def busy_core_while_generating(
    instance: Instance, shard: ShardMetadata
) -> Iterator[None]:
    """While in this context, a tensor-parallel model on a ring keeps one core busy"""
    if not wants_a_busy_core(instance, shard):
        yield
        return
    stop = threading.Event()

    def stay_busy() -> None:
        while not stop.is_set():
            hashlib.sha256(_BLOCK).digest()

    thread = threading.Thread(target=stay_busy, name="busy-core", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
