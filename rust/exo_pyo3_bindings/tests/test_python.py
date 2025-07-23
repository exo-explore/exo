import logging
import multiprocessing
import multiprocessing.queues
import pickle
import time
from collections.abc import Awaitable
from typing import Callable

import pytest
from exo_pyo3_bindings import ConnectionUpdate, Keypair, DiscoveryService


# # => `tokio::mpsc` channels are closed when all `Sender` are dropped, or when `Receiver::close` is called
# # => the only sender is `KillableTaskHandle.sender: Option<Option<Sender<KillKillableTask>>>`
# # => integrate with https://pyo3.rs/v0.25.1/class/protocols.html#garbage-collector-integration
# #    => set `sender` to `None` to drop the `Sender` & therefore trigger an automatic cleanup
# #    => TODO: there could be a bug where dropping `Sender` won't close the channel in time bc of unprocessed events
# #             so the handle drops and asyncio loop closes BEFORE the task dies...
# #             might wanna figure out some kind of `oneshot` "shutdown confirmed" blocking mechanism or something...??
# # => also there is "cancellable futures" stuff ?? => https://pyo3.rs/main/async-await.html
# #       
# #       For now, always explicitly call cleanup functions to avoid crashes
# #       in the future research tighter integration for automatic cleanup and safety!!!
# #       also look into `pyo3_async_runtimes::tokio::get_runtime()` for blocking calls in Rust
# @pytest.mark.asyncio
# async def test_handle_kill() -> None:
#     print("PYTHON: starting handle")
#     h: KillableTaskHandle = killable_task_spawn()

#     time.sleep(0.35)

#     # for i in range(0, 4):
#     #     print(f"PYTHON: waiting... {i}")
#     #     time.sleep(0.11)

#     # print("PYTHON: killing task")
#     # h.kill_task()

# def test_keypair_creation() -> None:    
#     kp = Keypair.generate_ecdsa()
#     kp_protobuf = kp.to_protobuf_encoding()
#     print(kp_protobuf)
#     kp = Keypair.from_protobuf_encoding(kp_protobuf)
#     assert kp.to_protobuf_encoding() == kp_protobuf


@pytest.mark.asyncio
async def test_discovery_callbacks() -> None:
    ident = Keypair.generate_ed25519()

    service = DiscoveryService(ident)
    service.add_connected_callback(add_connected_callback)
    service.add_disconnected_callback(disconnected_callback)

    for i in range(0, 1):
        print(f"PYTHON: tick {i} of 10")
        time.sleep(1)

    pass


def add_connected_callback(e: ConnectionUpdate) -> None:
    print(f"\n\nPYTHON: Connected callback: {e.peer_id}, {e.connection_id}, {e.local_addr}, {e.send_back_addr}")
    print(
        f"PYTHON: Connected callback: {e.peer_id.__repr__()}, {e.connection_id.__repr__()}, {e.local_addr.__repr__()}, {e.send_back_addr.__repr__()}\n\n")


def disconnected_callback(e: ConnectionUpdate) -> None:
    print(f"\n\nPYTHON: Disconnected callback: {e.peer_id}, {e.connection_id}, {e.local_addr}, {e.send_back_addr}")
    print(
        f"PYTHON: Disconnected callback: {e.peer_id.__repr__()}, {e.connection_id.__repr__()}, {e.local_addr.__repr__()}, {e.send_back_addr.__repr__()}\n\n")


# async def foobar(a: Callable[[str], Awaitable[str]]):
#     abc = await a("")
#     pass

# def test_keypair_pickling() -> None:
#     def subprocess_task(kp: Keypair, q: multiprocessing.queues.Queue[Keypair]):
#         logging.info("a")
#         assert q.get() == kp
#         logging.info("b")
#
#
#     kp = Keypair.generate_ed25519()
#     q: multiprocessing.queues.Queue[Keypair] = multiprocessing.Queue()
#
#     p = multiprocessing.Process(target=subprocess_task, args=(kp, q))
#     p.start()
#     q.put(kp)
#     p.join()