import asyncio
import time

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
    a = _add_connected_callback(service)
    d = _add_disconnected_callback(service)

    # stream_get_a, stream_put = _make_iter()
    # service.add_connected_callback(stream_put)
    #
    # stream_get_d, stream_put = _make_iter()
    # service.add_disconnected_callback(stream_put)

    # async for c in stream_get_a:
    # await connected_callback(c)

    for i in range(0, 10):
        print(f"PYTHON: tick {i} of 10")
        await asyncio.sleep(1)

    print(service, a, d)  # only done to prevent GC... TODO: come up with less hacky solution


def _add_connected_callback(d: DiscoveryService):
    stream_get, stream_put = _make_iter()
    d.add_connected_callback(stream_put)

    async def run():
        async for c in stream_get:
            await connected_callback(c)

    return asyncio.create_task(run())


def _add_disconnected_callback(d: DiscoveryService):
    stream_get, stream_put = _make_iter()

    async def run():
        async for c in stream_get:
            await disconnected_callback(c)

    d.add_disconnected_callback(stream_put)
    return asyncio.create_task(run())


async def connected_callback(e: ConnectionUpdate) -> None:
    print(f"\n\nPYTHON: Connected callback: {e.peer_id}, {e.connection_id}, {e.local_addr}, {e.send_back_addr}")
    print(
        f"PYTHON: Connected callback: {e.peer_id.__repr__()}, {e.connection_id.__repr__()}, {e.local_addr.__repr__()}, {e.send_back_addr.__repr__()}\n\n")


async def disconnected_callback(e: ConnectionUpdate) -> None:
    print(f"\n\nPYTHON: Disconnected callback: {e.peer_id}, {e.connection_id}, {e.local_addr}, {e.send_back_addr}")
    print(
        f"PYTHON: Disconnected callback: {e.peer_id.__repr__()}, {e.connection_id.__repr__()}, {e.local_addr.__repr__()}, {e.send_back_addr.__repr__()}\n\n")


def _foo_task() -> None:
    print("PYTHON: This simply runs in asyncio context")


def _make_iter():
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[ConnectionUpdate] = asyncio.Queue()

    def put(c: ConnectionUpdate) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, c)

    async def get():
        while True:
            yield await queue.get()

    return get(), put

# async def inputstream_generator(channels=1, **kwargs):
#     """Generator that yields blocks of input data as NumPy arrays."""
#     q_in = asyncio.Queue()
#     loop = asyncio.get_event_loop()
#
#     def callback(indata, frame_count, time_info, status):
#         loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
#
#     stream = sd.InputStream(callback=callback, channels=channels, **kwargs)
#     with stream:
#         while True:
#             indata, status = await q_in.get()
#             yield indata, status
