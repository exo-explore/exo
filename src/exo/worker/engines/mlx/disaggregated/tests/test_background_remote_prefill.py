import threading
import time
from typing import BinaryIO

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from exo.utils.ports import random_ephemeral_port
from exo.worker.disaggregated.server import PrefillRequest, PrefillServer
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.remote_prefill import RemotePrefillHandle

_SEQUENCE_LENGTH = 6


def _equal(a: mx.array, b: mx.array) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return bool(
        np.array_equal(np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16)))
    )


def _prefilled_cache() -> KVCache:
    mx.random.seed(0)
    cache = KVCache()
    shape = (1, 2, _SEQUENCE_LENGTH, 4)
    cache.keys = mx.random.normal(shape).astype(mx.bfloat16)
    cache.values = mx.random.normal(shape).astype(mx.bfloat16)
    # Evaluated here: the test's prefill server sends it from its own thread, and MLX
    # streams belong to the thread that made them
    mx.eval(cache.keys, cache.values)
    cache.offset = _SEQUENCE_LENGTH
    return cache


def _wait_until_done(handle: RemotePrefillHandle) -> None:
    deadline = time.monotonic() + 5
    while not handle.done:
        assert time.monotonic() < deadline, "background fetch did not finish"
        time.sleep(0.01)


class _FakeGenerationBatch:
    def __init__(self) -> None:
        self.uids: list[int] = []

    def __len__(self) -> int:
        return len(self.uids)


class _FakeMlxBatchGenerator:
    def __init__(self) -> None:
        self._unprocessed_sequences: list[int] = []
        self._prompt_batch: list[int] = []
        self._generation_batch = _FakeGenerationBatch()


def _empty_generator() -> ExoBatchGenerator:
    gen = object.__new__(ExoBatchGenerator)
    gen._mlx_gen = _FakeMlxBatchGenerator()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    gen._active_tasks = {}  # pyright: ignore[reportPrivateUsage]
    gen._finishing = set()  # pyright: ignore[reportPrivateUsage]
    gen._step_count = 0  # pyright: ignore[reportPrivateUsage]
    gen._pending_remote_prefills = {}  # pyright: ignore[reportPrivateUsage]
    gen._placeholder_by_uid = {}  # pyright: ignore[reportPrivateUsage]
    return gen


@pytest.mark.slow
def test_background_fetch_runs_while_the_caller_continues() -> None:
    gold = _prefilled_cache()
    release = threading.Event()

    def resolve(_job: PrefillRequest, wfile: BinaryIO) -> bool:
        # Hold the answer back, like a prefill node that is still working.
        release.wait(timeout=5)
        write_cache_to_wire(wfile, [gold], request_id="r", model_id="m")
        return True

    port = random_ephemeral_port()
    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=port)
    try:
        handle = RemotePrefillHandle(
            mx.arange(_SEQUENCE_LENGTH),
            endpoint=f"127.0.0.1:{port}",
            request_id="request-1",
            model_id="m",
        )
        # The constructor returned while the server is still holding the answer.
        time.sleep(0.1)
        assert not handle.done

        release.set()
        _wait_until_done(handle)
        received = KVCache()
        _, num_tokens, _ = handle.ingest([received])

        assert num_tokens == _SEQUENCE_LENGTH
        assert received.offset == _SEQUENCE_LENGTH
        assert received.keys is not None and gold.keys is not None
        assert received.values is not None and gold.values is not None
        assert _equal(received.keys, gold.keys)
        assert _equal(received.values, gold.values)
    finally:
        release.set()
        server.stop()


@pytest.mark.slow
def test_failed_background_fetch_raises_on_ingest() -> None:
    def resolve(_job: PrefillRequest, _wfile: BinaryIO) -> bool:
        return False

    port = random_ephemeral_port()
    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=port)
    try:
        handle = RemotePrefillHandle(
            mx.arange(_SEQUENCE_LENGTH),
            endpoint=f"127.0.0.1:{port}",
            request_id="request-2",
            model_id="m",
        )
        _wait_until_done(handle)
        with pytest.raises(RuntimeError):
            handle.ingest([KVCache()])
    finally:
        server.stop()


@pytest.mark.slow
def test_cancelled_background_fetch_is_never_inserted_or_answered() -> None:
    gold = _prefilled_cache()
    release = threading.Event()

    def resolve(_job: PrefillRequest, wfile: BinaryIO) -> bool:
        # Hold the answer back, like a prefill node that is still working.
        release.wait(timeout=5)
        write_cache_to_wire(wfile, [gold], request_id="r", model_id="m")
        return True

    port = random_ephemeral_port()
    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=port)
    try:
        handle = RemotePrefillHandle(
            mx.arange(_SEQUENCE_LENGTH),
            endpoint=f"127.0.0.1:{port}",
            request_id="request-cancel",
            model_id="m",
        )
        gen = _empty_generator()
        placeholder = -1
        # The queued request is never touched once cancelled, so a bare object stands in for it.
        gen._pending_remote_prefills[placeholder] = (  # pyright: ignore[reportPrivateUsage, reportArgumentType]
            handle,
            object(),
        )

        # Cancel while the fetch is still in flight.
        gen.cancel([placeholder])
        assert gen._pending_remote_prefills == {}  # pyright: ignore[reportPrivateUsage]

        # Let the fetch finish on its own; its result must be discarded.
        release.set()
        _wait_until_done(handle)

        results: list[object] = []
        for _ in range(3):
            results.extend(gen.step())

        assert results == []
        assert gen._pending_remote_prefills == {}  # pyright: ignore[reportPrivateUsage]
        assert gen._active_tasks == {}  # pyright: ignore[reportPrivateUsage]
        assert gen._mlx_gen._generation_batch.uids == []  # pyright: ignore[reportPrivateUsage]
    finally:
        release.set()
        server.stop()
