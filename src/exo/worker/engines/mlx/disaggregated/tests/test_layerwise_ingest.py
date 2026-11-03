import socket
import time
from collections.abc import Callable, Iterator, Sequence
from typing import BinaryIO, cast

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

import exo.worker.engines.mlx.disaggregated.client as client_module
import exo.worker.engines.mlx.generator.remote_prefill as remote_prefill_module
from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    DType,
    Header,
    KVChunk,
    Message,
    ProtocolError,
    TensorBlob,
    write_error,
    write_header,
    write_message,
)
from exo.worker.disaggregated.server import PrefillRequest, PrefillServer
from exo.worker.engines.mlx.disaggregated.adapter import array_to_bytes
from exo.worker.engines.mlx.disaggregated.client import (
    LayerwiseIngest,
    PrefillResult,
    ingest_into_mlx_cache,
)
from exo.worker.engines.mlx.generator.remote_prefill import (
    RemotePrefillHandle,
    remote_prefill,
)

Cache = KVCache | RotatingKVCache | ArraysCache

N_HEADS = 2
HEAD_DIM = 4
PREFIX = 3


def _bits(a: mx.array | None) -> tuple[str, tuple[int, ...], bytes] | None:
    if a is None:
        return None
    mx.eval(a)
    raw = a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
    return (str(a.dtype), tuple(a.shape), np.asarray(raw).tobytes())


def _state(caches: list[Cache]) -> list[object]:
    out: list[object] = []
    for c in caches:
        if isinstance(c, ArraysCache):
            out.append([_bits(a) for a in c.cache])
        elif isinstance(c, RotatingKVCache):
            out.append((_bits(c.keys), _bits(c.values), c.offset, c._idx))
        else:
            out.append((_bits(c.keys), _bits(c.values), c.offset))
    return out


def _rand(shape: tuple[int, ...], seed: int, dtype: mx.Dtype) -> mx.array:
    return (mx.random.uniform(shape=shape, key=mx.random.key(seed)) * 10).astype(dtype)


def _kv(layer: int, num_tokens: int, seed: int, dtype: DType = "bfloat16") -> KVChunk:
    mx_dtype = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}[
        dtype
    ]
    shape = (num_tokens, N_HEADS, HEAD_DIM)
    return KVChunk(
        layer_idx=layer,
        num_tokens=num_tokens,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        dtype=dtype,
        keys=array_to_bytes(_rand(shape, seed, mx_dtype)),
        values=array_to_bytes(_rand(shape, seed + 1000, mx_dtype)),
    )


def _messages() -> list[Message]:
    """Layers 0 and 3 are KVCache (layer 0 split over two chunks), 1 is a
    RotatingKVCache and 2 an ArraysCache; layers arrive out of order."""
    blobs = [
        TensorBlob(
            dtype="float32",
            shape=(1, 3),
            data=array_to_bytes(_rand((1, 3), 7, mx.float32)),
        ),
        TensorBlob(
            dtype="bfloat16",
            shape=(2, 2),
            data=array_to_bytes(_rand((2, 2), 8, mx.bfloat16)),
        ),
    ]
    return [
        _kv(0, 3, 1),
        _kv(3, 7, 2),
        ArraysState(layer_idx=2, arrays=blobs),
        _kv(1, 7, 3),
        _kv(0, 4, 4),
        Done(total_tokens=7),
    ]


def _caches(with_prefix: bool) -> list[Cache]:
    caches: list[Cache] = [
        KVCache(),
        RotatingKVCache(max_size=64),
        ArraysCache(2),
        KVCache(),
    ]
    if with_prefix:
        for i in (0, 3):
            c = caches[i]
            assert isinstance(c, KVCache)
            c.keys = _rand((1, N_HEADS, PREFIX + 2, HEAD_DIM), 50 + i, mx.bfloat16)
            c.values = _rand((1, N_HEADS, PREFIX + 2, HEAD_DIM), 60 + i, mx.bfloat16)
            c.offset = PREFIX + 2
    return caches


def _buffered(
    messages: list[Message],
    caches: list[Cache],
    start_pos: int,
    layer_offset: int = 0,
) -> int:
    """What remote_prefill_fetch + ingest_into_mlx_cache do with the same messages."""
    result = PrefillResult(header=Header(num_layers=4))
    for m in messages:
        if isinstance(m, KVChunk):
            result.kv_chunks.setdefault(m.layer_idx, []).append(m)
        elif isinstance(m, ArraysState):
            result.arrays[m.layer_idx] = m.arrays
    return ingest_into_mlx_cache(
        result, caches, start_pos=start_pos, layer_offset=layer_offset
    )


def _layerwise(
    messages: list[Message],
    caches: list[Cache],
    start_pos: int,
    staged: LayerwiseIngest | None = None,
) -> int:
    staged = staged or LayerwiseIngest()
    for m in messages:
        if isinstance(m, KVChunk):
            staged.add_kv_chunk(m)
        elif isinstance(m, ArraysState):
            staged.add_arrays_state(m)
        elif isinstance(m, Done):
            staged.finish(m)
    return staged.commit(caches, start_pos=start_pos)


@pytest.mark.parametrize("start_pos", [0, PREFIX])
def test_layerwise_ingest_equals_buffered_ingest(start_pos: int) -> None:
    messages = _messages()
    want, got = _caches(start_pos > 0), _caches(start_pos > 0)
    assert _buffered(messages, want, start_pos) == start_pos + 7
    assert _layerwise(messages, got, start_pos) == start_pos + 7
    assert _state(got) == _state(want)
    k0 = got[0]
    assert isinstance(k0, KVCache) and k0.keys is not None
    assert k0.offset == start_pos + 7 and int(k0.keys.shape[2]) == start_pos + 7


def test_layerwise_ingest_needs_done_and_nothing_after_it() -> None:
    staged = LayerwiseIngest()
    staged.add_kv_chunk(_kv(0, 3, 1))
    caches = _caches(True)
    before = _state(caches)
    with pytest.raises(ProtocolError, match="before Done"):
        staged.commit(caches, start_pos=PREFIX)
    assert _state(caches) == before
    staged.finish(Done(total_tokens=3))
    with pytest.raises(ProtocolError, match="after Done"):
        staged.add_kv_chunk(_kv(1, 3, 2))


def test_failed_commit_leaves_every_layer_as_it_was() -> None:
    """A layer that cannot be applied (here: wrong head count for the prefix) undoes
    the layers applied before it."""
    staged = LayerwiseIngest()
    staged.add_kv_chunk(_kv(0, 4, 1))
    bad = KVChunk(
        layer_idx=3,
        num_tokens=4,
        n_heads=N_HEADS + 1,
        head_dim=HEAD_DIM,
        dtype="bfloat16",
        keys=array_to_bytes(_rand((4, N_HEADS + 1, HEAD_DIM), 2, mx.bfloat16)),
        values=array_to_bytes(_rand((4, N_HEADS + 1, HEAD_DIM), 3, mx.bfloat16)),
    )
    staged.add_kv_chunk(bad)
    staged.finish(Done(total_tokens=4))
    caches = _caches(True)
    before = _state(caches)
    with pytest.raises(ValueError):
        staged.commit(caches, start_pos=PREFIX)
    assert _state(caches) == before


@pytest.mark.parametrize("start_pos", [0, PREFIX])
@pytest.mark.parametrize(
    ("layers", "decoded_layers"),
    [((2, 4), [3]), ((2, 3), [])],
    ids=["layers-2-3", "layer-2-only"],
)
def test_pipeline_rank_stages_only_its_layers(
    start_pos: int,
    layers: tuple[int, int],
    decoded_layers: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipeline rank (exo-2's layer_offset) gets the same cache and offset as buffered
    ingest; KV chunks of layers outside its range are dropped on arrival without being
    decoded. A rank whose only layer is an ArraysCache still takes the stream's offset
    from the other layers' KV chunks, as buffered ingest does."""
    decoded: list[int] = []
    decode = client_module.chunk_to_mlx_nhd

    def recorded(chunk: KVChunk) -> tuple[mx.array, mx.array]:
        decoded.append(chunk.layer_idx)
        return decode(chunk)

    monkeypatch.setattr(client_module, "chunk_to_mlx_nhd", recorded)
    messages = _messages()
    first, end = layers
    want = _caches(start_pos > 0)[first:end]
    got = _caches(start_pos > 0)[first:end]
    assert _buffered(messages, want, start_pos, layer_offset=first) == start_pos + 7
    decoded.clear()
    staged = LayerwiseIngest.for_caches(got, layer_offset=first)
    assert _layerwise(messages, got, start_pos, staged) == start_pos + 7
    assert decoded == decoded_layers
    assert _state(got) == _state(want)


# --------------------------------------------------------------------------- over a socket


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = cast(tuple[str, int], s.getsockname())[1]
    s.close()
    return port


Responder = Callable[[BinaryIO], None]


@pytest.fixture
def server() -> Iterator[tuple[str, list[Responder]]]:
    """A prefill server whose answer the test sets (one responder per request)."""
    answers: list[Responder] = []

    def resolve(_job: PrefillRequest, wfile: BinaryIO) -> bool:
        answers.pop(0)(wfile)
        return True

    port = _free_port()
    srv = PrefillServer(resolve=resolve, host="127.0.0.1", port=port)
    try:
        yield f"127.0.0.1:{port}", answers
    finally:
        srv.stop()


def _send(
    messages: Sequence[Message], tail: Callable[[BinaryIO], None] | None = None
) -> Responder:
    def respond(wfile: BinaryIO) -> None:
        write_header(wfile, Header(num_layers=4, dtype="bfloat16", start_pos=PREFIX))
        for m in messages:
            write_message(wfile, m)
        if tail is not None:
            tail(wfile)
        wfile.flush()

    return respond


def _run(endpoint: str, caches: list[Cache], layer_offset: int = 0) -> int:
    _, tokens, _ = remote_prefill(
        mx.array(list(range(PREFIX + 7))),
        list(caches),
        None,
        endpoint=endpoint,
        request_id="r",
        model_id="m",
        start_pos=PREFIX,
        layer_offset=layer_offset,
    )
    return tokens


@pytest.mark.slow
@pytest.mark.parametrize("layer_offset", [0, 2], ids=["all-layers", "pipeline-rank"])
def test_remote_prefill_layerwise_matches_buffered(
    server: tuple[str, list[Responder]],
    monkeypatch: pytest.MonkeyPatch,
    layer_offset: int,
) -> None:
    """Through remote_prefill, for a rank holding every layer and for a pipeline rank
    holding global layers 2 and 3 (exo-2's layer_offset)."""
    endpoint, answers = server
    want, got = _caches(True)[layer_offset:], _caches(True)[layer_offset:]
    answers.extend([_send(_messages()), _send(_messages())])
    monkeypatch.setattr(remote_prefill_module, "EXO_LAYERWISE_REMOTE_INGEST", False)
    assert _run(endpoint, want, layer_offset) == 7
    monkeypatch.setattr(remote_prefill_module, "EXO_LAYERWISE_REMOTE_INGEST", True)
    assert _run(endpoint, got, layer_offset) == 7
    assert _state(got) == _state(want)


@pytest.mark.slow
def test_background_fetch_buffers_whatever_the_flag(
    server: tuple[str, list[Responder]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """exo-3's background fetch thread must not make MLX calls, so it keeps buffering
    with EXO_LAYERWISE_REMOTE_INGEST on, and ingests as buffered ingest does."""
    endpoint, answers = server
    answers.append(_send(_messages()))
    monkeypatch.setattr(remote_prefill_module, "EXO_LAYERWISE_REMOTE_INGEST", True)
    fetch = remote_prefill_module.remote_prefill_fetch
    layerwise_args: list[object] = []

    def recorded(*args: object, **kwargs: object) -> PrefillResult:
        layerwise_args.append(kwargs.get("layerwise"))
        return fetch(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(remote_prefill_module, "remote_prefill_fetch", recorded)
    handle = RemotePrefillHandle(
        mx.array(list(range(PREFIX + 7))),
        endpoint=endpoint,
        request_id="r",
        model_id="m",
        start_pos=PREFIX,
    )
    for _ in range(500):
        if handle.done:
            break
        time.sleep(0.01)
    got, want = _caches(True), _caches(True)
    _, tokens, _ = handle.ingest(list(got))
    assert tokens == 7
    assert _buffered(_messages(), want, PREFIX) == PREFIX + 7
    assert _state(got) == _state(want)
    assert layerwise_args == [None]


def _error_frame(wfile: BinaryIO) -> None:
    write_error(wfile, code=500, message="producer failed mid-stream")


def _truncated_frame(wfile: BinaryIO) -> None:
    wfile.write((1000).to_bytes(4, "big"))
    wfile.write(b"\0" * 10)


@pytest.mark.slow
@pytest.mark.parametrize(
    "tail",
    [_error_frame, None, _truncated_frame],
    ids=["error-frame", "eof-at-frame-boundary", "truncated-frame"],
)
def test_mid_stream_failure_leaves_the_cache_clean(
    server: tuple[str, list[Responder]],
    monkeypatch: pytest.MonkeyPatch,
    tail: Callable[[BinaryIO], None] | None,
) -> None:
    """Two of four layers arrive, then the stream fails. The prefix cache the caller
    falls back to local prefill with must be exactly as it was."""
    endpoint, answers = server
    monkeypatch.setattr(remote_prefill_module, "EXO_LAYERWISE_REMOTE_INGEST", True)
    first_two = [m for m in _messages() if isinstance(m, KVChunk)][:2]
    answers.append(_send(first_two, tail))
    caches = _caches(True)
    before = _state(caches)
    with pytest.raises((RuntimeError, ConnectionError, ProtocolError)):
        _run(endpoint, caches)
    assert _state(caches) == before
