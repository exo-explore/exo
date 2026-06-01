import io

import pytest

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    ErrorMessage,
    Header,
    KVChunk,
    ProtocolError,
    TensorBlob,
    read_header,
    read_message,
    write_arrays_state,
    write_done,
    write_error,
    write_header,
    write_kv_chunk,
)


def _mk_bytes(n: int) -> bytes:
    return bytes(i & 0xFF for i in range(n))


def test_header_roundtrip() -> None:
    hdr = Header(
        request_id="r",
        model_id="m",
        num_layers=32,
        dtype="bfloat16",
        start_pos=42,
    )
    buf = io.BytesIO()
    write_header(buf, hdr)
    buf.seek(0)
    got = read_header(buf)
    assert got == hdr
    assert got.dtype == "bfloat16"
    assert got.num_layers == 32
    assert got.start_pos == 42


def test_kv_chunk_roundtrip() -> None:
    num_tokens, n_heads, head_dim = 7, 4, 8
    n_bytes = num_tokens * n_heads * head_dim * 2
    keys = _mk_bytes(n_bytes)
    values = _mk_bytes(n_bytes)[::-1]

    buf = io.BytesIO()
    write_kv_chunk(
        buf,
        layer_idx=3,
        num_tokens=num_tokens,
        n_heads=n_heads,
        head_dim=head_dim,
        dtype="bfloat16",
        keys=keys,
        values=values,
    )
    buf.seek(0)
    msg = read_message(buf)
    assert isinstance(msg, KVChunk)
    assert msg.layer_idx == 3
    assert msg.shape == (num_tokens, n_heads, head_dim)
    assert msg.dtype == "bfloat16"
    assert msg.keys == keys
    assert msg.values == values


def test_arrays_state_roundtrip() -> None:
    arrs = [
        TensorBlob(dtype="float32", shape=(2, 3), data=_mk_bytes(2 * 3 * 4)),
        TensorBlob(dtype="bfloat16", shape=(5,), data=_mk_bytes(5 * 2)),
    ]
    buf = io.BytesIO()
    write_arrays_state(buf, layer_idx=9, arrays=arrs)
    buf.seek(0)
    msg = read_message(buf)
    assert isinstance(msg, ArraysState)
    assert msg.layer_idx == 9
    assert len(msg.arrays) == 2
    assert msg.arrays[0].dtype == "float32"
    assert msg.arrays[0].shape == (2, 3)
    assert msg.arrays[0].data == arrs[0].data
    assert msg.arrays[1].dtype == "bfloat16"
    assert msg.arrays[1].shape == (5,)
    assert msg.arrays[1].data == arrs[1].data


def test_done_roundtrip() -> None:
    buf = io.BytesIO()
    write_done(buf, 1234)
    buf.seek(0)
    msg = read_message(buf)
    assert isinstance(msg, Done)
    assert msg.total_tokens == 1234


def test_error_roundtrip() -> None:
    buf = io.BytesIO()
    write_error(buf, code=42, message="boom")
    buf.seek(0)
    msg = read_message(buf)
    assert isinstance(msg, ErrorMessage)
    assert msg.code == 42
    assert msg.message == "boom"


def test_stream_of_messages() -> None:
    hdr = Header(num_layers=2, dtype="float32")
    buf = io.BytesIO()
    write_header(buf, hdr)
    write_kv_chunk(
        buf,
        layer_idx=0,
        num_tokens=1,
        n_heads=1,
        head_dim=2,
        dtype="float32",
        keys=_mk_bytes(1 * 1 * 2 * 4),
        values=_mk_bytes(1 * 1 * 2 * 4),
    )
    write_arrays_state(
        buf,
        layer_idx=1,
        arrays=[TensorBlob(dtype="float32", shape=(1,), data=_mk_bytes(4))],
    )
    write_done(buf, total_tokens=1)
    buf.seek(0)

    got_hdr = read_header(buf)
    assert got_hdr == hdr

    m1 = read_message(buf)
    m2 = read_message(buf)
    m3 = read_message(buf)
    m4 = read_message(buf)
    assert isinstance(m1, KVChunk)
    assert isinstance(m2, ArraysState)
    assert isinstance(m3, Done)
    assert m4 is None


def test_corrupt_message_raises() -> None:
    buf = io.BytesIO()
    write_header(buf, Header(num_layers=1, dtype="float32"))
    buf.write((5).to_bytes(4, "big"))
    buf.write(b"\xff\xff\xff\xff\xff")
    buf.seek(0)
    _ = read_header(buf)
    with pytest.raises(ProtocolError):
        _ = read_message(buf)
