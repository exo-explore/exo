import io

import pytest

from exo.worker.engines.mlx.disaggregated.protocol import (
    ArraysState,
    Done,
    ErrorMessage,
    KVChunk,
    ProtocolError,
    TensorBlob,
    dtype_size,
    header_dtype,
    header_int,
    make_header,
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
    hdr = make_header(
        num_layers=32, dtype="bfloat16", model_id="m", request_id="r", start_pos=42
    )
    buf = io.BytesIO()
    write_header(buf, hdr)
    buf.seek(0)
    got = read_header(buf)
    assert got == hdr
    assert header_dtype(got) == "bfloat16"
    assert header_int(got, "num_layers") == 32
    assert header_int(got, "start_pos") == 42


def test_header_dtype_rejects_unknown() -> None:
    with pytest.raises(ProtocolError):
        header_dtype({"dtype": "int4"})


def test_kv_chunk_roundtrip() -> None:
    dtype = "bfloat16"
    num_tokens, n_heads, head_dim = 7, 4, 8
    n_bytes = num_tokens * n_heads * head_dim * dtype_size(dtype)
    keys = _mk_bytes(n_bytes)
    values = _mk_bytes(n_bytes)[::-1]

    buf = io.BytesIO()
    write_kv_chunk(
        buf,
        layer_idx=3,
        num_tokens=num_tokens,
        n_heads=n_heads,
        head_dim=head_dim,
        keys=keys,
        values=values,
    )
    buf.seek(0)
    msg = read_message(buf, make_header(num_layers=1, dtype=dtype))
    assert isinstance(msg, KVChunk)
    assert msg.layer_idx == 3
    assert msg.shape == (num_tokens, n_heads, head_dim)
    assert msg.dtype == dtype
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
    msg = read_message(buf, make_header(num_layers=1, dtype="float32"))
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
    msg = read_message(buf, make_header(num_layers=1, dtype="float32"))
    assert isinstance(msg, Done)
    assert msg.total_tokens == 1234


def test_error_roundtrip() -> None:
    buf = io.BytesIO()
    write_error(buf, code=42, message="boom")
    buf.seek(0)
    msg = read_message(buf, make_header(num_layers=1, dtype="float32"))
    assert isinstance(msg, ErrorMessage)
    assert msg.code == 42
    assert msg.message == "boom"


def test_stream_of_messages() -> None:
    hdr = make_header(num_layers=2, dtype="float32")
    buf = io.BytesIO()
    write_header(buf, hdr)
    write_kv_chunk(
        buf,
        layer_idx=0,
        num_tokens=1,
        n_heads=1,
        head_dim=2,
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

    m1 = read_message(buf, got_hdr)
    m2 = read_message(buf, got_hdr)
    m3 = read_message(buf, got_hdr)
    m4 = read_message(buf, got_hdr)
    assert isinstance(m1, KVChunk)
    assert isinstance(m2, ArraysState)
    assert isinstance(m3, Done)
    assert m4 is None


def test_unknown_message_type_raises() -> None:
    # Construct a stream with a valid header and then an unknown tag.
    buf = io.BytesIO()
    write_header(buf, make_header(num_layers=1, dtype="float32"))
    buf.write(bytes([0xFF]))
    buf.seek(0)
    hdr = read_header(buf)
    with pytest.raises(ProtocolError):
        _ = read_message(buf, hdr)
