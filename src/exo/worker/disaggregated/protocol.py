from dataclasses import dataclass
from typing import BinaryIO, Literal

import msgspec

DType = Literal["bfloat16", "float16", "float32"]


class ProtocolError(Exception):
    pass


class Header(msgspec.Struct):
    request_id: str = ""
    model_id: str = ""
    num_layers: int = 0
    dtype: DType = "bfloat16"
    start_pos: int = 0


class TensorBlob(msgspec.Struct):
    dtype: DType
    shape: tuple[int, ...]
    data: bytes


class _KVChunkHeader(msgspec.Struct, tag="kv_chunk"):
    """Wire-side KV chunk metadata. Raw `keys` then `values` bytes follow on
    the stream, lengths given by `keys_len` / `values_len`. Splitting them out
    of the msgpack frame lets the producer pass tensor buffers via the buffer
    protocol straight into the socket (one host-side memcpy total).
    """

    layer_idx: int
    num_tokens: int
    n_heads: int
    head_dim: int
    dtype: DType
    keys_len: int
    values_len: int


@dataclass(frozen=True)
class KVChunk:
    """In-memory KV chunk reconstructed by `read_message` from
    `_KVChunkHeader` + the raw bytes that follow on the wire.
    """

    layer_idx: int
    num_tokens: int
    n_heads: int
    head_dim: int
    dtype: DType
    keys: bytes
    values: bytes

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_tokens, self.n_heads, self.head_dim)


class ArraysState(msgspec.Struct, tag="arrays_state"):
    layer_idx: int
    arrays: list[TensorBlob] = []


class Done(msgspec.Struct, tag="done"):
    total_tokens: int


class ErrorMessage(msgspec.Struct, tag="error"):
    code: int
    message: str


_WireMessage = _KVChunkHeader | ArraysState | Done | ErrorMessage
Message = KVChunk | ArraysState | Done | ErrorMessage

_msg_encoder = msgspec.msgpack.Encoder()
_msg_decoder: msgspec.msgpack.Decoder[_WireMessage] = msgspec.msgpack.Decoder(
    _WireMessage
)
_header_encoder = msgspec.msgpack.Encoder()
_header_decoder: msgspec.msgpack.Decoder[Header] = msgspec.msgpack.Decoder(Header)


def _read_exactly(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if len(buf) == 0:
                return b""
            raise ConnectionError(f"Connection closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    stream.write(len(payload).to_bytes(4, "big"))
    stream.write(payload)
    stream.flush()


def read_frame(stream: BinaryIO) -> bytes:
    raw = _read_exactly(stream, 4)
    if not raw:
        return b""
    length = int.from_bytes(raw, "big")
    return _read_exactly(stream, length)


def write_header(stream: BinaryIO, header: Header) -> None:
    write_frame(stream, _header_encoder.encode(header))


def read_header(stream: BinaryIO) -> Header:
    payload = read_frame(stream)
    if not payload:
        raise ConnectionError("No header received")
    try:
        return _header_decoder.decode(payload)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"Bad header: {exc}") from exc


def write_message(stream: BinaryIO, msg: _WireMessage) -> None:
    write_frame(stream, _msg_encoder.encode(msg))


def read_message(stream: BinaryIO) -> Message | None:
    payload = read_frame(stream)
    if not payload:
        return None
    try:
        msg = _msg_decoder.decode(payload)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"Bad message: {exc}") from exc
    if isinstance(msg, _KVChunkHeader):
        keys = _read_exactly(stream, msg.keys_len)
        values = _read_exactly(stream, msg.values_len)
        return KVChunk(
            layer_idx=msg.layer_idx,
            num_tokens=msg.num_tokens,
            n_heads=msg.n_heads,
            head_dim=msg.head_dim,
            dtype=msg.dtype,
            keys=keys,
            values=values,
        )
    return msg


def write_kv_chunk(
    stream: BinaryIO,
    *,
    layer_idx: int,
    num_tokens: int,
    n_heads: int,
    head_dim: int,
    dtype: DType,
    keys: bytes | memoryview,
    values: bytes | memoryview,
) -> None:
    """Stream KV chunk metadata + raw key/value bytes to the wire.

    `keys` / `values` may be bytes-like (bytes, bytearray, memoryview) — the
    raw payload is written directly to the buffered stream after the
    msgpack-framed header, avoiding a memcpy through the msgpack encoder.
    """
    keys_len = len(keys)
    values_len = len(values)
    header_payload = _msg_encoder.encode(
        _KVChunkHeader(
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            n_heads=n_heads,
            head_dim=head_dim,
            dtype=dtype,
            keys_len=keys_len,
            values_len=values_len,
        )
    )
    stream.write(len(header_payload).to_bytes(4, "big"))
    stream.write(header_payload)
    stream.write(keys)
    stream.write(values)
    # No per-chunk flush: the K/V payload is far larger than the
    # BufferedWriter's internal buffer so it bypasses to the socket directly.
    # The trailing `Done` frame's `write_frame` flushes once at the end.


def write_arrays_state(
    stream: BinaryIO, layer_idx: int, arrays: list[TensorBlob]
) -> None:
    write_message(stream, ArraysState(layer_idx=layer_idx, arrays=arrays))


def write_done(stream: BinaryIO, total_tokens: int) -> None:
    write_message(stream, Done(total_tokens=total_tokens))


def write_error(stream: BinaryIO, code: int, message: str) -> None:
    write_message(stream, ErrorMessage(code=code, message=message))
