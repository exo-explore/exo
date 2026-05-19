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


class KVChunk(msgspec.Struct, tag="kv_chunk"):
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


Message = KVChunk | ArraysState | Done | ErrorMessage

_msg_encoder = msgspec.msgpack.Encoder()
_msg_decoder: msgspec.msgpack.Decoder[Message] = msgspec.msgpack.Decoder(Message)
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


def write_message(stream: BinaryIO, msg: Message) -> None:
    write_frame(stream, _msg_encoder.encode(msg))


def read_message(stream: BinaryIO) -> Message | None:
    payload = read_frame(stream)
    if not payload:
        return None
    try:
        return _msg_decoder.decode(payload)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"Bad message: {exc}") from exc


def write_kv_chunk(
    stream: BinaryIO,
    *,
    layer_idx: int,
    num_tokens: int,
    n_heads: int,
    head_dim: int,
    dtype: DType,
    keys: bytes,
    values: bytes,
) -> None:
    write_message(
        stream,
        KVChunk(
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            n_heads=n_heads,
            head_dim=head_dim,
            dtype=dtype,
            keys=keys,
            values=values,
        ),
    )


def write_arrays_state(
    stream: BinaryIO, layer_idx: int, arrays: list[TensorBlob]
) -> None:
    write_message(stream, ArraysState(layer_idx=layer_idx, arrays=arrays))


def write_done(stream: BinaryIO, total_tokens: int) -> None:
    write_message(stream, Done(total_tokens=total_tokens))


def write_error(stream: BinaryIO, code: int, message: str) -> None:
    write_message(stream, ErrorMessage(code=code, message=message))
