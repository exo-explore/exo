import io
import json
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, cast

FORMAT_VERSION: int = 1

MSG_KV_CHUNK: int = 0x01
MSG_ARRAYS_STATE: int = 0x02
MSG_DONE: int = 0x03
MSG_ERROR: int = 0x04

DType = str
Layout = str

_DTYPE_SIZE: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


def dtype_size(dtype: DType) -> int:
    return _DTYPE_SIZE[dtype]


class ProtocolError(Exception):
    pass


@dataclass
class KVChunk:
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


@dataclass
class TensorBlob:
    dtype: DType
    shape: tuple[int, ...]
    data: bytes


@dataclass
class ArraysState:
    layer_idx: int
    arrays: list[TensorBlob] = field(default_factory=list[TensorBlob])


@dataclass
class Done:
    total_tokens: int


@dataclass
class ErrorMessage:
    code: int
    message: str


Message = KVChunk | ArraysState | Done | ErrorMessage


def _write_exactly(stream: BinaryIO, data: bytes) -> None:
    stream.write(data)
    stream.flush()


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


def _unpack(fmt: str, data: bytes) -> tuple[int, ...]:
    return cast(tuple[int, ...], struct.unpack(fmt, data))


def make_header(
    *,
    num_layers: int,
    dtype: DType,
    model_id: str = "",
    request_id: str = "",
    start_pos: int = 0,
    layout: Layout = "NHD",
    format_version: int = FORMAT_VERSION,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    h: dict[str, object] = {
        "format_version": format_version,
        "request_id": request_id,
        "model_id": model_id,
        "num_layers": num_layers,
        "dtype": dtype,
        "layout": layout,
        "start_pos": start_pos,
    }
    if extra:
        h.update(extra)
    return h


def write_header(stream: BinaryIO, header: dict[str, object]) -> None:
    payload = json.dumps(header).encode("utf-8")
    _write_exactly(stream, struct.pack(">I", len(payload)))
    _write_exactly(stream, payload)


def read_header(stream: BinaryIO) -> dict[str, object]:
    raw = _read_exactly(stream, 4)
    if not raw:
        raise ConnectionError("No header received")
    (length,) = _unpack(">I", raw)
    payload = _read_exactly(stream, length)
    return cast(dict[str, object], json.loads(payload.decode("utf-8")))


def header_dtype(header: dict[str, object]) -> DType:
    dtype = header.get("dtype")
    if not isinstance(dtype, str):
        raise ProtocolError(f"Missing or non-string 'dtype' in header: {dtype!r}")
    if dtype not in _DTYPE_SIZE:
        raise ProtocolError(f"Unsupported dtype on wire: {dtype!r}")
    return dtype


def header_int(header: dict[str, object], key: str, default: int = 0) -> int:
    v = header.get(key, default)
    if not isinstance(v, int):
        raise ProtocolError(f"Expected int for header[{key!r}], got {type(v).__name__}")
    return v


def write_kv_chunk(
    stream: BinaryIO,
    layer_idx: int,
    num_tokens: int,
    n_heads: int,
    head_dim: int,
    keys: bytes,
    values: bytes,
) -> None:
    header = struct.pack(
        ">BIIII", MSG_KV_CHUNK, layer_idx, num_tokens, n_heads, head_dim
    )
    _write_exactly(stream, header + keys + values)


def write_arrays_state(
    stream: BinaryIO, layer_idx: int, arrays: list[TensorBlob]
) -> None:
    buf = io.BytesIO()
    buf.write(struct.pack(">BI", MSG_ARRAYS_STATE, layer_idx))
    buf.write(struct.pack(">I", len(arrays)))
    for arr in arrays:
        dtype_bytes = arr.dtype.encode("utf-8")
        buf.write(struct.pack(">I", len(dtype_bytes)))
        buf.write(dtype_bytes)
        buf.write(struct.pack(">I", len(arr.shape)))
        for dim in arr.shape:
            buf.write(struct.pack(">I", dim))
        buf.write(arr.data)
    _write_exactly(stream, buf.getvalue())


def write_done(stream: BinaryIO, total_tokens: int) -> None:
    _write_exactly(stream, struct.pack(">BI", MSG_DONE, total_tokens))


def write_error(stream: BinaryIO, code: int, message: str) -> None:
    msg_bytes = message.encode("utf-8")
    _write_exactly(
        stream, struct.pack(">BII", MSG_ERROR, code, len(msg_bytes)) + msg_bytes
    )


def read_message(stream: BinaryIO, header: dict[str, object]) -> Message | None:
    type_byte = _read_exactly(stream, 1)
    if not type_byte:
        return None
    msg_type = type_byte[0]

    if msg_type == MSG_KV_CHUNK:
        layer_idx, num_tokens, n_heads, head_dim = _unpack(
            ">IIII", _read_exactly(stream, 16)
        )
        dtype = header_dtype(header)
        tensor_bytes = num_tokens * n_heads * head_dim * dtype_size(dtype)
        keys = _read_exactly(stream, tensor_bytes)
        values = _read_exactly(stream, tensor_bytes)
        return KVChunk(
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            n_heads=n_heads,
            head_dim=head_dim,
            dtype=dtype,
            keys=keys,
            values=values,
        )

    if msg_type == MSG_ARRAYS_STATE:
        (arr_layer_idx,) = _unpack(">I", _read_exactly(stream, 4))
        (num_arrays,) = _unpack(">I", _read_exactly(stream, 4))
        fallback_dtype = header_dtype(header)
        arrays: list[TensorBlob] = []
        for _ in range(num_arrays):
            (dtype_len,) = _unpack(">I", _read_exactly(stream, 4))
            if 0 < dtype_len < 20:
                arr_dtype = _read_exactly(stream, dtype_len).decode("utf-8")
                if arr_dtype not in _DTYPE_SIZE:
                    raise ProtocolError(f"Unsupported arrays dtype: {arr_dtype!r}")
            else:
                arr_dtype = fallback_dtype
            (ndim,) = _unpack(">I", _read_exactly(stream, 4))
            shape = _unpack(f">{ndim}I", _read_exactly(stream, ndim * 4))
            total_elems = 1
            for d in shape:
                total_elems *= d
            data = _read_exactly(stream, total_elems * dtype_size(arr_dtype))
            arrays.append(TensorBlob(dtype=arr_dtype, shape=shape, data=data))
        return ArraysState(layer_idx=arr_layer_idx, arrays=arrays)

    if msg_type == MSG_DONE:
        (total_tokens,) = _unpack(">I", _read_exactly(stream, 4))
        return Done(total_tokens=total_tokens)

    if msg_type == MSG_ERROR:
        code, msg_len = _unpack(">II", _read_exactly(stream, 8))
        msg_bytes = _read_exactly(stream, msg_len)
        return ErrorMessage(code=code, message=msg_bytes.decode("utf-8", "replace"))

    raise ProtocolError(f"Unknown message type: {msg_type:#x}")
